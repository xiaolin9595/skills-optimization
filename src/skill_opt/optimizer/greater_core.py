import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import gc
import random
import logging

logger = logging.getLogger(__name__)

def propose_candidates(
    model, 
    tokenizer,
    prompt_ids: torch.Tensor, # Not used directly if we trust input_ids_batch context construction? 
                              # Actually, we need prompt_ids to know *what* to optimize if separate.
                              # But here input_ids_batch should contain the prompt.
                              # Let's clarify: input_ids_batch = [Prompt+Input] for each sample?
                              # Yes, let's assume input_ids_batch is the full context for Stage 1.
    input_ids_batch: torch.Tensor, 
    position_idx: Union[int, List[int]],
    top_k: int = 50,
    num_intersections: int = 3,
    intersection_across_examples: int = 2,
    allow_non_ascii: bool = False,
    not_allowed_tokens: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Stage 1: Candidate Proposal
    Calculates logits for the target position across the batch and finds candidates using
    GreaTer's "Union of Intersections" strategy (Logic Intersection).
    
    Args:
        input_ids_batch: Batch of input contexts (Prompt + Input). 
                         Target position `position_idx` is where we want to propose replacements.
        position_idx: The index in input_ids_batch corresponding to the token being optimized.
        num_intersections: Number of random subsets to intersect.
        intersection_across_examples: Size of each random subset.
        not_allowed_tokens: Tensor of token IDs to suppress (set to -inf).
    """
    if device is None:
        device = model.device
        
    # Helper to clean up memory
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    lm_topks = []
    
    batch_size = input_ids_batch.shape[0]
    
    # Iterate over batch to get logits
    # We can do this in mini-batches if batch_size is large, but usually < 16.
    
    # Loop one by one or batch? Reference loops one by one (workers).
    # We can run batch forward if memory allows.
    
    # For GreaTer logic, we need independent top-k sets for each sample.
    
    # We'll batch it for efficiency on GPU.
    with torch.no_grad():
        outputs = model(input_ids_batch.to(device))
        logits = outputs.logits # [Batch, Seq, Vocab]
        
        # Logits at pos-1 predict pos
        # input_ids_batch contains the CURRENT token at position_idx.
        # We want to propose replacements for it.
        # So we look at logits
        if isinstance(position_idx, int):
             # Broadcast same position
             target_logits = logits[:, position_idx - 1, :]
        else:
             # Tensor indexing: [Batch, Pos[i]-1, :]
             # Ensure list length matches batch
             if len(position_idx) != batch_size:
                 raise ValueError("Length of position_idx list must match batch size")
             
             # Create batch indices [0, 1, 2...]
             batch_indices = torch.arange(batch_size, device=device)
             pos_indices = torch.tensor(position_idx, device=device) - 1
             
             target_logits = logits[batch_indices, pos_indices, :]
             
        if not allow_non_ascii and not_allowed_tokens is not None:
             target_logits[:, not_allowed_tokens] = -float('inf')
        elif not allow_non_ascii:
             # If no tokens provided, we can't filter effectively without expensive check
             # Assume caller provides it for efficiency.
             pass
             
        # Get Top Indicies for each sample
        # topk_indices: [Batch, K]
        _, topk_indices = torch.topk(target_logits, top_k, dim=1)
        
        lm_topks = [indices.tolist() for indices in topk_indices]
        
    cleanup()
    
    # Union of Intersections Strategy
    # If batch size is small, we might just intersect all.
    # Ref logic:
    # random_indices = random.sample(range(len(lm_topks)), intersection_across_examples)
    
    if batch_size < intersection_across_examples:
        # Fallback: Just intersect all available
        intersection = set(lm_topks[0])
        for s in lm_topks[1:]:
            intersection &= set(s)
        return torch.tensor(list(intersection), dtype=torch.long, device=device)
        
    all_intersections = []
    for _ in range(num_intersections):
        # Sample subset
        subset_indices = random.sample(range(batch_size), intersection_across_examples)
        
        # Intersect this subset
        intersection = set(lm_topks[subset_indices[0]])
        for idx in subset_indices[1:]:
            intersection &= set(lm_topks[idx])
            
        all_intersections.append(intersection)
        
    # Union
    final_candidates = set().union(*all_intersections)
    
    if not final_candidates:
        # Fallback if empty: return Union of all? Or Top-K of first?
        # Reference increases Top-K. We'll just return Top-K of first sample to avoid crash.
        logger.warning("Intersection of candidates resulted in empty set. Falling back to Top-K of first sample.")
        return topk_indices[0]
    
    cand_tensor = torch.tensor(list(final_candidates), dtype=torch.long, device=device)
    
    # Trace log: sample some candidates
    sample_size = min(len(cand_tensor), 5)
    sample_text = tokenizer.decode(cand_tensor[:sample_size])
    logger.info(f"Proposed {len(cand_tensor)} candidates. Sample: [{sample_text}...]")
    
    return cand_tensor

def generate_reasoning(
    model, 
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    decode: bool = True,
    return_only_new: bool = True,
    device: Optional[torch.device] = None
) -> Union[List[str], torch.Tensor]:
    """
    Stage 2: Reasoning Generation
    Generates a reasoning chain (CoT) given the input context.
    
    Args:
        input_ids: Batch of inputs [Batch, Seq] or [Seq].
        decode: If True, returns List[str] of generated text (SKIP special tokens).
        return_only_new: If True, returns only generated tokens.
        
    Returns:
        If decode=True: List[str] - generated texts for each sample in batch
        If decode=False: torch.Tensor - generated token IDs [Batch, NewLen]
    """
    if device is None:
        device = model.device
        
    # Input Normalization
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    input_ids = input_ids.to(device)
    input_len = input_ids.shape[1]
    
    with torch.no_grad():
        # Generate
        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=temperature,
            top_k=top_k
        )
        
    # Process Output
    if return_only_new:
        generated_ids = outputs[:, input_len:]
    else:
        generated_ids = outputs
        
    if decode:
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_ids

# ... get_embedding helper ...

def get_embedding_matrix(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight
    elif hasattr(model, "transformer"): # GPT-2 / GPT-J
        return model.transformer.wte.weight
    elif hasattr(model, "model"): # Llama / Mistral
        return model.model.embed_tokens.weight
    else:
        raise ValueError("Could not find embedding matrix")

def get_embeddings(model, input_ids):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()(input_ids)
    else:
        # Fallback
        embeds = model(input_ids).inputs_embeds
        return embeds
        
def compute_gradient(
    model,
    context_ids_with_gt: torch.Tensor,
    prompt_pos_idx: int,
    loss_slice: slice,
    candidates: torch.Tensor,
    control_slice: Optional[slice] = None,
    control_weight: float = 0.0,
    focused_target: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    grad_clip: float = 1.0,
    use_amp: bool = False
) -> torch.Tensor:
    """
    Stage 3: Gradient Calculation
    Backpropagates loss to find gradients for the candidates.
    Implements sparse embedding replacement: One-Hot(Candidates) @ Embeddings.
    
    Args:
        grad_clip: Gradient clipping threshold for numerical stability
        use_amp: Whether to use automatic mixed precision
    """
    if device is None:
        device = model.device
        
    # Enable mixed precision if requested
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    # 1. Prepare Embeddings
    embed_weights = get_embedding_matrix(model)
    
    # Restrict to candidates (Sparse)
    candidate_weights = embed_weights[candidates] # [K, Dim]
    
    # 2. Create One-Hot
    current_token_id = context_ids_with_gt[prompt_pos_idx].item()
    
    one_hot = torch.zeros(1, len(candidates), device=device, dtype=candidate_weights.dtype)
    
    # Find index of current token in candidates
    matches = (candidates == current_token_id).nonzero(as_tuple=True)[0]
    if len(matches) > 0:
        one_hot[0, matches[0]] = 1.0
        
    one_hot.requires_grad_()
    
    # 3. Compute Embeddings with Replacement
    sparse_embed = (one_hot @ candidate_weights).unsqueeze(0) 
    
    # Check for NaN or Inf in sparse_embed
    if torch.isnan(sparse_embed).any() or torch.isinf(sparse_embed).any():
        logger.warning("Sparse embeddings contain NaN or Inf values. Skipping gradient computation.")
        return torch.zeros_like(one_hot)
    
    # Get full embeddings for context
    full_embeds_static = get_embeddings(model, context_ids_with_gt.unsqueeze(0)).detach()
    
    # Check for NaN or Inf in static embeddings
    if torch.isnan(full_embeds_static).any() or torch.isinf(full_embeds_static).any():
        logger.warning("Static embeddings contain NaN or Inf values. Skipping gradient computation.")
        return torch.zeros_like(one_hot)
    
    # Stitch
    full_embeds = torch.cat([
        full_embeds_static[:, :prompt_pos_idx, :],
        sparse_embed,
        full_embeds_static[:, prompt_pos_idx+1:, :]
    ], dim=1)
    
    # Check for NaN or Inf in full embeddings
    if torch.isnan(full_embeds).any() or torch.isinf(full_embeds).any():
        logger.warning("Full embeddings contain NaN or Inf values after concatenation. Skipping gradient computation.")
        return torch.zeros_like(one_hot)
    
    # 4. Forward Pass with optional AMP
    try:
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs_embeds=full_embeds)
                logits = outputs.logits # [1, SeqLen, Vocab]
        else:
            outputs = model(inputs_embeds=full_embeds)
            logits = outputs.logits # [1, SeqLen, Vocab]
    except Exception as e:
        logger.warning(f"Forward pass failed with error: {e}. Skipping gradient computation.")
        return torch.zeros_like(one_hot)
    
    # Check for NaN or Inf in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.warning(f"Logits contain NaN or Inf values. Shape: {logits.shape}, Max: {logits.max().item():.4f}, Min: {logits.min().item():.4f}. Skipping loss computation.")
        return torch.zeros_like(one_hot)
    
    loss_fct = nn.CrossEntropyLoss()
    
    # 5. Target Loss (Standard vs Focused)
    
    # Debug: print loss_slice and sequence info
    seq_len = logits.shape[1]
    context_len = context_ids_with_gt.shape[0]
    logger.debug(f"Loss computation: loss_slice={loss_slice}, seq_len={seq_len}, context_len={context_len}")
    
    # Check if loss_slice is empty (start == stop)
    if loss_slice.start == loss_slice.stop:
        logger.warning(f"Empty loss_slice: {loss_slice}. Using full sequence for loss computation.")
        loss_slice = slice(1, seq_len)  # Use full sequence except first token
    
    if focused_target is not None:
        # Focused Loss Logic
        # We search for the best window in the generated text that matches focused_target
        # loss_slice covers the relevant generated text (Reasoning + GT potentially)
        # Logits: [1, SeqLen, Vocab]
        
        # Relevant logits: prediction for loss_slice tokens
        # logits[i] predicts token at [i+1]
        # We care about what was predicted within loss_slice
        
        # Validate loss_slice indices
        if loss_slice.start < 1 or loss_slice.stop > seq_len + 1:
            logger.warning(f"Invalid loss_slice: {loss_slice} for sequence length {seq_len}. Using fallback.")
            shift_logits = logits[0, 0:-1, :].contiguous()
            shift_labels = context_ids_with_gt[1:].to(device)
            total_loss = loss_fct(shift_logits, shift_labels)
        else:
            loss_logits = logits[:, loss_slice.start-1 : loss_slice.stop-1, :] # [1, T, V]
            
            # Check if loss_logits is empty
            if loss_logits.shape[1] == 0:
                logger.warning(f"Empty loss_logits after slicing. Using full sequence.")
                shift_logits = logits[0, 0:-1, :].contiguous()
                shift_labels = context_ids_with_gt[1:].to(device)
                total_loss = loss_fct(shift_logits, shift_labels)
            else:
                window_size = len(focused_target)
                
                if loss_logits.shape[1] < window_size:
                    # Fallback to standard Loss on GT if generated text too short
                    shift_logits = logits[0, loss_slice.start-1 : loss_slice.stop-1, :].contiguous()
                    shift_labels = context_ids_with_gt[loss_slice].to(device)
                    total_loss = loss_fct(shift_logits, shift_labels)
                else:
                    # Unfold: [1, T-W+1, V, W]
                    unfolded = loss_logits.unfold(1, window_size, 1)
                    # Transpose to [1, T-W+1, W, V] so last dim is Vocab for CrossEntropy
                    unfolded = unfolded.transpose(2, 3) # [1, Windows, WindowSize, Vocab]
                    
                    # Flatten Windows for Batch Processing or iterate
                    # Shapes:
                    # Prediction: [Windows * WindowSize, Vocab]
                    # Target: focused_target repeated for each window
                    
                    num_windows = unfolded.shape[1]
                    flat_preds = unfolded.reshape(-1, unfolded.shape[-1]) # [N_Win * W, V]
                    
                    flat_target = focused_target.repeat(num_windows).to(device) # [N_Win * W]
                    
                    # Compute loss per Token matching
                    # reduction='none' -> [N_Win * W]
                    losses = nn.CrossEntropyLoss(reduction='none')(flat_preds, flat_target)
                    
                    # Reshape to [N_Win, W]
                    losses = losses.view(num_windows, window_size)
                    
                    # Mean loss per window
                    window_losses = losses.mean(dim=1)
                    
                    # Select MINIMUM loss window (best match)
                    min_loss_val, min_idx = window_losses.min(dim=0)
                    total_loss = min_loss_val
                    
                    # Diagnostic: where did it find the answer?
                    logger.debug(f"Focused Loss: Best match found at window {min_idx.item()} with loss {min_loss_val.item():.4f}")
            
    else:
        # Standard Next Token Prediction on GT
        # Validate loss_slice indices
        if loss_slice.start < 1 or loss_slice.stop > seq_len + 1:
            logger.warning(f"Invalid loss_slice: {loss_slice} for sequence length {seq_len}. Using fallback.")
            shift_logits = logits[0, 0:-1, :].contiguous()
            shift_labels = context_ids_with_gt[1:].to(device)
            total_loss = loss_fct(shift_logits, shift_labels)
        else:
            shift_logits = logits[0, loss_slice.start-1 : loss_slice.stop-1, :].contiguous()
            
            # Check if shift_logits is empty
            if shift_logits.shape[0] == 0:
                logger.warning(f"Empty shift_logits. Using full sequence.")
                shift_logits = logits[0, 0:-1, :].contiguous()
                shift_labels = context_ids_with_gt[1:].to(device)
            else:
                shift_labels = context_ids_with_gt[loss_slice].to(device)
            
            total_loss = loss_fct(shift_logits, shift_labels)
            logger.debug(f"Standard Target Loss: {total_loss.item():.4f}")
    
    # 6. Control Loss
    if control_weight > 0 and control_slice is not None:
        if control_slice.start > 0:
            c_logits = logits[0, control_slice.start-1 : control_slice.stop-1, :].contiguous()
            c_labels = context_ids_with_gt[control_slice].to(device)
            c_loss = loss_fct(c_logits, c_labels)
            total_loss += control_weight * c_loss
    
    # 7. Backward with gradient clipping and stability checks
    # Check for NaN or Inf in loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"Loss is NaN or Inf: {total_loss.item()}. Skipping backward pass.")
        return torch.zeros_like(one_hot)
    
    # Clear previous gradients
    if one_hot.grad is not None:
        one_hot.grad.zero_()
    
    # Backward pass
    if use_amp and scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
    
    # Get gradient
    grad = one_hot.grad
    
    # Check for NaN or Inf in gradient
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        logger.warning("Gradient contains NaN or Inf values. Replacing with zeros.")
        grad = torch.zeros_like(grad)
    else:
        # Improved gradient normalization with epsilon for numerical stability
        grad_norm = grad.norm(dim=-1, keepdim=True)
        epsilon = 1e-6  # Small epsilon to avoid division by zero
        
        # Normalize gradient
        normalized_grad = grad / (grad_norm + epsilon)
        
        # Gradient clipping
        if grad_clip > 0:
            normalized_grad = torch.clamp(normalized_grad, -grad_clip, grad_clip)
        
        # Replace gradient with normalized version
        grad = normalized_grad
        
        logger.debug(f"Gradient stats: Mean={grad.mean().item():.6f}, Std={grad.std().item():.6f}, "
                    f"Max={grad.abs().max().item():.6f}, Norm={grad_norm.item():.6f}")
    
    return grad.clone()

def select_and_update(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    position_idx: int,
    candidates: torch.Tensor,
    gradients: torch.Tensor,
    validation_input_ids: torch.Tensor,
    validation_label_ids: torch.Tensor,
    top_mu: int = 5,
    extract_prompt_ids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, float]:
    """
    Stage 4: Selection and Update
    Selects the best candidate based on gradient and forward pass validation.
    
    Args:
        prompt_ids: Current prompt.
        position_idx: Token position to update.
        candidates: Candidate tokens.
        gradients: Gradients for candidates (from Stage 3).
        validation_input_ids: Input for validation (single sample or batch).
        validation_label_ids: Ground truth labels.
        top_mu: Number of candidates to validate.
        
    Returns:
        (best_prompt_ids, min_loss)
    """
    if device is None:
        device = model.device
        
    # 1. Filter Candidates by Gradient
    # We want candidates with most NEGATIVE gradient (reducing loss).
    # Gradients are shape [1, K].
    # Top-K negative gradients = Top-K smallest values.
    # actually, gradient descent moves in direction of -grad.
    # So if grad is negative, -grad is positive.
    # We want indices where gradient is SMALLEST (most strongly negative).
    
    # Ensure gradients same shape as candidates
    if gradients.shape[-1] != candidates.shape[0]:
         # This might happen if 1-hot logic was different.
         pass
         
    # Get top_mu candidates with lowest gradient values (most negative)
    # vals: [top_mu], sorted_indices: [top_mu]
    # topk returns LARGEST values, so we negate gradients to get largest MAGNITUDE negative values.
    
    grads_flat = gradients.squeeze()
    vals, sorted_indices = torch.topk(-grads_flat, top_mu) 
    
    selected_candidates = candidates[sorted_indices]
    
    # Log candidate stats
    logger.info(f"Top-{top_mu} Candidates for update: {tokenizer.decode(selected_candidates)}")
    logger.debug(f"Gradient stats: Min={grads_flat.min().item():.4f}, Max={grads_flat.max().item():.4f}, Mean={grads_flat.mean().item():.4f}")
    
    best_loss = float('inf')
    best_prompt = prompt_ids.clone()
    
    # Include current token as baseline? Usually yes.
    # For now iterate candidates.
    
    loss_fct = nn.CrossEntropyLoss()
    
    for cand_token in selected_candidates:
        # a. Construct Candidate Prompt
        cand_prompt = prompt_ids.clone()
        cand_prompt[position_idx] = cand_token
        
        # b. Generate Reasoning (Forward)
        # We need to run generation for the validation sample.
        # This is expensive, so usually done on small batch or single sample.
        # We assume validation_input_ids is a single sample or we handle batch.
        # For simplicity, treat as single sample.
        if validation_input_ids.dim() > 1:
             val_in = validation_input_ids[0]
             val_lbl = validation_label_ids[0]
        else:
             val_in = validation_input_ids
             val_lbl = validation_label_ids
             
        # Generate Reasoning Context
        # Manually construct context: [Prompt] [Input]
        context_ids = torch.cat([cand_prompt, val_in])
        
        # Ensure context_ids is 2D for batch processing
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)
        
        gen_ids = generate_reasoning(
            model, tokenizer, input_ids=context_ids, 
            decode=False, return_only_new=True, device=device
        )
        # gen_ids is [1, New] (batch=1)
        gen_ids = gen_ids[0]  # Extract the single sample 
        
        full_seq = torch.cat([context_ids.to(device), gen_ids.to(device)])
        
        if extract_prompt_ids is not None:
             full_seq = torch.cat([full_seq, extract_prompt_ids.to(device)])
        
        # c. Compute Loss on Ground Truth
        # We append Ground Truth to evaluate P(GT | Context)
        # Full = [Context] [GT]
        input_w_gt = torch.cat([full_seq, val_lbl.to(device)])
        
        # Forward pass on the GT part
        # We just need to evaluate loss on the last len(val_lbl) tokens.
        with torch.no_grad():
            outputs = model(input_w_gt.unsqueeze(0))
            logits = outputs.logits
            
            # Shift logits
            gt_len = len(val_lbl)
            # Logits indices: [SeqLen - GT_Len - 1 : SeqLen - 1]
            # to predict [SeqLen - GT_Len : SeqLen]
            
            start_idx = input_w_gt.shape[0] - gt_len - 1
            end_idx = input_w_gt.shape[0] - 1
            
            shift_logits = logits[0, start_idx:end_idx, :].contiguous()
            shift_labels = val_lbl.to(device)
            
            loss = loss_fct(shift_logits, shift_labels)
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_prompt = cand_prompt
            
    return best_prompt, best_loss

