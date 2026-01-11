from typing import List, Optional, Union
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
from skill_opt.core.interfaces import SkillOptimizer, Skill, OptimizedSkill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.utils import load_model_and_tokenizer
from skill_opt.optimizer import greater_core
from skill_opt.optimizer.prompter import SkillPrompter

logger = logging.getLogger(__name__)

class GreaterOptimizer(SkillOptimizer):
    def __init__(self, config: AppConfig):
        self.config = config
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        self.device = self.model.device
        logger.info(f"Initialized GreaterOptimizer with model on {self.device}")

    def optimize(self, skill: Skill, config: OptimizeConfig) -> OptimizedSkill:
        """
        Optimizes a skill using GreaTer 4-stage workflow with Refined Prompter.
        """
        logger.info(f"Starting optimization for skill: {skill.name}")
        
        from skill_opt.optimizer.utils import get_nonascii_toks
        not_allowed_tokens = get_nonascii_toks(self.tokenizer, device=self.device)
        
        # 1. Load Data
        train_pairs = self._load_training_data_raw(config)
        
        # 2. Control Initialization strategy
        # Sequential Increasing: Start with start_len
        current_control_text = skill.content
        
        # Enforce start_len logic by resizing if necessary
        # Tokenize current control to check length
        init_ids = self.tokenizer(current_control_text, add_special_tokens=False).input_ids
        
        if len(init_ids) != config.start_len:
            logger.warning(f"Initial control length ({len(init_ids)}) does not match start_len ({config.start_len}). Resizing...")
            if len(init_ids) > config.start_len:
                # Truncate
                init_ids = init_ids[:config.start_len]
                current_control_text = self.tokenizer.decode(init_ids, skip_special_tokens=True)
                logger.info(f"Truncated control to: {current_control_text}")
            else:
                # Pad
                diff = config.start_len - len(init_ids)
                current_control_text += " !" * diff
                logger.info(f"Padded control to: {current_control_text}")
        
        target_len = config.start_len
        
        best_loss = float('inf')
        loss_cache = {} # Key: (input_ids_tuple) -> loss
        
        patience_counter = 0
        
        for length_iter in range(config.start_len, config.end_len + 1):
             # Update prompters with current control text (potentially resized)
             # If this is the first iteration, we might need to initialize prompters
             # If length increased, we need to append token.
             
             if length_iter > config.start_len:
                 # Extend strategy: Append ONE placeholder token
                 current_control_text += " !" 
                 
                 # Verify length alignment
                 check_ids = self.tokenizer(current_control_text, add_special_tokens=False).input_ids
                 if len(check_ids) != length_iter:
                     # Attempt to fix if mismatch (e.g. tokenizer merge)
                     # Or just warn
                     logger.warning(f"Control length mismatch after extension. Expected {length_iter}, got {len(check_ids)}.")
                     
                 logger.info(f"Increasing control length to {length_iter}")
            
             # Re-initialize Prompters to ensure correct slicing with new length
             prompters = [
                SkillPrompter(
                    goal=p[0], 
                    target=p[1], 
                    tokenizer=self.tokenizer, 
                    control_init=current_control_text,
                    template_name=config.template_name,
                    final_target=p[2], # Uses extracted final target
                    device=self.device,
                    # reasoning=None # Initial no reasoning? Or generate before loop?
                    # extract_prompt=getattr(config, 'extract_prompt', None) # Safely get if config updated
                ) for p in train_pairs
             ]
             
             # Optimization Loop for this length
             # Reset patience for new length? Reference says yes.
             patience_counter = 0
             current_best_loss_for_len = float('inf')
             
             for epoch in range(config.iterations):
                if patience_counter >= config.patience:
                    logger.info("Patience reached for this length.")
                    break
                    
                logger.info(f"Len {length_iter} | Epoch {epoch+1}/{config.iterations}")

                # 0. Generate Reasoning (Once per Epoch)
                # We need batch inputs of [User+Goal+Control].
                # SkillPrompter.input_ids includes Target if not sliced out.
                # But during generation we want context up to Assistant Header.
                # SkillPrompter._assistant_role_slice gives end of Assistant Header (start of response).
                
                # Check 1st prompter to see slice
                ref_p = prompters[0]
                gen_input_list = [p.input_ids[:p._assistant_role_slice.stop] for p in prompters]
                gen_batch = pad_sequence(gen_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
                
                # Generate
                reasoning_outputs = greater_core.generate_reasoning(
                    self.model,
                    self.tokenizer,
                    input_ids=gen_batch,
                    temperature=config.temperature,
                    top_k=config.top_k, 
                    max_new_tokens=100,
                    decode=True,
                    return_only_new=True
                )
                
                # Update Prompters with Reasoning & Extract Prompt
                for idx, p in enumerate(prompters):
                    # reasoning_outputs is list of strings? 
                    # greater_core.generate_reasoning returns list of strings "reasoning_text"
                    p.update_reasoning(reasoning_outputs[idx])
                    # Also set extract prompt if configured
                    if config.extract_prompt:
                        p.extract_prompt = config.extract_prompt
                        p._update_ids() # Trigger reconstruct
                
                # Iterate over positions
                control_slice = prompters[0]._control_slice
                control_len_tokens = control_slice.stop - control_slice.start
                
                # Check consistency
                # assert control_len_tokens == length_iter # Might differ due to tokenization
                
                for i in range(control_len_tokens):
                    # 1. Prepare Batch inputs (Reasoning included now)
                    input_ids_list = [p.input_ids for p in prompters]
                    inputs_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
                    
                    batch_pos_indices = [(p._control_slice.start + i) for p in prompters]
                    
                    # Stage 1: Candidate Proposal
                    candidates = greater_core.propose_candidates(
                        self.model, 
                        self.tokenizer, 
                        prompt_ids=None,
                        input_ids_batch=inputs_batch, 
                        position_idx=batch_pos_indices,
                        top_k=config.top_k, 
                        allow_non_ascii=False,
                        not_allowed_tokens=not_allowed_tokens,
                        device=self.device
                    )
                    
                    if len(candidates) == 0:
                        continue
                        
                    # Stage 2 & 3: Gradient Calculation
                    accumulated_grads = torch.zeros(1, len(candidates), device=self.device)
                    grad_samples = min(len(prompters), config.batch_size)
                    
                    for b in range(grad_samples):
                        p = prompters[b]
                        
                        # Focused Target for Gradient
                        # We use the 'target' as focused target if available
                        focused_t = self.tokenizer(p.final_target, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.device) if p.final_target else None

                        grad = greater_core.compute_gradient(
                            self.model,
                            context_ids_with_gt=p.input_ids,
                            prompt_pos_idx=batch_pos_indices[b],
                            loss_slice=p._loss_slice,
                            candidates=candidates,
                            control_slice=p._control_slice,
                            control_weight=config.control_weight,
                            focused_target=focused_t,
                            device=self.device
                        )
                        
                        # Normalize Gradient (Fix 1: L2 Norm)
                        # Avoid div by zero
                        grad_norm = grad.norm(dim=-1, keepdim=True)
                        if grad_norm.item() > 1e-8:
                            grad = grad / grad_norm
                            
                        accumulated_grads += grad
                    
                    # Stage 4: Selection & Update with Cache
                    vals, sorted_indices = torch.topk(-accumulated_grads.squeeze(), config.top_mu)
                    best_candidates = candidates[sorted_indices]
                    
                    step_best_loss = float('inf')
                    best_cand_token = None
                    
                    # Validate on first sample (or batch)
                    val_prompter = prompters[0]
                    val_pos_idx = batch_pos_indices[0]
                    current_token = val_prompter.input_ids[val_pos_idx].item()
                    
                    eval_tokens = best_candidates.tolist()
                    if current_token not in eval_tokens:
                        eval_tokens.append(current_token)
                    
                    orig_token = val_prompter.input_ids[val_pos_idx].item()
                    
                    for token_id in eval_tokens:
                        # Apply change temporarily
                        val_prompter.input_ids[val_pos_idx] = token_id
                        
                        # Check Cache
                        # Key: tuple of full input ids? Or just control?
                        # Since context is deterministic, input_ids is safe key.
                        cache_key = tuple(val_prompter.input_ids.tolist())
                        
                        if cache_key in loss_cache:
                            loss_val = loss_cache[cache_key]
                        else:
                            # Forward Pass
                            with torch.no_grad():
                                loss_slice = val_prompter._loss_slice
                                outputs = self.model(val_prompter.input_ids.unsqueeze(0))
                                logits = outputs.logits
                                shift_logits = logits[0, loss_slice.start-1 : loss_slice.stop-1, :]
                                shift_labels = val_prompter.input_ids[loss_slice]
                                loss_t = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
                                loss_val = loss_t.item()
                                # Add to cache
                                loss_cache[cache_key] = loss_val
                        
                        if loss_val < step_best_loss:
                            step_best_loss = loss_val
                            best_cand_token = token_id
                            
                    # Restore for next candidate check
                    val_prompter.input_ids[val_pos_idx] = orig_token

                    # Update if improved
                    # We check against current global best or local best?
                    # Usually we take the best of candidates and UPDATE the prompt regardless,
                    # to escape local minima (Simulated Annealing or just Greedy).
                    # GreaTer is Greedy Coordinate Gradient. It updates to the best found neighbor.
                    
                    if best_cand_token != orig_token:
                        logger.info(f"Position {i}: Updated token {self.tokenizer.decode([orig_token])} -> {self.tokenizer.decode([best_cand_token])} | Loss: {step_best_loss:.4f} (was {current_best_loss_for_len:.4f})")
                    else:
                        logger.debug(f"Position {i}: No improvement (Loss: {step_best_loss:.4f})")
                        
                    # Update all prompters
                    current_ids = prompters[0].input_ids[prompters[0]._control_slice].clone()
                    if i < len(current_ids):
                         current_ids[i] = best_cand_token
                         
                    # Update prompters
                    for p in prompters:
                         p.update_control(current_ids)
                         
                    # Track Global Best
                    if step_best_loss < best_loss:
                        best_loss = step_best_loss
                        current_control_text = prompters[0].control
                        patience_counter = 0 # Reset patience on improvement
                    else:
                        pass
                        
                    if step_best_loss < current_best_loss_for_len:
                        logger.info(f"New best loss for len {length_iter} at epoch {epoch+1}: {step_best_loss:.4f}")
                        current_best_loss_for_len = step_best_loss
                    
                # End of Epoch check
                patience_counter += 1
                if current_best_loss_for_len < config.early_stop_threshold:
                     logger.info("Early stopping threshold reached.")
                     return OptimizedSkill(
                        name=skill.name,
                        description=skill.description,
                        content=current_control_text,
                        original_skill_name=skill.name,
                        optimization_metrics={"final_loss": best_loss}
                    )
                     
        return OptimizedSkill(
            name=skill.name,
            description=skill.description,
            content=current_control_text,
            original_skill_name=skill.name,
            optimization_metrics={"final_loss": best_loss}
        )

    def _load_training_data_raw(self, config):
        if config.dataset_path:
            import json
            import csv
            data = []
            try:
                # Check file format by reading the first line
                with open(config.dataset_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    f.seek(0)
                    
                    if first_line.startswith("goal,target,final_target"):
                        # BBH CSV Format
                        reader = csv.DictReader(f)
                        for i, row in enumerate(reader):
                            if i >= config.num_examples:
                                break
                            goal = row.get('goal', '').strip()
                            target = row.get('target', '').strip()
                            final_target = row.get('final_target', '').strip()
                            if goal and target:
                                data.append((goal, target, final_target))
                    else:
                        # Fallback to JSONL logic
                        for i, line in enumerate(f):
                            if i >= config.num_examples:
                                break
                            if not line.strip():
                                continue
                            item = json.loads(line)
                            # GSM8K Format usually: 'question', 'answer'
                            # Fallback to other common keys just in case
                            goal = item.get('question', item.get('input', ''))
                            target = item.get('answer', item.get('output', ''))
                            if goal and target:
                                 # Extract final target if possible (for GSM8K)
                                 final_target = None
                                 if "####" in target:
                                     final_target = target.split("####")[-1].strip()
                                 data.append((goal, target, final_target))
                             
                logger.info(f"Loaded {len(data)} examples from {config.dataset_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                
        # MVP Placeholder: List of (Goal, Target, FinalTarget)
        return [("Question 1", "Answer 1", None), ("Question 2", "Answer 2", None)]

    def evaluate_skill(self, skill: Skill, dataset_path: str, num_examples: int = 10, template_name: str = 'llama-2') -> dict:
        """
        Evaluates the skill on a test dataset.
        """
        import json
        logger.info(f"Evaluating skill '{skill.name}' on {dataset_path} ({num_examples} examples)")
        
        correct = 0
        total = 0
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
        # Limit examples
        lines = lines[:num_examples]
        
        for line in lines:
            try:
                item = json.loads(line)
                goal = item.get('question', item.get('input', ''))
                target = item.get('answer', item.get('output', ''))
                
                # Construct Prompt
                # We use SkillPrompter just to get the input_ids for the prompt part
                # Target is empty for generation context
                prompter = SkillPrompter(
                    goal=goal,
                    target="", 
                    tokenizer=self.tokenizer,
                    control_init=skill.content,
                    template_name=template_name,
                    device=self.device
                )
                
                # Input for generation: User + Goal + Control + Assistant Header
                # Slicing logic depends on template. 
                # Prompter._assistant_role_slice gives us where assistant starts.
                # input_ids should end right after assistant header.
                
                # Llama-3 in Prompter ends with "<|start_header_id|>assistant<|end_header_id|>\n\nTarget"
                # We want everything UP TO Target.
                # Valid input is prompter.input_ids[:prompter._target_slice.start]
                
                input_ids = prompter.input_ids[:prompter._target_slice.start]
                
                # Generate
                with torch.no_grad():
                    # Generate options
                    outputs = self.model.generate(
                        input_ids.unsqueeze(0),
                        max_new_tokens=100, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode generated part only
                generated_ids = outputs[0][len(input_ids):]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Comparison (Simple containment for GSM8K "#### Number")
                # GSM8K answer usually ends with "#### <Answer>"
                if "####" in target:
                    ground_truth = target.split("####")[-1].strip()
                else:
                    ground_truth = target.strip()
                    
                # Check if ground_truth is in generated text
                # Logic: Is the answer number in the output?
                # This is a bit loose but works for validation check.
                if ground_truth in generated_text:
                    correct += 1
                
                total += 1
                
                if total % 5 == 0:
                    logger.info(f"Evaluated {total}/{len(lines)}...")

            except Exception as e:
                logger.warning(f"Error evaluating example: {e}")
                continue
                
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Evaluation Complete. Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return {"accuracy": accuracy, "correct": correct, "total": total}
