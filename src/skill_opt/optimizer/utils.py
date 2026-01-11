from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from skill_opt.core.config import AppConfig

def get_nonascii_toks(tokenizer, device='cpu'):
    """
    Returns a tensor of token IDs that correspond to Non-ASCII characters.
    Used for filtering candidates.
    """
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    vocab_size = tokenizer.vocab_size
    
    # Batch decode to be faster
    batch_size = 1000
    for i in range(0, vocab_size, batch_size):
        ids = list(range(i, min(i + batch_size, vocab_size)))
        texts = tokenizer.batch_decode([[tid] for tid in ids])
        for tid, text in zip(ids, texts):
            if not is_ascii(text):
                non_ascii_toks.append(tid)
        
    # Also exclude special tokens explicitly
    for special_id in tokenizer.all_special_ids:
        non_ascii_toks.append(special_id)
        
    return torch.tensor(list(set(non_ascii_toks)), device=device, dtype=torch.long)

def load_model_and_tokenizer(
    config: AppConfig,
    device_map: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the model and tokenizer based on configuration.
    
    Args:
        config: Application configuration containing model path/name.
        device_map: Optional Override for device map (e.g., 'auto', 'cpu', 'cuda').
        
    Returns:
        (model, tokenizer)
    """
    # Determine Model Name/Path
    model_name = "gpt2" # Default fallback
    
    # Check if we should use config attributes
    if hasattr(config, "model_name"):
        model_name = config.model_name
    
    # Determine Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    # Mac MPS support
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left' # Important for batch generation
    )
    
    # Model Specific Tokenizer Fixes (GreaTer Requirement)
    if 'llama-2' in model_name.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        
    if 'llama-3' in model_name.lower():
        # Llama-3 often uses <|eot_id|> or <|end_of_text|> as EOS.
        # It has a distinct PAD token usually, but if not set:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    # Load Model
    # Note: For GreaTer, we need gradient access to embeddings.
    dtype = torch.float16 if device != "cpu" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map # Let transformers handle map if provided
    )
    
    if device_map is None:
        model.to(device)
        
    model.eval() # Set to eval mode by default
    
    return model, tokenizer
