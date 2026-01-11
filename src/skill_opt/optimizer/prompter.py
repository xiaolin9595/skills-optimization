import torch
from typing import List, Optional, Tuple, Any

class ConversationTemplate:
    """Minimal Conversation Template holder."""
    def __init__(self, name, roles, sep, sep2=None):
        self.name = name
        self.roles = roles
        self.sep = sep
        self.sep2 = sep2 or sep
        self.messages = []

    def append_message(self, role, message):
        self.messages.append([role, message])
        
    def get_prompt(self):
        # Very Basic implementation for Llama-2
        if self.name == 'llama-2':
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += f"[INST] {message} [/INST]"
                    else:
                        ret += f" {message}"
                else:
                    # None message usually means start of generation?
                    pass 
            return ret
        return ""
        
    def update_last_message(self, message):
        self.messages[-1][1] = message

class SkillPrompter:
    """
    Manages the prompt construction and slicing for GreaTer.
    Adapted from referenceSolution/GreaTer/llm_opt/base/attack_manager.py
    """
    def __init__(self,
                 goal: str,
                 target: str,
                 tokenizer: Any,
                 template_name: str = 'llama-2',
                 control_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 final_target: Optional[str] = None,
                 device='cpu',
                 reasoning: Optional[str] = None,
                 extract_prompt: Optional[str] = None):
        self.goal = goal
        self.target = target
        self.control = control_init
        self.final_target = final_target
        self.reasoning = reasoning
        self.extract_prompt = extract_prompt
        self.tokenizer = tokenizer
        self.device = device
        self.template_name = template_name
        self.control_pos = "post" # default
        
        # Slices
        self._user_role_slice = slice(0,0)
        self._goal_slice = slice(0,0)
        self._control_slice = slice(0,0)
        self._assistant_role_slice = slice(0,0)
        self._target_slice = slice(0,0)
        self._loss_slice = slice(0,0)
        self._focused_target_slice = None
        
        self.input_ids = torch.empty(0)
        self._update_ids()

    def _find_subarray_indices(self, full_ids: List[int], sub_str: str):
        # Helper to find sub_str tokens in full_ids
        # Tokenizer dependent logic akin to reference
        sub_ids = self.tokenizer(sub_str, add_special_tokens=False).input_ids
        if not sub_ids:
             return -1, -1
             
        # Simple backward search
        n = len(full_ids)
        m = len(sub_ids)
        for i in range(n - m, -1, -1):
            if full_ids[i:i+m] == sub_ids:
                return i, i+m
        
        # Try with leading space often tokenizer issue
        sub_ids_sp = self.tokenizer(" " + sub_str, add_special_tokens=False).input_ids
        m = len(sub_ids_sp)
        for i in range(n - m, -1, -1):
            if full_ids[i:i+m] == sub_ids_sp:
                return i, i+m
                
        return -1, -1

    def _update_ids(self):
        # Implementation for Llama-2 style slicing
        # This is a simplified version of the reference logic
        
        if self.template_name == 'llama-2':
            # Llama-2 Format: <s>[INST] {System} {User} [/INST] {Asst} </s>
            
            # Step 1: User Start
            full_input = ""
            # Base
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._user_role_slice = slice(None, len(toks))
            
            # Goal
            full_input += f"[INST] {self.goal}"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks))
            
            # Control (Post Goal)
            sep = " "
            full_input += f"{sep}{self.control}"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))
            
            # End User / Start Assistant
            full_input += " [/INST]"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            
            # Target
            full_input += f" {self.target}"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks))
            
            # Loss Slice logic from GreaTer Reference (Llama-2)
            end_offset = 3 if len(toks) > 3 else 1
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - end_offset)
            
            # Focused Target Slice
            if self.final_target:
                idx1, idx2 = self._find_subarray_indices(toks, self.final_target)
                if idx1 != -1:
                    self._focused_target_slice = slice(idx1, idx2)
            
            self.input_ids = torch.tensor(toks, device=self.device)
            
        elif self.template_name == 'llama-3':
            # Llama-3 Format:
            # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{Goal} {Control}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{Target}
            
            # NOTE: We assume tokenizer adds <|begin_of_text|> automatically on encode if configured, 
            # or we might need to add it. For safety, we use add_special_tokens=True for first call.
            
            # 1. User Header
            full_input = "<|start_header_id|>user<|end_header_id|>\n\n"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._user_role_slice = slice(0, len(toks))
            
            # 2. Goal
            full_input += f"{self.goal}"
            # Llama-3 tokenizer might merge tokens oddly if we just concat.
            # But re-tokenizing full string is safer.
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks))
            
            # 3. Control
            sep = " "
            full_input += f"{sep}{self.control}"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))
            
            # 4. User End / Assistant Header
            full_input += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            
            # 5. Reasoning & Extract Prompt (Dynamic)
            if self.reasoning:
                full_input += f"{self.reasoning}"
                if self.extract_prompt:
                    full_input += f" {self.extract_prompt}"
                # Re-tokenize
                toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
                # Assistant Role Slice covers up to end of reasoning/extract prompt?
                # GreaTer logic usually treats reasoning as part of context.
                # _assistant_role_slice usually marks where 'user input end'.
                # But for Loss calculation we want to predict Target.
                # So inputs up to Target are Context.
                # We can keep _assistant_role_slice as header, and everything else is just 'prefix'.
                # But let's verify if _assistant_role_slice is used elsewhere.
                # It is used for `_loss_slice` calculation relative to it?
                # No, loss slice depends on target position.
                pass
            
            # 6. Target
            full_input += f"{self.target}"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            
            # Target slice starts after whatever came before (Reasoning or Header)
            # Find start index
            # If we had reasoning, target starts after reasoning tokens.
            # If not, checks previous length.
            
            # Since we re-tokenized full_input at step 5, len(toks) then was 'start'.
            # But wait, python variable `toks` is overwritten.
            # Let's save length before adding target.
            
            # Clean Implementation:
            input_so_far = full_input[:-(len(self.target))] # Strip target back
            toks_prefix = self.tokenizer(input_so_far, add_special_tokens=True).input_ids
            start_idx = len(toks_prefix)
            
            self._target_slice = slice(start_idx, len(toks))
            
            # Loss Slice
            # Predict Target.
            # Target starts at `_assistant_role_slice.stop`.
            # Logits index needed: `_assistant_role_slice.stop - 1`.
            # Llama-3 EOS behavior? Usually <|eot_id|> or similar.
            # If target string doesn't include EOS, we predict up to last token of target.
            
            start = self._target_slice.start
            end = self._target_slice.stop
            # We want to predict from Target[0] to Target[N].
            # Input[start-1] predicts Input[start].
            # Input[end-1] predicts Input[end] (if end < len).
            
            self._loss_slice = slice(start - 1, end - 1)
            
            # Focused Target
            if self.final_target:
                idx1, idx2 = self._find_subarray_indices(toks, self.final_target)
                if idx1 != -1:
                    self._focused_target_slice = slice(idx1, idx2)
            
            self.input_ids = torch.tensor(toks, device=self.device)
            
        else:
            # Fallback to simple concatenation
            full_text = f"{self.goal} {self.control} {self.target}"
            # Ensure 1D tensor
            encoded = self.tokenizer(full_text, return_tensors='pt').input_ids
            if encoded.dim() > 1:
                encoded = encoded[0]
            self.input_ids = encoded.to(self.device)
            
            # Dummy slices
            # Approximate just to be safe if used
            self._control_slice = slice(len(self.goal)//4, (len(self.goal)+len(self.control))//4) # Very rough approximation

    def update_control(self, new_control_ids: torch.Tensor):
        self.control = self.tokenizer.decode(new_control_ids, skip_special_tokens=True)
        self._update_ids()

    def update_reasoning(self, reasoning: str):
        self.reasoning = reasoning
        self._update_ids()
