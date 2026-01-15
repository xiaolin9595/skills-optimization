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
        if self.name == "llama-2":
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

    def __init__(
        self,
        goal: str,
        target: str,
        tokenizer: Any,
        template_name: str = "llama-2",
        control_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        final_target: Optional[str] = None,
        device="cpu",
        reasoning: Optional[str] = None,
        extract_prompt: Optional[str] = None,
    ):
        self.goal = goal
        self.target = target
        self.control = control_init
        self.final_target = final_target
        self.reasoning = reasoning
        self.extract_prompt = extract_prompt
        self.tokenizer = tokenizer
        self.device = device
        self.template_name = template_name
        self.control_pos = "post"  # default

        # Slices
        self._user_role_slice = slice(0, 0)
        self._goal_slice = slice(0, 0)
        self._control_slice = slice(0, 0)
        self._assistant_role_slice = slice(0, 0)
        self._target_slice = slice(0, 0)
        self._loss_slice = slice(0, 0)
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
            if full_ids[i : i + m] == sub_ids:
                return i, i + m

        # Try with leading space often tokenizer issue
        sub_ids_sp = self.tokenizer(" " + sub_str, add_special_tokens=False).input_ids
        m = len(sub_ids_sp)
        for i in range(n - m, -1, -1):
            if full_ids[i : i + m] == sub_ids_sp:
                return i, i + m

        return -1, -1

    def _update_ids(self):
        # Implementation for Llama-2 style slicing
        # This is a simplified version of the reference logic

        if self.template_name == "llama-2":
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
            self._loss_slice = slice(
                self._assistant_role_slice.stop - 1, len(toks) - end_offset
            )

            # Focused Target Slice
            if self.final_target:
                idx1, idx2 = self._find_subarray_indices(toks, self.final_target)
                if idx1 != -1:
                    self._focused_target_slice = slice(idx1, idx2)

            self.input_ids = torch.tensor(toks, device=self.device)

        elif self.template_name == "llama-3":
            # Llama-3 Format:
            # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{Goal} {Control}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{Reasoning}{Extract}{Target}

            # NOTE: We assume tokenizer adds <|begin_of_text|> automatically on encode if configured,
            # or we might need to add it. For safety, we use add_special_tokens=True for first call.

            # 1. User Header
            full_input = "<|start_header_id|>user<|end_header_id|>\n\n"
            toks = self.tokenizer(full_input, add_special_tokens=True).input_ids
            self._user_role_slice = slice(0, len(toks))

            # 2. Goal
            full_input += f"{self.goal}"
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
            # FIX: Save token length BEFORE adding reasoning for proper slice calculation
            pre_reasoning_len = len(toks)

            if self.reasoning:
                full_input += f"{self.reasoning}"
                if self.extract_prompt:
                    full_input += f" {self.extract_prompt}"
                # Re-tokenize to get updated token count
                toks = self.tokenizer(full_input, add_special_tokens=True).input_ids

            # 6. Target - FIX: Save token length BEFORE adding target
            pre_target_len = len(toks)

            if self.target:
                # IMPORTANT: Tokenizer behavior issue
                # When extract_prompt ends with space and target starts without space,
                # tokenizer may merge them (e.g., "$ " + "False" -> " False" token replaces " " token)
                #
                # Solution: Tokenize target separately and compare lengths
                # This ensures we correctly identify the target portion

                # Try adding target and see how many new tokens we get
                full_input_with_target = full_input + self.target
                toks_with_target = self.tokenizer(
                    full_input_with_target, add_special_tokens=True
                ).input_ids

                target_token_count = len(toks_with_target) - len(toks)

                if target_token_count <= 0:
                    # Token merging occurred - target was absorbed into previous tokens
                    # Tokenize target separately to find its expected token representation
                    target_toks = self.tokenizer(
                        self.target, add_special_tokens=False
                    ).input_ids

                    # Search for these tokens in the final sequence (from the end)
                    # They should be near the end of the sequence
                    found_start = -1
                    for i in range(len(toks_with_target) - len(target_toks), -1, -1):
                        if toks_with_target[i : i + len(target_toks)] == target_toks:
                            found_start = i
                            break

                    if found_start == -1:
                        # Try with space prefix
                        target_toks_sp = self.tokenizer(
                            " " + self.target, add_special_tokens=False
                        ).input_ids
                        for i in range(
                            len(toks_with_target) - len(target_toks_sp), -1, -1
                        ):
                            if (
                                toks_with_target[i : i + len(target_toks_sp)]
                                == target_toks_sp
                            ):
                                found_start = i
                                break
                        if found_start != -1:
                            target_token_count = len(target_toks_sp)
                    else:
                        target_token_count = len(target_toks)

                    if found_start != -1:
                        pre_target_len = found_start

                full_input = full_input_with_target
                toks = toks_with_target

            # FIX: Use saved token lengths for accurate slice calculation
            self._target_slice = slice(pre_target_len, len(toks))

            # Loss Slice - aligned with reference implementation
            # Predict Target tokens: logits[i] predicts token[i+1]
            # So to predict target[start:stop], we need logits[start-1:stop-1]
            start = self._target_slice.start
            end = self._target_slice.stop

            # Validate target slice
            if start >= end:
                # Target is empty - this is a problem, log warning
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Empty target slice detected: start={start}, end={end}, target='{self.target}'"
                )
                # Fallback: use loss on assistant response (after header)
                # Aligned with reference: loss_slice = slice(assistant_role_slice.stop-1, len(toks)-1)
                self._loss_slice = slice(
                    self._assistant_role_slice.stop - 1, len(toks) - 1
                )
            else:
                # Standard loss slice calculation
                self._loss_slice = slice(start - 1, end - 1)

            # Validate loss slice bounds
            if self._loss_slice.start < 0:
                self._loss_slice = slice(0, self._loss_slice.stop)
            if self._loss_slice.stop > len(toks):
                self._loss_slice = slice(self._loss_slice.start, len(toks))

            # Focused Target - search in full token sequence
            if self.final_target:
                idx1, idx2 = self._find_subarray_indices(toks, self.final_target)
                if idx1 != -1:
                    self._focused_target_slice = slice(idx1, idx2)
                else:
                    self._focused_target_slice = None

            self.input_ids = torch.tensor(toks, device=self.device)

        else:
            # Fallback to simple concatenation
            full_text = f"{self.goal} {self.control} {self.target}"
            # Ensure 1D tensor
            encoded = self.tokenizer(full_text, return_tensors="pt").input_ids
            if encoded.dim() > 1:
                encoded = encoded[0]
            self.input_ids = encoded.to(self.device)

            # Dummy slices
            # Approximate just to be safe if used
            self._control_slice = slice(
                len(self.goal) // 4, (len(self.goal) + len(self.control)) // 4
            )  # Very rough approximation

    def update_control(self, new_control_ids: torch.Tensor):
        self.control = self.tokenizer.decode(new_control_ids, skip_special_tokens=True)
        self._update_ids()

    def update_reasoning(self, reasoning: str):
        self.reasoning = reasoning
        self._update_ids()
