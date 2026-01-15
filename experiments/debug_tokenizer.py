"""
Debug tokenizer behavior for target slice
"""

import logging
import torch
from skill_opt.core.config import AppConfig
from skill_opt.optimizer.utils import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    app_config = AppConfig(model_name="/workspace/llama3/Llama-3.2-1B-Instruct")
    model, tokenizer = load_model_and_tokenizer(app_config)

    # Simulate the prompt construction
    reasoning = "Let's think step by step. The answer is obviously"
    extract_prompt = "Therefore, the final answer is $ "
    target = "False"

    # Build prompt incrementally
    base = "<|start_header_id|>user<|end_header_id|>\n\nQuestion<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Step 1: Base
    toks_base = tokenizer(base, add_special_tokens=True).input_ids
    logger.info(f"Base tokens: {len(toks_base)}")

    # Step 2: Add reasoning
    with_reasoning = base + reasoning
    toks_reasoning = tokenizer(with_reasoning, add_special_tokens=True).input_ids
    logger.info(f"With reasoning: {len(toks_reasoning)}")

    # Step 3: Add extract prompt
    with_extract = with_reasoning + " " + extract_prompt
    toks_extract = tokenizer(with_extract, add_special_tokens=True).input_ids
    logger.info(f"With extract: {len(toks_extract)}")

    # Step 4: Add target without space
    with_target_nospace = with_extract + target
    toks_target_nospace = tokenizer(
        with_target_nospace, add_special_tokens=True
    ).input_ids
    logger.info(f"With target (no space): {len(toks_target_nospace)}")
    logger.info(f"Target tokens added: {len(toks_target_nospace) - len(toks_extract)}")

    # Step 5: Add target with space
    with_target_space = with_extract + " " + target
    toks_target_space = tokenizer(with_target_space, add_special_tokens=True).input_ids
    logger.info(f"With target (with space): {len(toks_target_space)}")
    logger.info(f"Target tokens added: {len(toks_target_space) - len(toks_extract)}")

    # Decode to see what's happening
    logger.info(f"\n--- Decoded extract prompt ending ---")
    logger.info(f"Last 10 chars of with_extract: '{with_extract[-30:]}'")
    logger.info(f"Last 3 tokens: {tokenizer.decode(toks_extract[-3:])}")

    logger.info(f"\n--- Target tokenization ---")
    target_only = tokenizer(target, add_special_tokens=False).input_ids
    logger.info(f"Target alone: {target_only} -> '{tokenizer.decode(target_only)}'")

    target_with_space = tokenizer(" " + target, add_special_tokens=False).input_ids
    logger.info(
        f"Target with leading space: {target_with_space} -> '{tokenizer.decode(target_with_space)}'"
    )

    # Check if extract_prompt ends with space
    logger.info(f"\n--- Extract prompt analysis ---")
    logger.info(f"Extract prompt: '{extract_prompt}'")
    logger.info(f"Ends with space: {extract_prompt.endswith(' ')}")

    # The issue: extract_prompt ends with "$ " so adding target creates "$ False"
    # The tokenizer might merge "$ " + "False" differently
    combined = extract_prompt + target
    combined_toks = tokenizer(combined, add_special_tokens=False).input_ids
    logger.info(f"Combined '$ False': {combined_toks}")

    separate_extract = tokenizer(extract_prompt, add_special_tokens=False).input_ids
    logger.info(f"Separate extract: {separate_extract}")
    logger.info(
        f"Difference: {len(combined_toks) - len(separate_extract)} tokens for target"
    )


if __name__ == "__main__":
    main()
