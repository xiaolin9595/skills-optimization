"""
Deep debug script to identify the root cause of Loss NaN issues in GreaTer
"""

import logging
import torch
import csv
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig
from skill_opt.optimizer.utils import load_model_and_tokenizer
from skill_opt.optimizer.prompter import SkillPrompter
from skill_opt.optimizer import greater_core

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_bbh_sample(path, n=1):
    """Load first n samples from BBH"""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            data.append(
                {
                    "goal": row["goal"],
                    "target": row["target"],
                    "final_target": row.get("final_target", ""),
                }
            )
    return data


def main():
    # Initialize
    app_config = AppConfig(model_name="/workspace/llama3/Llama-3.2-1B-Instruct")
    model, tokenizer = load_model_and_tokenizer(app_config)
    device = model.device

    # Load sample
    data = load_bbh_sample(
        "referenceSolution/GreaTer/data/BBH/boolean_expressions.json", n=1
    )
    sample = data[0]

    logger.info("=" * 80)
    logger.info("DEBUG: Loss NaN Root Cause Analysis")
    logger.info("=" * 80)
    logger.info(f"Goal: {sample['goal']}")
    logger.info(f"Target: {sample['target']}")
    logger.info(f"Final Target: {sample['final_target']}")

    # Create Prompter
    control_init = " proper logical reasoning and think step by step. Finally give the actual correct answer."

    prompter = SkillPrompter(
        goal=sample["goal"],
        target=sample["target"],
        tokenizer=tokenizer,
        control_init=control_init,
        template_name="llama-3",
        final_target=sample["final_target"],
        device=device,
    )

    logger.info("=" * 80)
    logger.info("Prompter Analysis")
    logger.info("=" * 80)
    logger.info(f"Input IDs shape: {prompter.input_ids.shape}")
    logger.info(f"Input IDs length: {len(prompter.input_ids)}")
    logger.info(f"User Role Slice: {prompter._user_role_slice}")
    logger.info(f"Goal Slice: {prompter._goal_slice}")
    logger.info(f"Control Slice: {prompter._control_slice}")
    logger.info(f"Assistant Role Slice: {prompter._assistant_role_slice}")
    logger.info(f"Target Slice: {prompter._target_slice}")
    logger.info(f"Loss Slice: {prompter._loss_slice}")
    logger.info(f"Focused Target Slice: {prompter._focused_target_slice}")

    # Decode and show the full prompt
    full_text = tokenizer.decode(prompter.input_ids, skip_special_tokens=False)
    logger.info(f"\nFull Prompt (decoded):\n{full_text}")

    # Check Loss Slice validity
    logger.info("=" * 80)
    logger.info("Loss Slice Validation")
    logger.info("=" * 80)

    loss_slice = prompter._loss_slice
    seq_len = len(prompter.input_ids)

    logger.info(f"Loss Slice: start={loss_slice.start}, stop={loss_slice.stop}")
    logger.info(f"Sequence Length: {seq_len}")

    # Check for empty slice
    if loss_slice.start >= loss_slice.stop:
        logger.error("PROBLEM: Loss slice is empty (start >= stop)!")
        logger.error(
            "This will cause NaN loss because there are no tokens to compute loss on."
        )

    # Check for out-of-bounds
    if loss_slice.start < 0:
        logger.error(f"PROBLEM: Loss slice start ({loss_slice.start}) is negative!")

    if loss_slice.stop > seq_len:
        logger.error(
            f"PROBLEM: Loss slice stop ({loss_slice.stop}) exceeds sequence length ({seq_len})!"
        )

    # Check Target Slice
    target_slice = prompter._target_slice
    logger.info(f"\nTarget Slice: start={target_slice.start}, stop={target_slice.stop}")

    if target_slice.start >= target_slice.stop:
        logger.error("PROBLEM: Target slice is empty!")
        logger.error(
            "This means the target text is not properly tokenized or positioned."
        )

    # Show what tokens are in the loss slice
    if (
        loss_slice.start < loss_slice.stop
        and loss_slice.start >= 0
        and loss_slice.stop <= seq_len
    ):
        loss_tokens = prompter.input_ids[loss_slice]
        loss_text = tokenizer.decode(loss_tokens, skip_special_tokens=False)
        logger.info(f"\nTokens in Loss Slice: {loss_tokens.tolist()}")
        logger.info(f"Text in Loss Slice: {loss_text}")
    else:
        logger.error("Cannot decode loss slice - indices are invalid!")

    # Attempt direct loss computation
    logger.info("=" * 80)
    logger.info("Direct Loss Computation Test")
    logger.info("=" * 80)

    try:
        with torch.no_grad():
            outputs = model(prompter.input_ids.unsqueeze(0))
            logits = outputs.logits

            logger.info(f"Logits shape: {logits.shape}")
            logger.info(f"Logits max: {logits.max().item():.4f}")
            logger.info(f"Logits min: {logits.min().item():.4f}")
            logger.info(f"Logits mean: {logits.mean().item():.4f}")

            # Check for NaN/Inf in logits
            if torch.isnan(logits).any():
                logger.error("PROBLEM: Logits contain NaN values!")
            if torch.isinf(logits).any():
                logger.error("PROBLEM: Logits contain Inf values!")

            # Try computing loss with the loss_slice
            if loss_slice.start >= 1 and loss_slice.stop <= seq_len:
                # Shift logits
                shift_logits = logits[
                    0, loss_slice.start - 1 : loss_slice.stop - 1, :
                ].contiguous()
                shift_labels = prompter.input_ids[loss_slice].to(device)

                logger.info(f"\nShift Logits shape: {shift_logits.shape}")
                logger.info(f"Shift Labels shape: {shift_labels.shape}")

                if shift_logits.shape[0] == 0:
                    logger.error("PROBLEM: shift_logits is empty after slicing!")
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
                    logger.info(f"Computed Loss: {loss.item():.4f}")

                    if torch.isnan(loss):
                        logger.error("PROBLEM: Computed loss is NaN!")
                    elif torch.isinf(loss):
                        logger.error("PROBLEM: Computed loss is Inf!")
                    else:
                        logger.info("SUCCESS: Loss computation works correctly!")
            else:
                logger.error(
                    f"Cannot compute loss - slice indices invalid for sequence length {seq_len}"
                )

    except Exception as e:
        logger.error(f"Error during loss computation: {e}")

    # Test gradient computation
    logger.info("=" * 80)
    logger.info("Gradient Computation Test")
    logger.info("=" * 80)

    # Get non-ASCII tokens
    from skill_opt.optimizer.utils import get_nonascii_toks

    not_allowed_tokens = get_nonascii_toks(tokenizer, device=device)

    # Test candidate proposal
    input_batch = prompter.input_ids.unsqueeze(0)
    control_slice = prompter._control_slice
    pos_idx = control_slice.start

    try:
        candidates = greater_core.propose_candidates(
            model,
            tokenizer,
            prompt_ids=None,
            input_ids_batch=input_batch,
            position_idx=[pos_idx],
            top_k=40,
            allow_non_ascii=False,
            not_allowed_tokens=not_allowed_tokens,
            device=device,
        )
        logger.info(f"Proposed {len(candidates)} candidates")
        logger.info(f"Sample candidates: {tokenizer.decode(candidates[:5])}")

        # Test gradient computation
        focused_t = (
            tokenizer(
                sample["final_target"], return_tensors="pt", add_special_tokens=False
            )
            .input_ids[0]
            .to(device)
            if sample["final_target"]
            else None
        )

        logger.info(
            f"\nFocused Target IDs: {focused_t.tolist() if focused_t is not None else None}"
        )

        grad = greater_core.compute_gradient(
            model,
            context_ids_with_gt=prompter.input_ids,
            prompt_pos_idx=pos_idx,
            loss_slice=loss_slice,
            candidates=candidates,
            control_slice=control_slice,
            control_weight=0.1,
            focused_target=focused_t,
            device=device,
            grad_clip=1.0,
            use_amp=False,
        )

        logger.info(f"\nGradient shape: {grad.shape}")
        logger.info(f"Gradient sum: {grad.sum().item():.6f}")
        logger.info(f"Gradient max: {grad.max().item():.6f}")
        logger.info(f"Gradient min: {grad.min().item():.6f}")

        if grad.sum().item() == 0:
            logger.warning(
                "WARNING: All gradients are zero - likely due to NaN loss being skipped"
            )
        else:
            logger.info("SUCCESS: Gradient computation produced non-zero gradients!")

    except Exception as e:
        logger.error(f"Error during gradient computation: {e}")
        import traceback

        traceback.print_exc()

    logger.info("=" * 80)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
