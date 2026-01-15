"""
Debug script to investigate the NaN issue after reasoning generation
"""

import logging
import torch
import csv
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig
from skill_opt.optimizer.utils import load_model_and_tokenizer
from skill_opt.optimizer.prompter import SkillPrompter
from skill_opt.optimizer import greater_core
from torch.nn.utils.rnn import pad_sequence

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_bbh_samples(path, n=4):
    """Load n samples from BBH"""
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

    # Load samples
    data = load_bbh_samples(
        "referenceSolution/GreaTer/data/BBH/boolean_expressions.json", n=4
    )

    logger.info("=" * 80)
    logger.info("DEBUG: Investigating NaN after Reasoning Generation")
    logger.info("=" * 80)

    control_init = " proper logical reasoning and think step by step. Finally give the actual correct answer."

    # Create prompters
    prompters = [
        SkillPrompter(
            goal=sample["goal"],
            target=sample["target"],
            tokenizer=tokenizer,
            control_init=control_init,
            template_name="llama-3",
            final_target=sample["final_target"],
            device=device,
        )
        for sample in data
    ]

    # Step 1: Check initial prompters
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Initial Prompter States")
    logger.info("=" * 80)
    for i, p in enumerate(prompters):
        logger.info(f"Prompter {i}:")
        logger.info(f"  Input IDs length: {len(p.input_ids)}")
        logger.info(f"  Loss Slice: {p._loss_slice}")
        logger.info(f"  Target Slice: {p._target_slice}")
        logger.info(f"  Control Slice: {p._control_slice}")

        # Check if loss slice is valid
        if p._loss_slice.start >= p._loss_slice.stop:
            logger.error(f"  PROBLEM: Loss slice is empty!")
        if p._loss_slice.stop > len(p.input_ids):
            logger.error(
                f"  PROBLEM: Loss slice stop ({p._loss_slice.stop}) exceeds input length ({len(p.input_ids)})!"
            )

    # Step 2: Generate reasoning (like in the optimization loop)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Generate Reasoning")
    logger.info("=" * 80)

    ref_p = prompters[0]
    gen_input_list = [p.input_ids[: p._assistant_role_slice.stop] for p in prompters]

    logger.info(f"Generation input lengths: {[len(x) for x in gen_input_list]}")

    gen_batch = pad_sequence(
        gen_input_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)
    logger.info(f"Padded batch shape: {gen_batch.shape}")

    # Generate
    reasoning_outputs = greater_core.generate_reasoning(
        model,
        tokenizer,
        input_ids=gen_batch,
        temperature=0.7,
        top_k=50,
        max_new_tokens=100,
        decode=True,
        return_only_new=True,
    )

    logger.info(f"Generated {len(reasoning_outputs)} reasoning outputs")
    for i, r in enumerate(reasoning_outputs):
        logger.info(f"Reasoning {i}: {r[:100]}...")

    # Step 3: Update prompters with reasoning
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Update Prompters with Reasoning")
    logger.info("=" * 80)

    extract_prompt = (
        "Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
    )

    for idx, p in enumerate(prompters):
        logger.info(f"\nPrompter {idx} BEFORE update:")
        logger.info(f"  Input IDs length: {len(p.input_ids)}")
        logger.info(f"  Loss Slice: {p._loss_slice}")
        logger.info(f"  Target Slice: {p._target_slice}")

        # Update with reasoning
        p.update_reasoning(reasoning_outputs[idx])
        p.extract_prompt = extract_prompt
        p._update_ids()

        logger.info(f"Prompter {idx} AFTER update:")
        logger.info(f"  Input IDs length: {len(p.input_ids)}")
        logger.info(f"  Loss Slice: {p._loss_slice}")
        logger.info(f"  Target Slice: {p._target_slice}")

        # Check if loss slice is valid after update
        if p._loss_slice.start >= p._loss_slice.stop:
            logger.error(f"  PROBLEM: Loss slice is empty after reasoning update!")
        if p._loss_slice.stop > len(p.input_ids):
            logger.error(
                f"  PROBLEM: Loss slice stop ({p._loss_slice.stop}) exceeds input length ({len(p.input_ids)})!"
            )
        if p._loss_slice.start < 0:
            logger.error(
                f"  PROBLEM: Loss slice start ({p._loss_slice.start}) is negative!"
            )

    # Step 4: Try computing gradient after reasoning update
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Gradient Computation After Reasoning Update")
    logger.info("=" * 80)

    from skill_opt.optimizer.utils import get_nonascii_toks

    not_allowed_tokens = get_nonascii_toks(tokenizer, device=device)

    for idx, p in enumerate(prompters):
        logger.info(f"\nTesting gradient computation for Prompter {idx}:")

        input_batch = p.input_ids.unsqueeze(0)
        pos_idx = p._control_slice.start

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

            focused_t = (
                tokenizer(p.final_target, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .to(device)
                if p.final_target
                else None
            )

            grad = greater_core.compute_gradient(
                model,
                context_ids_with_gt=p.input_ids,
                prompt_pos_idx=pos_idx,
                loss_slice=p._loss_slice,
                candidates=candidates,
                control_slice=p._control_slice,
                control_weight=0.1,
                focused_target=focused_t,
                device=device,
                grad_clip=1.0,
                use_amp=False,
            )

            logger.info(f"  Gradient sum: {grad.sum().item():.6f}")
            if grad.sum().item() == 0:
                logger.warning(f"  WARNING: Zero gradient (likely due to NaN loss)")
            else:
                logger.info(f"  SUCCESS: Non-zero gradient computed!")

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
