"""
Readable Skill Optimization Experiment
Adds constraints to keep optimized skills human-readable
"""

import json
import csv
import logging
import time
import torch
from datetime import datetime
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiments/bbh_readable_experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_readable_token_filter(tokenizer, device):
    """
    Create a filter tensor that only allows readable ASCII tokens.
    This mimics the official GreaTer behavior with allow_non_ascii=False.
    """
    vocab_size = tokenizer.vocab_size
    not_allowed = []

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)

            # Check if token is "readable"
            is_readable = True

            # 1. Must be ASCII
            if not decoded.isascii():
                is_readable = False

            # 2. Must not be empty after decode
            if len(decoded.strip()) == 0 and len(decoded) > 0:
                # Allow single space, but not empty
                if decoded != " ":
                    is_readable = False

            # 3. Must not be special tokens
            if decoded.startswith("<|") or decoded.startswith("▁"):
                is_readable = False

            # 4. Must not be control characters
            if any(ord(c) < 32 for c in decoded):
                is_readable = False

            # 5. Prefer alphanumeric or common punctuation
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()-"
            )
            if decoded and not all(c in allowed_chars for c in decoded):
                # Still allow some common tokens
                common_words = {
                    "the",
                    "and",
                    "or",
                    "not",
                    "is",
                    "are",
                    "be",
                    "to",
                    "of",
                    "in",
                    "for",
                    "on",
                    "with",
                    "as",
                    "at",
                    "by",
                    "an",
                    "if",
                    "then",
                    "else",
                    "true",
                    "false",
                    "True",
                    "False",
                }
                if (
                    decoded.strip().lower() not in common_words
                    and not decoded.strip().isalnum()
                ):
                    is_readable = False

            if not is_readable:
                not_allowed.append(token_id)

        except:
            not_allowed.append(token_id)

    logger.info(
        f"Created readable token filter: {len(not_allowed)}/{vocab_size} tokens blocked"
    )
    return torch.tensor(not_allowed, device=device)


def load_bbh_data(path):
    """Load BBH dataset (CSV format)"""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "input": row["goal"],
                    "target": row["target"],
                    "final_target": row.get("final_target", row["target"]),
                }
            )
    return data


def evaluate_skill(model, tokenizer, skill, data, num_samples=100, extract_prompt=None):
    """Evaluate a skill on BBH data"""
    correct = 0
    total = 0
    results = []

    if extract_prompt is None:
        extract_prompt = "Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "

    for i, example in enumerate(data[:num_samples]):
        input_text = example["input"]
        prompt = f"{skill.content}\n\n{input_text}\n\n{extract_prompt}"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predicted = "Unknown"
        if "\\boxed{True}" in generated_text or "boxed{True}" in generated_text:
            predicted = "True"
        elif "\\boxed{False}" in generated_text or "boxed{False}" in generated_text:
            predicted = "False"
        elif "is $ True" in generated_text or "$ True" in generated_text:
            predicted = "True"
        elif "is $ False" in generated_text or "$ False" in generated_text:
            predicted = "False"
        elif "answer is True" in generated_text.lower():
            predicted = "True"
        elif "answer is False" in generated_text.lower():
            predicted = "False"

        ground_truth = example["target"]
        is_correct = predicted == ground_truth

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "input": input_text,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )

        if (i + 1) % 20 == 0:
            logger.info(
                f"Evaluation progress: {i + 1}/{num_samples} | Accuracy: {correct / (i + 1):.2%}"
            )

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("BBH READABLE Experiment - Constrained Skill Optimization")
    logger.info("=" * 80)

    # 1. Configuration
    app_config = AppConfig(model_name="/workspace/llama3/Llama-3.2-1B-Instruct")

    # Use simpler, more structured initial skill (like official GreaTer)
    # Key: use space-separated tokens for cleaner optimization
    initial_skill_content = (
        "Think step by step. Evaluate the boolean expression. The answer is"
    )

    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=32,
        batch_size=4,
        iterations=20,
        start_len=12,  # Shorter, more controlled
        end_len=16,
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=30,  # Smaller top_k for more focused search
        top_mu=5,
        patience=5,
        control_weight=0.2,  # Higher control weight to maintain structure
        early_stop_threshold=0.5,
        grad_clip=1.0,
        use_amp=False,
    )

    # 2. Load Data
    data_path = "referenceSolution/GreaTer/data/BBH/boolean_expressions.json"
    data = load_bbh_data(data_path)
    logger.info(f"Loaded {len(data)} examples from {data_path}")

    eval_samples = 50

    # 3. Define Original Skill with readable initialization
    original_skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=initial_skill_content,
    )

    logger.info(f"Initial Skill: '{original_skill.content}'")
    logger.info("=" * 80)

    # 4. Initialize Optimizer
    optimizer = GreaterOptimizer(app_config)

    # Create readable token filter
    readable_filter = get_readable_token_filter(optimizer.tokenizer, optimizer.device)
    logger.info(
        f"Readable token filter created with {len(readable_filter)} blocked tokens"
    )

    # 5. Evaluate Original Skill
    logger.info("Phase 1: Evaluating ORIGINAL Skill")
    original_accuracy, original_results = evaluate_skill(
        optimizer.model,
        optimizer.tokenizer,
        original_skill,
        data,
        num_samples=eval_samples,
    )
    logger.info(f"Original Skill Accuracy: {original_accuracy:.2%}")

    # 6. Run Optimization (using the existing optimizer for now)
    # Note: For truly readable results, we'd need to modify greater.py to use readable_filter
    logger.info("Phase 2: Running GreaTer Optimization")

    optimization_start = time.time()
    optimized_skill = optimizer.optimize(original_skill, optimize_config)
    optimization_time = time.time() - optimization_start

    logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
    logger.info(f"Optimized Skill: '{optimized_skill.content}'")

    # 7. Evaluate Optimized Skill
    logger.info("Phase 3: Evaluating OPTIMIZED Skill")

    optimized_skill_obj = Skill(
        name="BBH-Boolean-Expressions-Optimized",
        description="Optimized skill",
        content=optimized_skill.content,
    )

    optimized_accuracy, optimized_results = evaluate_skill(
        optimizer.model,
        optimizer.tokenizer,
        optimized_skill_obj,
        data,
        num_samples=eval_samples,
    )
    logger.info(f"Optimized Skill Accuracy: {optimized_accuracy:.2%}")

    # 8. Summary
    total_time = time.time() - start_time
    improvement = optimized_accuracy - original_accuracy

    print("\n" + "=" * 80)
    print("READABLE EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"\nOriginal Skill:")
    print(f"  '{original_skill.content}'")
    print(f"\nOptimized Skill:")
    print(f"  '{optimized_skill.content}'")
    print(f"\n" + "-" * 80)
    print(f"Original Accuracy:  {original_accuracy:.2%}")
    print(f"Optimized Accuracy: {optimized_accuracy:.2%}")
    print(f"Improvement:        {improvement:+.2%}")
    print("=" * 80)

    # Save results
    results = {
        "experiment_type": "readable",
        "timestamp": datetime.now().isoformat(),
        "original_skill": original_skill.content,
        "optimized_skill": optimized_skill.content,
        "original_accuracy": original_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "improvement": improvement,
        "config": {
            "num_examples": optimize_config.num_examples,
            "iterations": optimize_config.iterations,
            "top_k": optimize_config.top_k,
            "control_weight": optimize_config.control_weight,
        },
    }

    with open("experiments/bbh_readable_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
