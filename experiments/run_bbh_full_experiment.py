"""
Full BBH Experiment: Optimization + Validation
Tests the fixed GreaTer optimizer on Boolean Expressions task
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
        logging.FileHandler("experiments/bbh_full_experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


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


def evaluate_skill(model, tokenizer, skill, data, num_samples=50, extract_prompt=None):
    """
    Evaluate a skill on BBH data
    Returns accuracy and detailed results
    """
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

        # Extract answer with multiple patterns
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

        if (i + 1) % 10 == 0:
            logger.info(
                f"Evaluation progress: {i + 1}/{num_samples} | Accuracy: {correct / (i + 1):.2%}"
            )

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("BBH Full Experiment - Testing Fixed GreaTer Optimizer")
    logger.info("=" * 80)

    # 1. Configuration
    app_config = AppConfig(model_name="/workspace/llama3/Llama-3.2-1B-Instruct")

    # Optimization config - moderate settings for reasonable runtime
    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=32,  # Training samples
        batch_size=4,  # Batch size for gradient computation
        iterations=10,  # Iterations per length
        start_len=16,  # Start with original skill length
        end_len=20,  # Optimize up to 20 tokens
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=50,
        top_mu=8,
        patience=3,
        control_weight=0.1,
        early_stop_threshold=0.5,
        grad_clip=1.0,
        use_amp=False,
    )

    # 2. Load Data
    data_path = "referenceSolution/GreaTer/data/BBH/boolean_expressions.json"
    data = load_bbh_data(data_path)
    logger.info(f"Loaded {len(data)} examples from {data_path}")

    # 3. Define Original Skill
    original_skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer.",
    )

    logger.info(f"Original Skill: {original_skill.content}")

    # 4. Initialize Optimizer
    optimizer = GreaterOptimizer(app_config)

    # 5. Evaluate Original Skill (before optimization)
    logger.info("=" * 80)
    logger.info("Phase 1: Evaluating ORIGINAL Skill")
    logger.info("=" * 80)

    original_accuracy, original_results = evaluate_skill(
        optimizer.model, optimizer.tokenizer, original_skill, data, num_samples=50
    )
    logger.info(f"Original Skill Accuracy: {original_accuracy:.2%}")

    # 6. Run Optimization
    logger.info("=" * 80)
    logger.info("Phase 2: Running GreaTer Optimization")
    logger.info("=" * 80)

    optimization_start = time.time()
    optimized_skill = optimizer.optimize(original_skill, optimize_config)
    optimization_time = time.time() - optimization_start

    logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    logger.info(
        f"Final Loss: {optimized_skill.optimization_metrics.get('final_loss', 'N/A')}"
    )

    # 7. Evaluate Optimized Skill
    logger.info("=" * 80)
    logger.info("Phase 3: Evaluating OPTIMIZED Skill")
    logger.info("=" * 80)

    optimized_skill_obj = Skill(
        name="BBH-Boolean-Expressions-Optimized",
        description="Optimized logical reasoning for boolean expressions",
        content=optimized_skill.content,
    )

    optimized_accuracy, optimized_results = evaluate_skill(
        optimizer.model, optimizer.tokenizer, optimized_skill_obj, data, num_samples=50
    )
    logger.info(f"Optimized Skill Accuracy: {optimized_accuracy:.2%}")

    # 8. Summary
    total_time = time.time() - start_time
    improvement = optimized_accuracy - original_accuracy

    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Original Skill Accuracy:  {original_accuracy:.2%} ({int(original_accuracy * 50)}/50)"
    )
    logger.info(
        f"Optimized Skill Accuracy: {optimized_accuracy:.2%} ({int(optimized_accuracy * 50)}/50)"
    )
    logger.info(f"Improvement:              {improvement:+.2%}")
    logger.info(f"Optimization Time:        {optimization_time:.1f}s")
    logger.info(f"Total Experiment Time:    {total_time:.1f}s")
    logger.info(
        f"Final Loss:               {optimized_skill.optimization_metrics.get('final_loss', 'N/A')}"
    )

    # 9. Save Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "original_skill": original_skill.content,
        "optimized_skill": optimized_skill.content,
        "original_accuracy": original_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "improvement": improvement,
        "optimization_time_seconds": optimization_time,
        "total_time_seconds": total_time,
        "optimization_metrics": optimized_skill.optimization_metrics,
        "config": {
            "num_examples": optimize_config.num_examples,
            "batch_size": optimize_config.batch_size,
            "iterations": optimize_config.iterations,
            "start_len": optimize_config.start_len,
            "end_len": optimize_config.end_len,
            "top_k": optimize_config.top_k,
            "top_mu": optimize_config.top_mu,
            "patience": optimize_config.patience,
            "control_weight": optimize_config.control_weight,
        },
        "original_results": original_results,
        "optimized_results": optimized_results,
    }

    output_path = "experiments/bbh_full_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Original Skill:  '{original_skill.content}'")
    print(f"Optimized Skill: '{optimized_skill.content}'")
    print(
        f"\nAccuracy: {original_accuracy:.2%} -> {optimized_accuracy:.2%} ({improvement:+.2%})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
