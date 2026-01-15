"""
Validation script for scaled BBH experiment
"""
import json
import csv
import logging
import torch
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_bbh_data(path):
    """Load BBH dataset (CSV format)"""
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'input': row['goal'],
                'target': row['target']
            })
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
        # Construct prompt
        input_text = example['input']
        prompt = f"{skill.content}\n\n{input_text}\n\n{extract_prompt}"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "is $ \\boxed{True}" in generated_text or "is $ \boxed{True}" in generated_text:
            predicted = "True"
        elif "is $ \\boxed{False}" in generated_text or "is $ \boxed{False}" in generated_text:
            predicted = "False"
        elif "\\boxed{True}" in generated_text or "boxed{True}" in generated_text:
            predicted = "True"
        elif "\\boxed{False}" in generated_text or "boxed{False}" in generated_text:
            predicted = "False"
        elif "is $ True" in generated_text:
            predicted = "True"
        elif "is $ False" in generated_text:
            predicted = "False"
        else:
            predicted = "Unknown"
        
        # Check correctness
        ground_truth = example['target']
        is_correct = predicted == ground_truth
        
        if is_correct:
            correct += 1
        
        total += 1
        
        results.append({
            'input': input_text,
            'generated': generated_text[-300:],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': is_correct
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{num_samples} | Accuracy so far: {correct/(i+1):.2%}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

def main():
    # 1. Configuration
    app_config = AppConfig(
        model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
    )
    
    # 2. Load Data
    data_path = "referenceSolution/GreaTer/data/BBH/boolean_expressions.json"
    data = load_bbh_data(data_path)
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    
    # 3. Load optimized skill from results
    results_path = "experiments/bbh_scaled_results.json"
    with open(results_path, 'r') as f:
        optimization_results = json.load(f)
    
    # 4. Define Skills
    original_skill = Skill(
        name="BBH-Boolean-Expressions-Original",
        description="Logical reasoning for boolean expressions",
        content=optimization_results['original_skill']
    )
    
    optimized_skill = Skill(
        name="BBH-Boolean-Expressions-Optimized",
        description="Optimized logical reasoning for boolean expressions",
        content=optimization_results['optimized_skill']
    )
    
    logger.info(f"Original Skill: {original_skill.content}")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    
    # 5. Initialize Optimizer
    optimizer = GreaterOptimizer(app_config)
    
    # 6. Evaluate Original Skill
    logger.info("=" * 80)
    logger.info("Evaluating ORIGINAL Skill (50 samples)")
    logger.info("=" * 80)
    original_accuracy, original_results = evaluate_skill(
        optimizer.model,
        optimizer.tokenizer,
        original_skill,
        data,
        num_samples=50
    )
    
    # 7. Evaluate Optimized Skill
    logger.info("=" * 80)
    logger.info("Evaluating OPTIMIZED Skill (50 samples)")
    logger.info("=" * 80)
    optimized_accuracy, optimized_results = evaluate_skill(
        optimizer.model,
        optimizer.tokenizer,
        optimized_skill,
        data,
        num_samples=50
    )
    
    # 8. Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Original Skill Accuracy: {original_accuracy:.2%} ({original_accuracy*100:.2f}%)")
    logger.info(f"Optimized Skill Accuracy: {optimized_accuracy:.2%} ({optimized_accuracy*100:.2f}%)")
    logger.info(f"Improvement: {(optimized_accuracy - original_accuracy):.2%} ({(optimized_accuracy - original_accuracy)*100:.2f}%)")
    
    # 9. Save Results
    results = {
        'original_skill': {
            'content': original_skill.content,
            'accuracy': original_accuracy,
            'correct': int(original_accuracy * 50),
            'total': 50,
            'results': original_results
        },
        'optimized_skill': {
            'content': optimized_skill.content,
            'accuracy': optimized_accuracy,
            'correct': int(optimized_accuracy * 50),
            'total': 50,
            'results': optimized_results
        },
        'improvement': optimized_accuracy - original_accuracy,
        'optimization_config': optimization_results['config']
    }
    
    output_path = "experiments/bbh_scaled_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
