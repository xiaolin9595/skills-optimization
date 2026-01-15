"""
Scaled BBH Experiment for GreaTer
Increased iterations, larger dataset, and tuned hyperparameters
"""
import logging
import torch
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 1. Configuration
    app_config = AppConfig(
        model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
    )
    
    # Scaled Configuration
    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=64,          # Increased from 16 to 64
        batch_size=8,             # Increased from 4 to 8
        iterations=20,            # Increased from 5 to 20
        start_len=20,
        end_len=25,               # Increased from 22 to 25
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=80,                 # Increased from 40 to 80
        top_mu=10,                # Increased from 5 to 10
        patience=3,               # Increased from 2 to 3
        control_weight=0.1,       # Reduced from 0.2 to 0.1
        early_stop_threshold=0.005  # Reduced from 0.01 to 0.005
    )

    # 2. Initial Skill
    bbh_skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
    )

    # 3. Initialize Optimizer
    try:
        optimizer = GreaterOptimizer(app_config)
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        return

    # 4. Run Optimization
    logger.info("=" * 80)
    logger.info("Starting SCALED BBH Optimization")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Dataset size: {optimize_config.num_examples}")
    logger.info(f"  - Batch size: {optimize_config.batch_size}")
    logger.info(f"  - Iterations: {optimize_config.iterations}")
    logger.info(f"  - Control length: {optimize_config.start_len} → {optimize_config.end_len}")
    logger.info(f"  - Top-K: {optimize_config.top_k}")
    logger.info(f"  - Top-Mu: {optimize_config.top_mu}")
    logger.info(f"  - Control weight: {optimize_config.control_weight}")
    logger.info("=" * 80)
    
    optimized_skill = optimizer.optimize(bbh_skill, optimize_config)

    logger.info("=" * 80)
    logger.info("Optimization Complete!")
    logger.info("=" * 80)
    logger.info(f"Original Skill: {bbh_skill.content}")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    logger.info(f"Final Metrics: {optimized_skill.optimization_metrics}")
    
    # Save results
    import json
    results = {
        'original_skill': bbh_skill.content,
        'optimized_skill': optimized_skill.content,
        'metrics': optimized_skill.optimization_metrics,
        'config': {
            'num_examples': optimize_config.num_examples,
            'batch_size': optimize_config.batch_size,
            'iterations': optimize_config.iterations,
            'start_len': optimize_config.start_len,
            'end_len': optimize_config.end_len,
            'top_k': optimize_config.top_k,
            'top_mu': optimize_config.top_mu,
            'control_weight': optimize_config.control_weight
        }
    }
    
    output_path = "experiments/bbh_scaled_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()