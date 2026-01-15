"""
Debug script to identify the root cause of NaN in gradients
"""
import torch
import logging
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    app_config = AppConfig(
        model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
    )
    
    # Minimal configuration for debugging
    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=2,  # Minimal dataset
        batch_size=1,
        iterations=1,
        start_len=20,
        end_len=20,
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=10,  # Fewer candidates
        top_mu=2,
        patience=1,
        control_weight=0.0,  # Disable control weight
        early_stop_threshold=0.01,
        grad_clip=1.0,
        use_amp=False,
        grad_norm_epsilon=1e-6
    )

    bbh_skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
    )

    try:
        optimizer = GreaterOptimizer(app_config)
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        return

    logger.info("=" * 80)
    logger.info("DEBUG: Minimal configuration to identify NaN source")
    logger.info("=" * 80)
    
    optimized_skill = optimizer.optimize(bbh_skill, optimize_config)

    logger.info("=" * 80)
    logger.info("Optimization Complete!")
    logger.info("=" * 80)
    logger.info(f"Original Skill: {bbh_skill.content}")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    logger.info(f"Final Metrics: {optimized_skill.optimization_metrics}")

if __name__ == "__main__":
    main()