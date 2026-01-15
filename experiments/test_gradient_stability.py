"""
Test script to verify gradient stability improvements
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
    # Test with gradient stability improvements
    app_config = AppConfig(
        model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
    )
    
    # Configuration with gradient stability features
    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=16,
        batch_size=4,
        iterations=3,
        start_len=20,
        end_len=22,
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=40,
        top_mu=5,
        patience=2,
        control_weight=0.1,
        early_stop_threshold=0.01,
        # New gradient stability parameters
        grad_clip=1.0,          # Gradient clipping threshold
        use_amp=False,          # Enable mixed precision
        grad_norm_epsilon=1e-6  # Epsilon for gradient normalization
    )

    # Initial Skill
    bbh_skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
    )

    # Initialize Optimizer
    try:
        optimizer = GreaterOptimizer(app_config)
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        return

    # Run Optimization
    logger.info("=" * 80)
    logger.info("Testing Gradient Stability Improvements")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Grad Clip: {optimize_config.grad_clip}")
    logger.info(f"  - Use AMP: {optimize_config.use_amp}")
    logger.info(f"  - Grad Norm Epsilon: {optimize_config.grad_norm_epsilon}")
    logger.info(f"  - Control Weight: {optimize_config.control_weight}")
    logger.info("=" * 80)
    
    optimized_skill = optimizer.optimize(bbh_skill, optimize_config)

    logger.info("=" * 80)
    logger.info("Optimization Complete!")
    logger.info("=" * 80)
    logger.info(f"Original Skill: {bbh_skill.content}")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    logger.info(f"Final Metrics: {optimized_skill.optimization_metrics}")
    
    # Check if final_loss is still inf
    final_loss = optimized_skill.optimization_metrics.get('final_loss', None)
    if final_loss == float('inf'):
        logger.warning("⚠️  Final loss is still infinity! Further improvements needed.")
    elif final_loss is None:
        logger.warning("⚠️  Final loss is None! Check optimization metrics.")
    else:
        logger.info(f"✅ Final loss is finite: {final_loss:.4f}")

if __name__ == "__main__":
    main()