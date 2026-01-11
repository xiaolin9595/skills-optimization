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
        model_name="meta-llama/Meta-Llama-3-8B-Instruct"  # You'd use a local path or HF ID here
    )
    
    # Official BBH Initial Prompt and Extractor
    # Note: Llama-3-8B Instruct is recommended for GreaTer experiments
    # We use boolean_expressions as a representative BBH task.
    optimize_config = OptimizeConfig(
        dataset_path="referenceSolution/GreaTer/data/BBH/boolean_expressions.json",
        num_examples=16,          # Small batch for validation
        batch_size=4,
        iterations=5,             # Short run for verification
        start_len=20,             # GreaTer usually starts with a fixed length
        end_len=22,               # Sequential increasing to verify logic
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=40,
        top_mu=5,
        patience=2,
        control_weight=0.2,
        early_stop_threshold=0.01
    )

    # 2. Initial Skill
    # Official BBH prompt for GreaTer
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
        logger.info("Falling back to mock initialization for script verification")
        # In a real environment with GPU and models, this would load normally.
        return

    # 4. Run Optimization
    logger.info("Starting BBH Optimization...")
    optimized_skill = optimizer.optimize(bbh_skill, optimize_config)

    logger.info("Optimization Complete!")
    logger.info(f"Original Skill: {bbh_skill.content}")
    logger.info(f"Optimized Skill: {optimized_skill.content}")
    logger.info(f"Final Metrics: {optimized_skill.optimization_metrics}")

if __name__ == "__main__":
    main()
