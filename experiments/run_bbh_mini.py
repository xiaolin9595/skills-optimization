import logging
import torch
import os
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Configure logging to see our new trace logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set core to DEBUG for deep trace
logging.getLogger('skill_opt.optimizer.greater_core').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    model_path = "/Volumes/TSU302/models/llama-3.2-3b-instruct"
    dataset_path = "referenceSolution/GreaTer/data/BBH/boolean_expressions.json"
    
    app_config = AppConfig()
    app_config.model_name = model_path
    
    # Mini Trace Configuration
    optimize_config = OptimizeConfig(
        dataset_path=dataset_path,
        num_examples=1,           # 1 sample
        iterations=1,             # 1 epoch
        start_len=20,             
        end_len=20,               # No increasing for trace
        batch_size=1,
        template_name="llama-3",
        extract_prompt="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
        top_k=20,
        top_mu=3,                 # Test 3 candidates
        patience=1,
        control_weight=0.2,
        early_stop_threshold=0.0
    )

    initial_skill = Skill(
        name="BBH-Mini-Trace",
        description="Trace run for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
    )

    logger.info("Starting Mini-Trace Experiment...")
    try:
        optimizer = GreaterOptimizer(app_config)
        # Verify dataset loads
        data = optimizer._load_training_data_raw(optimize_config)
        logger.info(f"Successfully loaded {len(data)} examples.")
        
        # Run optimization
        optimized_skill = optimizer.optimize(initial_skill, optimize_config)
        
        logger.info("Trace Run Complete!")
        logger.info(f"Final Skill: {optimized_skill.content}")
        
    except Exception as e:
        logger.error(f"Trace Experiment Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
