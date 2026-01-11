import os
import sys
import logging
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.core.interfaces import Skill
from skill_opt.optimizer.greater import GreaterOptimizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Configuration
    model_path = "/Volumes/TSU302/models/llama-3.2-3b-instruct"
    dataset_path = "referenceSolution/GreaTer/data/grade_school_math/data/train.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    # initialize Config
    app_config = AppConfig()
    # Manually override model name to path
    app_config.optimization.template_name = "llama-3" # Use our new template logic
    
    # We cheat a bit by patching the load function or passing it in config if supported
    # Use 'model_name' attr we inject into config via monkeypatch or if class supports.
    # AppConfig doesn't have model_name field explicitly but utils checks `hasattr(config, "model_name")`
    app_config.model_name = model_path
    
    opt_config = OptimizeConfig(
        dataset_path=dataset_path,
        num_examples=4, # Small batch for testing
        iterations=5,   # Short run
        start_len=2,    # Short initial control
        end_len=4,
        patience=2,
        batch_size=1,   # Safe for memory
        template_name="llama-3",
        top_k=20,       # Reduce search space for speed
        top_mu=2        # Reduce validations
    )

    # 2. Initialize Optimizer
    optimizer = GreaterOptimizer(app_config)

    # 3. Create a Dummy Skill to optimize
    # In GreaTer loop, the 'goal/target' comes from dataset, 
    # the 'Skill' object provides the INITIAL CONTROL ("! ! ...")
    initial_skill = Skill(
        name="GSM8K_Solver",
        description="Solves Grade School Math problems",
        content="Let's think step by step." # Initial seed
    )

    # 4. Evaluation (Zero-shot / Initial)
    test_data_path = dataset_path.replace("train.jsonl", "test.jsonl")
    if os.path.exists(test_data_path):
        print(f"\nEvaluating Initial Skill on Test Set...")
        pre_metrics = optimizer.evaluate_skill(initial_skill, test_data_path, num_examples=5, template_name="llama-3")
        print(f"Initial Accuracy: {pre_metrics['accuracy']:.2%}")

    # 5. Run Optimization
    print(f"\nStarting GSM8K Optimization with Llama-3...")
    optimized_skill = optimizer.optimize(initial_skill, opt_config)

    # 6. Evaluation (Post-Optimization)
    if os.path.exists(test_data_path):
        print(f"\nEvaluating Optimized Skill on Test Set...")
        post_metrics = optimizer.evaluate_skill(optimized_skill, test_data_path, num_examples=5, template_name="llama-3")
        print(f"Optimized Accuracy: {post_metrics['accuracy']:.2%}")

    # 7. Output Result
    print("\nOptimization Complete!")
    print(f"Final Control Prompt: {optimized_skill.content}")
    print(f"Final Loss: {optimized_skill.optimization_metrics.get('final_loss')}")

if __name__ == "__main__":
    main()
