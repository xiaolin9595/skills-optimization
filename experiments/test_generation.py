"""
Test script to see what the model actually generates
"""
import torch
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig
from skill_opt.optimizer.greater import GreaterOptimizer

# Initialize
app_config = AppConfig(
    model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
)
optimizer = GreaterOptimizer(app_config)

# Test skill
skill = Skill(
    name="Test",
    description="Test",
    content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
)

# Test input
test_input = "not ( True ) and ( True ) is"

# Construct prompt
prompt = f"{skill.content}\n\n{test_input}"

# Tokenize
inputs = optimizer.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(optimizer.model.device) for k, v in inputs.items()}

print(f"Prompt: {prompt}")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Input IDs: {inputs['input_ids']}")
print(f"Decoded input: {optimizer.tokenizer.decode(inputs['input_ids'][0])}")
print()

# Generate
with torch.no_grad():
    outputs = optimizer.model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=optimizer.tokenizer.eos_token_id
    )

# Decode
generated_text = optimizer.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text:\n{generated_text}")
print()

# Try with extract prompt
extract_prompt = "Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
prompt_with_extract = f"{skill.content}\n\n{test_input}\n\n{extract_prompt}"

inputs2 = optimizer.tokenizer(prompt_with_extract, return_tensors="pt", padding=True, truncation=True)
inputs2 = {k: v.to(optimizer.model.device) for k, v in inputs2.items()}

print(f"Prompt with extract: {prompt_with_extract}")
print()

with torch.no_grad():
    outputs2 = optimizer.model.generate(
        **inputs2,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=optimizer.tokenizer.eos_token_id
    )

generated_text2 = optimizer.tokenizer.decode(outputs2[0], skip_special_tokens=True)
print(f"Generated text with extract prompt:\n{generated_text2}")