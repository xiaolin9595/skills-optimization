"""
可视化训练过程中的样本、label 和模型回答
"""
import json
import csv
import torch
import logging
from skill_opt.core.interfaces import Skill
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.optimizer.greater import GreaterOptimizer
from skill_opt.optimizer.prompter import SkillPrompter

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

def main():
    # Initialize
    app_config = AppConfig(
        model_name="/workspace/llama3/Llama-3.2-1B-Instruct"
    )
    
    optimizer = GreaterOptimizer(app_config)
    
    # Load data
    data_path = "referenceSolution/GreaTer/data/BBH/boolean_expressions.json"
    data = load_bbh_data(data_path)
    
    # Create a sample skill
    skill = Skill(
        name="BBH-Boolean-Expressions",
        description="Logical reasoning for boolean expressions",
        content=" proper logical reasoning and think step by step. Finally give the actual correct answer."
    )
    
    # Create a prompter to see the full input structure
    sample_data = data[0]
    
    print("=" * 80)
    print("训练数据可视化")
    print("=" * 80)
    
    print("\n【1. 原始数据样本】")
    print(f"  输入 (Input): {sample_data['input']}")
    print(f"  目标 (Target): {sample_data['target']}")
    
    print("\n【2. Skill 内容】")
    print(f"  {skill.content}")
    
    print("\n【3. 完整的 Prompt 结构】")
    print("=" * 80)
    
    # Create prompter with extract prompt
    extract_prompt = "Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
    
    prompter = SkillPrompter(
        goal=sample_data['input'],
        control_init=skill.content,
        target=sample_data['target'],
        tokenizer=optimizer.tokenizer,
        template_name="llama-3",
        extract_prompt=extract_prompt,
        device=optimizer.model.device
    )
    
    print(f"\n完整输入文本（包含所有特殊标记）:")
    print("-" * 80)
    full_text = optimizer.tokenizer.decode(prompter.input_ids)
    print(full_text)
    print("-" * 80)
    
    print(f"\nToken IDs (前50个):")
    print(f"  {prompter.input_ids[:50].tolist()}")
    
    print(f"\nToken IDs (后50个):")
    print(f"  {prompter.input_ids[-50:].tolist()}")
    
    print(f"\n【4. 各个切片的位置】")
    print(f"  总长度: {len(prompter.input_ids)} tokens")
    print(f"  _user_role_slice: {prompter._user_role_slice} (长度: {prompter._user_role_slice.stop - prompter._user_role_slice.start})")
    print(f"  _goal_slice: {prompter._goal_slice} (长度: {prompter._goal_slice.stop - prompter._goal_slice.start})")
    print(f"  _control_slice: {prompter._control_slice} (长度: {prompter._control_slice.stop - prompter._control_slice.start})")
    print(f"  _assistant_role_slice: {prompter._assistant_role_slice} (长度: {prompter._assistant_role_slice.stop - prompter._assistant_role_slice.start})")
    print(f"  _target_slice: {prompter._target_slice} (长度: {prompter._target_slice.stop - prompter._target_slice.start})")
    print(f"  _loss_slice: {prompter._loss_slice} (长度: {prompter._loss_slice.stop - prompter._loss_slice.start})")
    
    print(f"\n【5. 各个切片对应的文本】")
    print(f"\n  User Header:")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._user_role_slice])}")
    
    print(f"\n  Goal (用户问题):")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._goal_slice])}")
    
    print(f"\n  Control (Skill 提示词):")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._control_slice])}")
    
    print(f"\n  Assistant Header:")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._assistant_role_slice])}")
    
    print(f"\n  Target (正确答案):")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._target_slice])}")
    
    print(f"\n  Loss Slice (用于计算 Loss 的部分):")
    print(f"    {optimizer.tokenizer.decode(prompter.input_ids[prompter._loss_slice])}")
    
    print(f"\n【6. 模型训练时的 Loss 计算】")
    print("=" * 80)
    print(f"  模型需要预测的 tokens (Loss Slice):")
    target_tokens = prompter.input_ids[prompter._loss_slice]
    print(f"    Token IDs: {target_tokens.tolist()}")
    print(f"    Token 文本: {optimizer.tokenizer.decode(target_tokens)}")
    
    print(f"\n  模型输入的上下文 (Loss Slice 之前的所有 tokens):")
    context_tokens = prompter.input_ids[:prompter._loss_slice.start]
    print(f"    Token IDs (最后10个): {context_tokens[-10:].tolist()}")
    print(f"    Token 文本 (最后10个): {optimizer.tokenizer.decode(context_tokens[-10:])}")
    
    print(f"\n【7. 模型推理时的回答】")
    print("=" * 80)
    
    # Generate without target
    inference_prompt = f"{skill.content}\n\n{sample_data['input']}\n\n{extract_prompt}"
    inputs = optimizer.tokenizer(inference_prompt, return_tensors="pt").to(optimizer.model.device)
    
    with torch.no_grad():
        outputs = optimizer.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=optimizer.tokenizer.eos_token_id
        )
    
    generated_text = optimizer.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n  推理 Prompt:")
    print(f"    {inference_prompt}")
    
    print(f"\n  模型生成的完整回答:")
    print(f"    {generated_text}")
    
    print(f"\n  提取的答案:")
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
    
    print(f"    预测: {predicted}")
    print(f"    真实: {sample_data['target']}")
    print(f"    正确: {predicted == sample_data['target']}")
    
    print(f"\n【8. 梯度优化过程】")
    print("=" * 80)
    print(f"  在 GreaTer 优化中:")
    print(f"  1. 我们会修改 Control 部分的 tokens (Skill 提示词)")
    print(f"  2. 计算这些修改对 Loss 的影响 (梯度)")
    print(f"  3. 选择能够降低 Loss 的 token 替换")
    print(f"  4. 迭代优化直到找到最优的 Skill 提示词")
    
    print(f"\n  当前 Control 部分 (可优化):")
    control_tokens = prompter.input_ids[prompter._control_slice]
    print(f"    Token IDs: {control_tokens.tolist()}")
    print(f"    Token 文本: {optimizer.tokenizer.decode(control_tokens)}")
    print(f"    长度: {len(control_tokens)} tokens")
    
    print(f"\n  优化目标:")
    print(f"    - 降低 Loss (从当前值开始)")
    print(f"    - 保持语义连贯性")
    print(f"    - 提高模型在任务上的准确率")

if __name__ == "__main__":
    main()