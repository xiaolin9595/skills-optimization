# 提案：构建核心架构与工程脚手架

## 背景
根据 `openspec/project.md`，我们需要构建一个 "Agent Skill 自动优化系统"。目前项目目录为空，缺乏基础的工程结构、依赖管理和模块划分。为了支持 GreaTer 优化器、PromptBridge 和 iFlow 执行器的集成，我们需要一个清晰、模块化的 Python 项目架构。

## 目标
建立项目的核心架构，包括：
1. **工程脚手架**: 配置 Python 项目环境 (`pyproject.toml`)，管理 PyTorch, Transformers 等核心依赖。
2. **目录结构**: 按照 `project.md` 中的架构模式（优化器、转移器、执行器分离）创建源码目录。
3. **核心接口**: 定义系统关键组件的抽象基类（Abstract Base Classes），确保模块间的解耦和规范交互。

## 范围
- **包含**:
    - `pyproject.toml` 依赖配置
    - `src/` 目录结构创建
    - `Optimizer`, `Bridge`, `Executor` 等核心抽象类定义
    - 基础配置与日志模块
- **不包含**:
    - GreaTer 算法的具体移植（将在后续变更中进行）
    - PromptBridge 的具体实现
    - iFlow 的实际对接代码

## 成功标准
- 项目可以成功安装依赖 (`pip install -e .` 或类似)。
- 核心模块可以通过导入测试。
- 能够通过简单的单元测试验证接口定义的正确性。
