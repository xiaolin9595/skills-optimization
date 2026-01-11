# 设计：核心架构与模块划分

## 架构概览

遵循 `project.md` 中定义的架构模式，我们将系统划分为以下核心模块：

```mermaid
graph TD
    Config[配置管理] --> Optimizer
    Config --> Bridge
    Config --> Executor
    
    subgraph Core[核心层]
        Interfaces[抽象接口]
        Utils[通用工具]
        Logger[日志系统]
    end
    
    Optimizer[优化层 (GreaTer)] -->|Output: Optimized Skill| Bridge[转移层 (PromptBridge)]
    Bridge -->|Output: Adapted Skill| Executor[执行层 (iFlow)]
    Executor -->|Feedback: Metrics| Optimizer
```

## 目录结构设计

```text
solutionImpelement/
├── src/
│   ├── skill_opt/              # 顶层包名
│   │   ├── __init__.py
│   │   ├── core/               # 核心基础模块
│   │   │   ├── __init__.py
│   │   │   ├── config.py       # 配置加载
│   │   │   ├── interfaces.py   # 核心抽象基类 (ABC)
│   │   │   └── logger.py       # 日志封装
│   │   ├── optimizer/          # 优化器模块
│   │   │   ├── __init__.py
│   │   │   └── greater.py      # GreaTer 实现 (占位)
│   │   ├── bridge/             # 转移模块
│   │   │   ├── __init__.py
│   │   │   └── prompt_bridge.py # PromptBridge 实现 (占位)
│   │   ├── executor/           # 执行模块
│   │   │   ├── __init__.py
│   │   │   └── iflow_runner.py # iFlow 运行器 (占位)
│   │   └── utils/              # 工具函数
│   │       ├── __init__.py
│   │       └── file_io.py
├── tests/                      # 测试目录
├── pyproject.toml              # 项目配置与依赖
└── README.md
```

## 接口定义 (Interfaces)

我们将定义以下核心抽象基类 (`src/skill_opt/core/interfaces.py`)：

### 1. `SkillOptimizer`
负责对 Skill 进行梯度优化。
- `optimize(skill: Skill, config: OptimizeConfig) -> OptimizedSkill`

### 2. `SkillBridge`
负责跨模型 Skill 转移。
- `transfer(skill: OptimizedSkill, target_model: str) -> AdaptedSkill`

### 3. `SkillExecutor`
负责在特定环境中运行 Skill 并收集反馈。
- `execute(skill: AdaptedSkill, task: Task) -> ExecutionResult`

## 依赖管理

使用 `pyproject.toml` 管理依赖，核心依赖包括：
- `torch`: 深度学习计算
- `transformers`: 模型加载
- `pydantic`: 配置与数据模型验证
- `typer` / `click`: CLI 构建
