# system-architecture Specification

## Purpose
TBD - created by archiving change scaffold-core-architecture. Update Purpose after archive.
## 需求
### 需求：模块化目录结构
系统代码必须遵循模块化设计，源代码位于 `src/skill_opt` 目录下，并包含以下独立子模块：
- `core`: 存放通用接口、配置和日志工具。
- `optimizer`: 存放 GreaTer 相关的优化逻辑。
- `bridge`: 存放 PromptBridge 相关的转移逻辑。
- `executor`: 存放 iFlow 相关的执行逻辑。

#### 场景：导入核心模块
开发者应该能够通过标准 Python 导入语句访问各个子模块，例如：
```python
from skill_opt.core.config import Config
from skill_opt.optimizer import SkillOptimizer
```

---

### 需求：基于接口的组件设计
核心组件（优化器、转移器、执行器）必须定义为抽象基类（ABC），以支持未来的扩展和不同实现的可插拔性。

#### 场景：扩展优化器
如果需要添加新的优化算法，开发者应继承 `SkillOptimizer` 基类并实现 `optimize` 方法，而无需修改调用方的代码。

---

### 需求：统一配置管理
系统必须使用 Pydantic 或类似库实现类型安全的配置管理，支持从 YAML/JSON 文件或环境变量加载配置。

#### 场景：加载实验配置
系统启动时，应能读取配置文件（如 `config.yaml`）并将其转换为强类型的配置对象，供后续模块使用。

