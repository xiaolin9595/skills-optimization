## 修改需求

### 需求：第二阶段 - 推理生成与答案提取 (Reasoning Generation)
系统必须强制模型生成推理过程，而非直接预测答案。

#### 场景：接口鲁棒性 (New)
`generate_reasoning` 接口必须支持批量输入（Batch Processing）并提供灵活的输出格式控制。
- **批量支持**: 必须接受 `input_ids [Batch, Seq]` 作为输入。
- **解码控制**: 必须通过参数（如 `decode=True`）控制返回解码后的文本列表（用于 Prompt 注入）还是原始 Token IDs（用于计算梯度或拼接）。
- **用途分离**:
    - 在 **Optimization Loop** 中，系统调用该接口获取 **Text** 以更新 `SkillPrompter`。
    - 在 **Selection/Validation** 中，系统调用该接口获取 **IDs** 以进行快速的 Loss 计算拼接。
