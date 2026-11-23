# Moondream 3 Preview 开源状态与许可证

> 结论：**不是 OSI 开源，而是 "open weights"（权重开放）**。权重以 **Business Source License 1.1 + Additional Use Grant** 发布（M87 Labs, Inc.，2025），首个公开版本 2 年后自动转为 Apache-2.0。**代码/权重可下载、可修改、可量化、可再分发**，但**不得**把它做成与 M87 付费产品竞争的托管/嵌入式服务。

## 1. 许可条款（LICENSE.md 实测）

| 项 | 内容 |
|---|---|
| License | Business Source License (BSL) 1.1 |
| Licensor | M87 Labs, Inc. |
| Licensed Work | "Moondream 3 (Preview)"，含 **Model Weights** 与 Derivatives（微调、合并、**量化**、权重转换等） |
| Additional Use Grant | 允许 production use，前提是**不向第三方提供与 M87 付费版本竞争的 hosted / embedded 服务** |
| Change Date | 本版本首次公开发布后 **2 年** |
| Change License | Apache License 2.0 |

允许（无需额外协议）：
- 个人项目、研究、benchmark、微调、合并、**量化**、权重 delta；
- 公司**内部**生产使用（含 affiliates）；
- 在自己产品里使用（不与 M87 付费产品实质重叠）；
- 免费/零价格服务（社区 demo、非商业工具）；
- 再分发（需附带本许可证，Derivatives 同样受 BSL 约束）。

需要单独商业协议：
- 售卖暴露同类视觉/通用 AI 能力的 **hosted API**；
- 给客户做 Moondream 托管（"Moondream-as-a-service"）；
- 把权重/代码嵌入**付费 SDK 或设备**并交付同等能力；
- 与 M87 付费版本实质重叠的 B2B CV/标注/通用 API。

> 注：官方还发布了 4bit MLX 量化权重 [`moondream/md3p-int4`](https://huggingface.co/moondream/md3p-int4)，其 README 标注 `license: apache-2.0`；但按上游 LICENSE.md，"quantizations" 属于 Derivatives，仍应同时遵守 BSL 1.1。商用/再分发前建议复核并联系 contact@m87.ai。

## 2. 我们如何使用

- 本项目从 `moondream/md3p-int4`（MoE experts int4 group64 + 其余 bf16）转换出自用 W4A16 格式；
- 转换产物是 **Derivative**，仓库内保留 `ref/LICENSE.md` 与 `MODIFICATIONS.md`（格式重打包、bf16→fp16、int4 group64→group32、只保留文本+视觉塔）；
- 用途：个人研究/端侧部署实验，非对外托管服务。

## 3. 参考链接

- 模型卡：https://huggingface.co/moondream/moondream3-preview
- 许可证：https://huggingface.co/moondream/moondream3-preview/blob/main/LICENSE.md
- 官方 int4：https://huggingface.co/moondream/md3p-int4
- 发布说明：https://moondream.ai/blog/moondream-3-preview
- 模型总览：https://moondream.ai/models
