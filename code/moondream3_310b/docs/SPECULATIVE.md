# 投机解码实验（prompt-lookup）

## 设计

- draft：在当前序列（prompt + 已生成）里找最后 n 个 token 的最近出现位置，取其后最多 k 个 token 作为草稿。
- verify：把 k 个 draft token 的 embedding 一次性送入 prefillEmbeds(T=k)，对每个位置取 lm_head logits（复用 24 层 prefill 的 KV cache）。
- 接受规则（greedy 等价）：
  - draft[0] 与当前 logits 的 argmax 比较；draft[i]（i≥1）与 allLogits[i-1] 的 argmax 比较；
  - 接受最长匹配前缀 m；下一 token = 对应位置 logits 的 argmax（m=0 时用当前 logits）。
- KV 安全：prefill 已把 draft 的 KV 写入 pos..pos+k-1；接受 m 个后，下一 token 用 forwardOne(nextTok, pos+m) 覆盖该位置，后续 token 依次覆盖，无脏读。

## 结果（板端，重复模式 "1 2 3 1 2 3 ..."）

基线:   TOKENS 37 225 38 225 39 225 ...  decode 2.38 s/tok  total 136 s
lookup: TOKENS 37 225 38 225 39 225 ...  decode 1.24 s/tok  total 138 s
        lookup rounds=4 accepted=11/16 (68.8%)

- 输出与基线 greedy 完全一致（正确性验证通过）。
- decode 阶段 1.92x；端到端无净收益，因为 verify 的 prefill 每 token ≈1.4 s，抵消了节省。
- 改进方向：MoE prefill 调度下沉（每 token prefill 成本降到 <1 s）、增大 k 摊薄、或对长文本（接受率高、verify 摊薄）使用。