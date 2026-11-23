# 发布准备（GitHub / Hugging Face）

项目目录：`code/moondream3_310b/`（0.4 MB，44 个文件；本目录已在工作区 git 仓库提交）。

## 许可决策（已按官方 LICENSE 核实）

| 对象 | 许可 | 依据 |
|---|---|---|
| HF 权重 repo（文本 W4A16 + vision f16 + meta + tokenizer） | **BSL 1.1**（HF 字段 `license: other` + `license_name: business-source-license-1.1`） | 官方 LICENSE 第 4 行：Licensed Work 含 "Model Weights and any Derivatives"，Derivatives 明确包括 quantizations 与 weight-level conversions |
| GitHub 代码 repo（ops/ tools/ docs/） | **MIT** | 项目原创代码，可自由选择；与权重许可独立 |
| `ref/` 官方源码 | BSL 1.1（保留原 LICENSE） | 公开发布的 GitHub 仓库建议**移除 ref/** 或单独分层声明 |
| 权重再次分发 | 必须显眼展示 BSL 1.1 全文 | 官方 LICENSE 第 23 行 "You must conspicuously display this License on each original or modified copy" |

要点：

- **不能**给权重标 MIT/Apache-2.0——即使用自己的转换脚本，量化/转换权重属于 Derivative，仍受 BSL 1.1；
- Additional Use Grant 允许生产使用，但禁止与 M87 Labs 付费版竞争的 hosted/embedded 付费产品；组织内部使用不受限；
- Change Date：该版本首次公开发布后 **两年**，之后按 Apache-2.0；
- 商业授权联系 contact@m87.ai。

## GitHub（独立仓库，保留历史）

`gh` CLI 当前未安装；可安装后执行，或直接用 https/ssh remote：

```bash
# 方式 A：从工作区分出独立历史（推荐）
git subtree split --prefix=code/moondream3_310b -b moondream3-publish
git push <你的 GitHub 仓库地址> moondream3-publish:main

# 方式 B：干净副本
git archive moondream3-publish | tar -x -C ../moondream3_310b_repo
cd ../moondream3_310b_repo && git init && git add -A && git commit -m "init"
git remote add origin <repo> && git push -u origin main
```

## Hugging Face（model repo）

`hf` CLI 已安装（`C:\Users\25421\.local\bin\hf.exe`），登录后：

```powershell
hf auth login   # 输入你的 HF token（不要在文件/命令历史里公开）

# 1) 构建上传目录（硬链接，不额外占盘；--tar-weights 则打包权重）
& "<python>" tools/prepare_hf_upload.py --out D:\hf_upload_moondream3

# 2) 创建并上传
hf repo create <user>/moondream3-preview-ascend310b --repo-type model
hf upload <user>/moondream3-preview-ascend310b D:\hf_upload_moondream3 . --repo-type model
```

HF 上传目录结构（脚本生成）：

```
README.md            # MODEL_CARD.md 内容（模型卡）
LICENSE
weights/             # converted 权重原样（meta.txt + 文本 W4 + vision/*.f16）
tokenizer/           # starmie-v1 tokenizer 文件
tools/  docs/        # 转换/部署/基准工具与文档
```

板端使用：把 `weights/` 内容解压/拷贝到板端 `~/code/moondream3_310b/models/moondream3_310b/` 即可。

## 安全（已处理）

- 已从项目内移除板卡 IP / 本地私钥路径：`run_qa.py`、`run_demo.py` 使用环境变量 `MD3_BOARD`、`MD3_KEY`；
- 文档中的板卡地址改为 `<板卡IP>` 占位；
- 不要提交：私钥、`.hf_token`、`HF_TOKEN`、板卡密码。