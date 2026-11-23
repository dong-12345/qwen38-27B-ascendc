# 发布准备（GitHub / Hugging Face）

项目目录：`code/moondream3_310b/`（0.4 MB，44 个文件；本目录已在工作区 git 仓库提交）。

## 待你确认的 4 件事

| 项 | 当前准备 | 需要你确认 |
|---|---|---|
| 代码许可 | LICENSE 暂为 MIT | 是否改为 Apache-2.0 / 其他 |
| GitHub 仓库名 | 未创建 | `gh repo create <user>/<repo>` 的名字/账号 |
| HF 仓库 | 未创建 | `<user>/moondream3-preview-ascend310b`（model repo）是否合适 |
| HF 是否含权重 | 准备了 6.3 GB 权重（文本 W4 5.4GB + vision f16 0.85GB） | 上传权重、只传代码+文档、或权重打 tar |

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