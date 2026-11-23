# 发布操作手册（照着填即可）

更新：2026-11-23

## A. GitHub（代码，MIT）

已打开页面：

```
https://github.com/new?name=moondream3_310b&description=Moondream%203%20Preview%20(9B%20MoE%20VLM)%20deployment%20on%20Ascend%20310B1%20/%20OrangePi%20AI%20Pro%2020T%3A%20AscendC%20ops%2C%20hybrid%20engine%2C%20benchmarks
```

| 字段 | 填什么 |
|---|---|
| Owner | `dong-12345` |
| Repository name | `moondream3_310b` |
| Description | `Moondream 3 Preview (9B MoE VLM) deployment on Ascend 310B1 / OrangePi AI Pro 20T: AscendC ops, hybrid engine, benchmarks` |
| Visibility | **Public**（要私有也行，后续命令不变） |
| Add a README / .gitignore / license | **全部不要勾**（仓库里已有这些文件） |

创建后**不要**执行 GitHub 页面上给的 git 命令（那一套会生成新 commit，导致 push 冲突）。
告诉我仓库地址，我在本地执行：

```bash
git push git@github.com:dong-12345/moondream3_310b.git moondream3-publish:main
# 或使用已导出的干净副本：cd D:\moondream3_310b_repo && git remote add origin ... && git push -u origin main
```

---

## B. Hugging Face（权重，BSL 1.1）

> ⚠️ 当前这台机器**直连 huggingface.co 失败**（HEAD 请求被重置，`hf whoami` 不通）。
> 需要先开代理/VPN，再打开下面的页面与执行上传命令。

已尝试打开：`https://huggingface.co/new`

| 字段 | 填什么 |
|---|---|
| Owner | 你的账号（未拿到用户名，登录后即见） |
| Model name | `moondream3-preview-ascend310b` |
| License | **Other**（列表里选 Other；BSL 1.1 不在标准列表） |
| **Model name 请保持全小写** | HF 不接受大写，别写成 Moondream3-Preview-Ascend310B |

选 **Other** 后，模型卡 YAML 里已写好：

```yaml
license: other
license_name: business-source-license-1.1
license_link: https://huggingface.co/moondream/moondream3-preview/blob/main/LICENSE
```

创建后（网页上选“不要”创建任何初始文件），回到本机上传：

```powershell
# 1) 登录（交互输入 token，不要写进文件）
hf auth login

# 2) 确认网络与账号
hf auth whoami

# 3) 上传（目录已备好：6.4GB，8914 文件）
hf upload <你的用户名>/moondream3-preview-ascend310b D:\hf_upload_moondream3 . --repo-type model
```

若网页版上传更顺手：直接把 `D:\hf_upload_moondream3` 下的 `README.md`、`LICENSE`、`LICENSE-MIT`、`weights/`、`tokenizer/` 拖进去。

---

## C. 创建完成后的收尾

1. 把 GitHub 仓库地址告诉我 → 我 push 代码；
2. 把 HF 用户名告诉我 → 我把 MODEL_CARD 里的 GitHub 链接改成真实地址并重建上传目录；
3. 上传完成后我会用 `hf download` 回拉关键文件（meta.txt、vision/L0.qkv.f16）校验完整性。