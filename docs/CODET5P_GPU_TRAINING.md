# CodeT5+ 220M GPU 训练与交接

系统已经注册 `Salesforce/codet5p-220m`，注册名为 `codet5p-220m-base`。它是可训练的基础权重，不会在未经训练和测试的情况下作为检测结果来源。

## 打包前可选：把基础权重一起放进系统

联网机器在项目根目录执行：

```powershell
D:\software\anaconda\envs\drone\python.exe -m pip install -r backend\requirements-transformer.txt
D:\software\anaconda\envs\drone\python.exe scripts\download_deep_models.py --profile deep
```

下载结果位于：

```text
backend/models/pretrained/codet5p-220m/
```

系统会优先使用这个本地目录；没有该目录时，训练器会从 Hugging Face 的 `Salesforce/codet5p-220m` 下载并缓存。需要离线交接时，必须在压缩包中保留整个 `backend/models/pretrained/codet5p-220m/`。

## GPU 机器准备

1. 根据对方显卡驱动和 CUDA 版本安装官方 CUDA 版 PyTorch。
2. 安装其余依赖：

```powershell
python -m pip install -r backend\requirements-transformer.txt
```

3. 设置系统调用的深度学习 Python：

```powershell
$env:XIEZHI_DEEP_PYTHON = (Get-Command python).Source
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一条命令必须显示 `True` 和显卡名称。只显示 CPU 时不要提交 CodeT5+ 训练。

## 网页训练流程

管理员进入“模型中心 → 训练任务”，依次选择：

1. 模型版本：`CodeT5+ 220M / codet5p-220m-base`
2. 训练任务：`漏洞风险检测` 或 `恶意意图检测`
3. 目标语言：单一语言，或“全部可用语言”
4. 训练集文件：JSONL 或 CSV

网页选择的基础版本、任务和语言会真实传给训练进程，不只是写进任务记录。

训练完成后系统会保存候选版本。只有 validation、独立 test 以及每个有正负样本的语言分组都同时满足以下条件，候选版本才自动进入运行时：

- Precision 不低于 93%
- FPR 不高于 5%
- FNR 不高于 5%

未通过的候选版本保留在 `backend/models/candidates/web_training/`，不会替换当前检测模型。

## 训练集必要字段

```json
{
  "code": "待检测代码",
  "label": "vulnerable",
  "language": "go",
  "split": "train",
  "review_status": "approved",
  "label_confidence": 1.0,
  "family": "project-or-repository-family",
  "pair_id": "optional-fix-pair-id",
  "pair_slot": "before"
}
```

`label` 按任务使用：

- 漏洞任务：`vulnerable` / `benign`
- 恶意意图任务：`malicious` / `benign`

同一个 `family` 不能跨 train、validation、test，否则训练器会拒绝运行，防止仓库或补丁家族泄漏导致虚高指标。Go/PHP 的修复对建议保留 `pair_id` 和 `pair_slot`，训练器会加入 pair-ranking loss。

## 训练产物

- 注册表：`backend/models/codet5p_registry.json`
- 已发布模型：`backend/models/codet5p_artifacts/<version>/`
- 未通过门禁的候选：`backend/models/candidates/web_training/codet5p/<version>/`

通过门禁后，standard、deep 和 auto 升级路径会优先调用对应语言和任务的 CodeT5+ 版本；没有可用路由时继续使用现有 ByteCNN-TCN，不伪造结果。
