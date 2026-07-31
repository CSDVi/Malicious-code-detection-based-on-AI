# 獬豸安码：面向软件供应链的AI代码风险检测与可解释分析系统

獬豸安码面向源代码、脚本与依赖配置的安全检测场景，融合规则引擎、机器学习模型和语义启发式分析，识别 SQL 注入、XSS、WebShell、命令执行、SSRF、路径穿越、反序列化风险、敏感信息泄露、恶意下载/远程加载、混淆逃逸/编码隐藏等风险，并输出证据行、风险原因、修复建议和综合置信度。

## 名称寓意

“獬豸”是古代传说中能辨曲直、识善恶的神兽，用来象征系统对恶意代码、隐蔽后门和供应链风险的辨析能力；“安码”表示对代码风险进行审视、识别和解释。

## 主要功能

- 单文件检测：支持 Python、PHP、JS/TS、Java/Kotlin、Go、C/C++、C#、Shell、PowerShell、Batch/CMD、HTML/HTA、配置与文本文件，以及 EXE/DLL 的只读 PE 分析。
- zip 项目批量扫描：输出项目风险等级、攻击类型分布和可分页的全部高风险文件。
- 多引擎融合：规则检测、TF-IDF 机器学习模型、语义启发式投票。
- 可解释报告：展示命中规则、CWE、证据片段、风险原因、三引擎投票、评分公式和修复建议，并支持导出 Markdown 报告。
- OWASP Top 10:2025：十个大类均有基础静态检测入口；覆盖边界见 [OWASP Top 10:2025 检测覆盖说明](docs/OWASP_TOP10_2025_COVERAGE.md)。
- 模型评测页面：展示准确率、召回率、F1、误报率和混淆矩阵。

## 目录结构

```text
獬豸安码/
├─ XiezhiCodeGuard.exe   桌面启动器
├─ README.md
├─ backend/      后端服务、检测引擎、模型、数据与测试
├─ frontend/     前端模板与静态资源
└─ docs/         参赛说明文档
```

## 启动方式

首次使用时安装依赖：

```powershell
cd D:\Z_TEMP_photo\AI\code
python -m pip install -r backend\requirements.txt
```

日常使用请双击根目录启动器，或运行：

```text
XiezhiCodeGuard.exe
```

启动器负责启动后端、使用系统默认浏览器打开检测页面，并在关闭检测页面后询问是否同时停止服务。不要直接运行 `python backend\app.py` 作为日常启动方式；直接启动后端时没有桌面启动器进程，无法显示关闭服务确认框。

检测界面地址为 `http://127.0.0.1:3000`，演示账号：`admin / admin123`。

## 第一阶段数据与模型

训练集统一保存在 `practicesets/`：漏洞数据位于 `vulnerability_detection/`，恶意代码数据按语言位于 `malware_detection/`，跨语言或混合数据位于 `malware_detection/other/`。流水线只读取白名单源码，不执行样本，不运行安装脚本，并对文件大小、包大小、路径穿越、符号链接、嵌套归档和压缩比设置门禁。

标签定义：

- `benign`：正常或明确安全代码。
- `vulnerable`：存在安全漏洞，但不代表作者具有恶意意图。
- `malicious`：具有下载执行、窃密、持久化、安装钩子或混淆载荷等主动恶意行为。

可复现命令：

```powershell
cd D:\Z_TEMP_photo\AI\code\backend
python -m attack_detection.safe_extract ..\practicesets\malware_detection\python\malicious-software-packages-dataset-samples-pypi-malicious_intent ..\practicesets\malware_detection\python\pypi_malicious_intent_static_v2 --password infected
python -m attack_detection.phase1_acquire ..\practicesets .\data\raw_metadata
python -m attack_detection.phase1_builder ..\practicesets .\data
python -m attack_detection.trainer --dataset .\data\processed\phase1_dataset.jsonl
```

当前模型采用两个独立的词级/字符级 TF-IDF 线性分类器，并只使用验证集进行概率校准和阈值选择。模型版本保存在 `backend/models/registry/`，当前版本由 `backend/models/registry.json` 指定，可回滚到历史版本。

评测页面同时展示支持语言的部署指标和包含未支持语言的压力测试指标。模型没有对应语言的正样本时，扫描页面明确显示 `Unavailable`，不会生成启发式伪概率。

# 新增分析能力配置

检测中心现在把本地证据、外部信誉和动态分析分开显示：

- 本地默认启用字符串/IOC 提取、JavaScript 静态去混淆、行为链组合检测，以及 EXE/DLL 的有界只读 PE 解析。它们不会执行上传代码。
- SHA256 外部信誉默认关闭。需要启用 VirusTotal 哈希查询时设置 `XIEZHI_REPUTATION_PROVIDER=virustotal` 和 `XIEZHI_VT_API_KEY`。只发送 SHA256，不上传源码；网络失败不会阻断本地检测。
- 隔离沙箱默认不提交样本。只有配置 `XIEZHI_SANDBOX_URL` 并显式设置 `XIEZHI_SANDBOX_AUTO_SCAN=1` 后，才会向指定的隔离服务提交样本；另设 `XIEZHI_SANDBOX_POLL=1` 才会有界轮询 `/v1/samples/{id}`。服务端必须保证一次性 VM/容器、无真实互联网、运行后销毁；本项目不会在 Flask 进程中调用 `subprocess`。
- 信誉查询和沙箱的返回结果只是辅助证据。没有可定位的本地代码/二进制证据时，结论不会被强行改成“恶意”。

支持的单文件扩展名以 `backend/attack_detection/languages.py` 为准，包含 `.ps1/.psm1/.psd1`、`.bat/.cmd`、`.html/.htm/.xhtml/.hta` 等命令脚本与网页载荷格式。PE/DLL 解析只读取 DOS/PE 头、节区、熵、覆盖区和字符串/导入线索；它不是完整逆向工具，也不能替代人工分析或真正的沙箱。

## GitHub 精简仓库说明

为满足 GitHub 文件大小限制并避免提交样本、历史记录和机器本地数据，仓库不包含以下内容：

- `practicesets/`、`testsets/` 和 `backend/data/` 中的训练语料、测试样本、扫描历史及中间数据；
- `artifacts/`、模型候选、历史回滚副本、缓存和运行日志；
- `backend/models/codet5p_artifacts/` 中单文件约 512 MB 的 CodeT5+ 微调权重。
- 首页粒子效果的 81 MB GLB 原始模型；仓库保留并优先使用约 1 MB 的预计算粒子数据。

仓库保留规则引擎、静态证据分析、TF-IDF、XGBoost、ByteCNN-TCN、GATv2 的当前运行产物，以及 CodeT5+ 的接入代码和注册信息。未单独放置 CodeT5+ 微调权重及对应深度学习解释器时，相关深度引擎会明确返回 `Unavailable`，其他本地检测能力仍可运行。

深度模型的环境和下载边界见 [多语言深度模型下载与接入](docs/深度模型下载与接入.md)。
