# 獬豸安码（比赛提交版）

这是可直接运行的 Windows x64 完整成果包，保留前后端源代码、启动器、便携
Python 运行环境，以及项目使用的三个模型：XGBoost、GATv2、CodeT5+。
检测历史、扫描任务、训练任务、日志、缓存、未发布候选模型及旧版本归档均未包含。

## 一键启动

1. 将整个 `獬豸安码` 文件夹完整解压到本机磁盘。
2. 保持目录结构不变，双击 `XiezhiCodeGuard.exe`。
3. 等待浏览器自动打开 `http://127.0.0.1:3000`。

默认登录信息：

```text
用户名：admin
密码：admin123
```

启动器优先使用随包提供的 `python\python.exe`，无需提前安装 Python 或依赖。
首次启动会自动创建空白数据库和管理员账号，因此历史页初始为空属于正常现象。

## 目录结构

```text
獬豸安码/
├─ XiezhiCodeGuard.exe       # 双击启动入口
├─ python/                   # Windows x64 便携运行环境及依赖
├─ backend/                  # Flask 后端、检测引擎与当前启用模型
├─ frontend/                 # 页面模板与静态资源
├─ demo_samples/             # 功能演示样例
├─ README.md
└─ 一键运行说明.txt
```

请勿单独移动 EXE，也不要删除 `backend\models` 中的当前模型文件；标准/深度检测
依赖这些本地模型，全部放在包内是为了保证离线运行。

## 三个模型的职责

- XGBoost：单文件与批量快速初筛，覆盖已发布的语言路由；
- CodeT5+：标准/深度模式中的代码语义分析；
- GATv2：项目级调用关系与跨文件图分析。

## 命令行启动

若需要查看控制台输出，可在项目根目录打开 PowerShell：

```powershell
.\python\python.exe backend\app.py
```

停止服务时按 `Ctrl+C`。

## 可选配置

```powershell
$env:XIEZHI_PORT = "3000"
$env:XIEZHI_ADMIN_PASSWORD = "自行设置的密码"
$env:XIEZHI_SECRET_KEY = "自行生成的随机密钥"
.\XiezhiCodeGuard.exe
```

- `XIEZHI_PORT`：监听端口，默认 `3000`；
- `XIEZHI_ADMIN_PASSWORD`：只在首次创建空白数据库时设置管理员密码；
- `XIEZHI_SECRET_KEY`：生产环境会话密钥；
- `XIEZHI_DEEP_PYTHON`：可选的深度学习解释器；不设置时使用随包 Python。

## 从系统 Python 运行（可选）

便携环境损坏或需要二次开发时，可使用 Python 3.12/3.13 创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m pip install -r backend\requirements-transformer.txt
python backend\app.py
```

## 运行后生成的本地内容

程序运行后会重新生成以下内容，它们已写入 `.gitignore`，提交源代码时无需包含：

- `backend\data\attack_detection.db*`：账号、检测记录和任务记录；
- `backend\logs\`：启动器及后端日志；
- `backend\data\tmp\`：扫描/推理临时文件；
- `backend\models\candidates\`、`backend\models\archive\`：后续训练产物；
- `__pycache__`、`.pytest_cache`、`*.pyc`：Python 测试与字节码缓存。
