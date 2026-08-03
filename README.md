# 獬豸安码运行说明

## 一、文件要求

请确保以下文件和目录位于同一个根目录中，并保持相对位置不变：

```text
獬豸安码/
├─ XiezhiCodeGuard.exe
├─ backend/
└─ frontend/
```

不要只复制或移动 `XiezhiCodeGuard.exe`，否则启动器将无法找到后端和前端文件。

## 二、搭建运行环境

### 1. 安装 Python

本系统建议使用 **Python 3.12（64 位）**。

安装 Python 时请勾选 **Add Python to PATH**。安装完成后打开 PowerShell，执行：

```powershell
python --version
```

如果显示 `Python 3.12.x`，说明 Python 已安装成功。

如果 `python` 命令不可用，也可以检查 Python 启动器：

```powershell
py -3.12 --version
```

### 2. 创建虚拟环境

在存放 `XiezhiCodeGuard.exe` 的根目录打开 PowerShell，然后执行：

```powershell
python -m venv .venv
```

如果电脑上需要通过 `py` 调用 Python，则执行：

```powershell
py -3.12 -m venv .venv
```

### 3. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示禁止运行脚本，请先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

该设置只对当前 PowerShell 窗口生效。

### 4. 安装后端依赖

虚拟环境激活后执行：

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

依赖安装完成后，运行环境即搭建完毕。以上步骤通常只需在首次运行前执行一次。

## 三、启动系统

### 方法一：双击 EXE 启动（推荐）

直接双击根目录中的：

```text
XiezhiCodeGuard.exe
```

启动器会自动查找 Python、启动后端服务并打开浏览器。

系统默认访问地址：

```text
http://127.0.0.1:3000
```

默认登录信息：

```text
用户名：admin
密码：admin123
```

关闭浏览器页面后，启动器会询问是否同时关闭后端服务。

### 方法二：通过 PowerShell 启动

如果双击 EXE 无法启动，或者需要查看错误信息，可在根目录打开 PowerShell，执行：

```powershell
.\.venv\Scripts\Activate.ps1
python backend\app.py
```

看到服务启动信息后，在浏览器中打开：

```text
http://127.0.0.1:3000
```

停止服务时，回到 PowerShell 窗口按 `Ctrl+C`。

## 四、常见问题

### 1. 提示找不到 Python

确认 Python 是否可用：

```powershell
python --version
```

也可以明确指定 EXE 使用项目虚拟环境中的 Python：

```powershell
$env:XIEZHI_PYTHON = (Resolve-Path ".\.venv\Scripts\python.exe").Path
.\XiezhiCodeGuard.exe
```

### 2. 提示缺少 Python 模块

使用虚拟环境中的 Python 重新安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 3. 提示端口 3000 被占用

这通常表示系统已经启动，或者其他程序正在使用端口 `3000`。先关闭之前启动的后端或占用该端口的程序，再重新双击 EXE。

可通过以下命令查看端口占用情况：

```powershell
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
```

### 4. 启动器长时间停留在加载界面

首次启动加载模型可能需要较长时间。可以在 PowerShell 中延长等待时间后启动：

```powershell
$env:XIEZHI_STARTUP_TIMEOUT_SECONDS = "180"
.\XiezhiCodeGuard.exe
```

如果仍然无法启动，请使用 PowerShell 直接运行 `python backend\app.py`，根据窗口中显示的错误信息检查 Python 和依赖安装情况。
