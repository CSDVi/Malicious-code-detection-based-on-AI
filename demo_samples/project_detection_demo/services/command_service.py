import subprocess


def run_report(name: str) -> str:
    # 故意的跨文件命令注入特征，供静态检测演示。
    completed = subprocess.run(
        "echo report-" + name,
        shell=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
