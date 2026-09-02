

import base64
import os
import subprocess


def suspicious_demo(user_input: str) -> str:
    # 故意保留多类高风险静态特征，便于演示规则命中与 AI 解释。
    token = os.getenv("DEMO_SECRET_TOKEN", "demo-token")
    decoded = base64.b64decode("ZWNobyBkZW1v").decode()
    command = decoded + " " + user_input
    subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"https://demo.invalid/collect?token={token}&input={user_input}"


if __name__ == "__main__":
    print("Static-analysis demo only; suspicious_demo() is not invoked.")
