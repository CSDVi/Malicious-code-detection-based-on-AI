"""Small, auditable mappings from detector categories to security frameworks."""

from __future__ import annotations

from typing import Any


API_SECURITY_BY_CATEGORY: dict[str, str] = {
    "Insecure Direct Object Reference": "API1:2023 对象级授权失效",
    "Missing Authorization Check": "API5:2023 功能级授权失效",
    "Client Side Only Authorization": "API5:2023 功能级授权失效",
    "JWT Verification Disabled": "API2:2023 身份认证失效",
    "Plaintext Password Handling": "API2:2023 身份认证失效",
    "Weak Session Management": "API2:2023 身份认证失效",
    "Default Credentials": "API2:2023 身份认证失效",
    "Missing Abuse Controls": "API6:2023 敏感业务流程缺少限制",
    "Unbounded Resource Consumption": "API4:2023 资源消耗缺少限制",
    "SSRF": "API7:2023 服务端请求伪造",
    "Debug Mode Enabled": "API8:2023 安全配置错误",
    "TLS Verification Disabled": "API8:2023 安全配置错误",
    "Permissive CORS": "API8:2023 安全配置错误",
    "Missing Security Headers": "API8:2023 安全配置错误",
    "Download or Remote Load": "API10:2023 不安全使用第三方API",
    "Unverified Artifact": "API10:2023 不安全使用第三方API",
}

SUPPLY_CHAIN_CATEGORIES = {
    "Download or Remote Load",
    "Remote Load or Write",
    "Download and Execute",
    "Install Hook Execution",
    "Unpinned Dependency",
    "Unverified Artifact",
    "Untrusted Plugin Loading",
    "Unsigned Update",
}

MALICIOUS_BEHAVIOR_CATEGORIES = {
    "WebShell",
    "Download and Execute",
    "Obfuscated Payload",
    "Install Hook Execution",
    "Credential Exfiltration",
    "Credential Collection",
    "Persistence",
}

CONFIGURATION_CATEGORIES = {
    "Debug Mode Enabled",
    "TLS Verification Disabled",
    "Permissive CORS",
    "Missing Security Headers",
    "Default Credentials",
}


def taxonomy_for_category(category: str) -> dict[str, Any]:
    domains = ["应用代码风险"]
    if category in API_SECURITY_BY_CATEGORY:
        domains.append("API安全")
    if category in SUPPLY_CHAIN_CATEGORIES:
        domains.append("软件供应链")
    if category in MALICIOUS_BEHAVIOR_CATEGORIES:
        domains.append("恶意行为")
    if category in CONFIGURATION_CATEGORIES:
        domains.append("安全配置")
    return {
        "risk_domains": domains,
        "api_security_category": API_SECURITY_BY_CATEGORY.get(category),
    }
