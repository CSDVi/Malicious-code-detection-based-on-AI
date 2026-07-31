"""Auditable OWASP Top 10:2025 baseline coverage registry.

`baseline` means the product has at least one concrete static signal for the
category.  It deliberately does not mean every CWE, framework, data flow, or
business-logic variant can be detected by static analysis.
"""

from __future__ import annotations

from typing import Any


OWASP_TOP10_2025: tuple[dict[str, Any], ...] = (
    {
        "id": "A01:2025",
        "name": "访问控制失效",
        "status": "baseline",
        "detectors": ["AUTHZ-001", "PATH-001", "PATH-002", "SSRF-001"],
        "limitations": "对象级授权仍需结合路由、身份与数据所有权做跨文件分析。",
    },
    {
        "id": "A02:2025",
        "name": "安全配置错误",
        "status": "baseline",
        "detectors": ["MISCONFIG-001", "MISCONFIG-002", "MISCONFIG-003", "ERROR-001"],
        "limitations": "运行环境、网关和云平台中的最终配置不一定存在于源码内。",
    },
    {
        "id": "A03:2025",
        "name": "软件供应链失效",
        "status": "baseline",
        "detectors": ["SUPPLY-001", "DL-001", "DL-002"],
        "limitations": "依赖漏洞、签名和制品来源还需要联网情报、SBOM 与构建证明。",
    },
    {
        "id": "A04:2025",
        "name": "密码学失效",
        "status": "baseline",
        "detectors": ["CRYPTO-001", "CRYPTO-002", "CRYPTO-003", "SECRET-001"],
        "limitations": "密钥生命周期、实际 TLS 配置和密码参数需要部署环境证据。",
    },
    {
        "id": "A05:2025",
        "name": "注入",
        "status": "baseline",
        "detectors": ["SQL-001", "SQL-002", "SQL-003", "XSS-001", "XSS-002", "CMD-001", "CMD-002"],
        "limitations": "当前以常见源码模式和保守数据流为主，尚未覆盖所有框架的污点传播。",
    },
    {
        "id": "A06:2025",
        "name": "不安全设计",
        "status": "baseline",
        "detectors": ["DESIGN-001", "DESIGN-002", "DESIGN-003"],
        "limitations": "业务逻辑和架构缺陷不能仅从单个源码文件可靠判断，需要威胁建模与人工评审。",
    },
    {
        "id": "A07:2025",
        "name": "身份认证失效",
        "status": "baseline",
        "detectors": ["AUTHN-001", "AUTHN-002", "SECRET-001"],
        "limitations": "账户恢复、MFA 和会话撤销策略通常需要跨服务配置与运行验证。",
    },
    {
        "id": "A08:2025",
        "name": "软件或数据完整性失效",
        "status": "baseline",
        "detectors": ["DESER-001", "DL-001", "DL-002", "SUPPLY-001"],
        "limitations": "是否实际验证签名与可信发布链需要制品、密钥和流水线证据。",
    },
    {
        "id": "A09:2025",
        "name": "安全日志与告警失效",
        "status": "baseline",
        "detectors": ["LOG-001", "LOG-002"],
        "limitations": "缺少某类日志属于全局否定事实，单文件静态分析只能发现明显泄露或吞错。",
    },
    {
        "id": "A10:2025",
        "name": "异常条件处理不当",
        "status": "baseline",
        "detectors": ["ERROR-001", "ERROR-002", "ERROR-003", "DESIGN-003"],
        "limitations": "真实故障恢复和资源耗尽行为仍需故障注入、压力测试或沙箱运行。",
    },
)


def coverage_summary() -> dict[str, Any]:
    """Return a small immutable-style summary suitable for API responses."""

    return {
        "standard": "OWASP Top 10:2025",
        "level": "baseline",
        "covered_categories": len(OWASP_TOP10_2025),
        "total_categories": 10,
        "absolute_coverage_claimed": False,
        "categories": [dict(item) for item in OWASP_TOP10_2025],
    }
