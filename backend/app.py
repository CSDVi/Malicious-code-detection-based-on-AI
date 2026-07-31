"""Xiezhi CodeGuard Flask application."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, flash, redirect, request, url_for
from flask_login import LoginManager
from werkzeug.exceptions import RequestEntityTooLarge

from attack_detection.database import (
    get_user_by_id,
    init_database,
    reconcile_interrupted_scan_jobs,
)
from attack_detection.jobs import scan_jobs
from attack_detection.project_scanner import MAX_ARCHIVE_SIZE
from web.routes.attack_routes import attack_bp
from web.routes.main_routes import main_bp


DISPLAY_ZH = {
    "benign": "正常", "malicious": "恶意", "vulnerable": "存在漏洞",
    "safe": "安全", "critical": "严重", "high": "高危", "medium": "中危", "low": "低危",
    "unknown": "需复核", "unavailable": "不可用", "positive": "阳性", "negative": "阴性",
    "uncertain": "不确定", "risky": "有风险", "clear": "未发现风险",
    "auto": "自动", "quick": "快速", "standard": "标准", "deep": "深度",
    "queued": "排队中", "running": "检测中", "cancelling": "正在停止",
    "cancelled": "已停止", "completed": "已完成", "failed": "失败",
    "not_run": "未执行", "skipped": "已跳过", "disabled": "已停用",
    "submitted": "已提交", "finished": "已完成", "terminated": "已终止",
    "rule_engine": "规则引擎", "xgboost_malicious": "XGBoost 恶意代码模型",
    "xgboost_project_malicious": "XGBoost 项目恶意代码模型",
    "static_evidence": "静态证据分析", "codet5p": "CodeT5+ 220M 语义模型",
    "gatv2": "GATv2 项目图模型", "pe_static": "PE/DLL 只读解析",
    "hash_reputation": "SHA256 外部信誉", "isolated_sandbox": "隔离动态沙箱",
    "malicious_intent": "恶意代码模型", "vulnerability_risk": "已下线任务",
    "linear_svm": "线性支持向量机", "logistic_regression": "逻辑回归",
    "python": "Python", "javascript": "JavaScript/TypeScript", "typescript": "TypeScript",
    "java": "Java", "php": "PHP", "bash": "Bash/Shell", "powershell": "PowerShell",
    "batch": "批处理/CMD", "binary": "Windows 可执行文件", "config": "config",
    "json": "JSON", "yaml": "YAML", "toml": "TOML", "ini": "INI", "conf": "CONF",
    "text": "TXT",
    "go": "Go", "c": "C", "cpp": "C++", "csharp": "C#", "ruby": "Ruby",
    "npm_official_registry": "npm 官方仓库", "pypi_official_registry": "PyPI 官方仓库",
    "datadog_compromised_package_diff": "Datadog 受感染版本差异",
    "datadog_malicious_intent": "Datadog 主动恶意包", "owasp_benchmark_java": "OWASP Java 基准",
    "nist_sard_php_sqli": "NIST SARD PHP SQL 注入集", "evasion_suite": "逃逸测试集",
    "paired_clean_version": "配对正常版本", "SQL Injection": "SQL 注入", "XSS": "跨站脚本",
    "Command Execution": "命令执行", "SSRF": "服务端请求伪造", "Path Traversal": "路径穿越",
    "Unsafe Deserialization": "不安全反序列化", "Secret Exposure": "敏感信息泄露",
    "WebShell": "网页后门", "Download or Remote Load": "下载或远程加载",
    "Download and Execute": "下载并执行", "Obfuscated Payload": "混淆载荷",
    "Install Hook Execution": "安装钩子执行", "Credential Exfiltration": "凭据外传",
    "Credential Collection": "凭据收集", "Persistence": "持久化",
    "Insecure Direct Object Reference": "不安全的直接对象引用",
    "Missing Authorization Check": "缺少授权检查",
    "Permissive CORS": "过度宽松的跨域配置",
    "Debug Mode Enabled": "生产环境启用调试模式",
    "TLS Verification Disabled": "关闭TLS证书验证",
    "Missing Security Headers": "缺少安全响应头",
    "Default Credentials": "默认凭据",
    "Unpinned Dependency": "依赖版本未固定",
    "Unverified Artifact": "制品未经验证",
    "Weak Cryptographic Hash": "弱密码学哈希",
    "Insecure Randomness": "不安全随机数",
    "ECB Cipher Mode": "ECB加密模式",
    "Cleartext Sensitive Transport": "敏感数据明文传输",
    "Fail Open Security Decision": "安全判断异常放行",
    "Client Side Only Authorization": "仅在客户端实施授权",
    "Missing Abuse Controls": "缺少滥用防护",
    "Plaintext Password Handling": "明文口令处理",
    "JWT Verification Disabled": "关闭JWT验签",
    "Weak Session Management": "薄弱会话管理",
    "Untrusted Plugin Loading": "加载不可信插件",
    "Unsigned Update": "未签名更新",
    "Sensitive Data Logging": "日志记录敏感数据",
    "Log Injection": "日志注入",
    "Missing Security Audit Trail": "缺少安全审计记录",
    "Empty Exception Handler": "空异常处理",
    "Stack Trace Disclosure": "堆栈信息泄露",
    "Unbounded Resource Consumption": "无边界资源消耗",
    "AI Semantic Risk": "AI语义风险",
    "ai_signal": "AI关注信号",
    "Command and Scripting Interpreter": "命令与脚本解释器",
    "Ingress Tool Transfer": "工具传入",
    "Obfuscated Files or Information": "混淆文件或信息",
    "Credentials from Password Stores": "从密码存储中获取凭据",
    "Exfiltration Over C2 Channel": "通过命令与控制通道外传",
    "Event Triggered Execution": "事件触发执行",
    "IOC 线索": "IOC 线索", "高熵字符串": "高熵字符串", "JavaScript 静态去混淆": "JavaScript 静态去混淆",
    "PE 高熵节区": "PE 高熵节区", "PE 可疑导入": "PE 可疑导入", "PE 可疑字符串": "PE 可疑字符串",
    "context": "上下文线索", "indicator_of_compromise": "IOC 指标", "unknown_or_non_pe": "非 PE 或未知二进制",
    "not_applicable": "不适用于该文件", "bounded_read_only": "有界只读解析",
    "binary input bypasses source model": "二进制文件不使用源码模型",
    "no language has both task classes in every split": "没有任何语言在各数据分区中同时包含正常与恶意样本",
    "a deployment-language split is missing one class": "上线语言的数据分区缺少正常或恶意样本",
    "no language route passed Precision/FPR/FNR release gate": "没有语言路由通过精确率、误报率和漏报率发布门禁",
    "no vulnerability language route passed the configured gate": "没有漏洞语言路由通过当前发布门禁",
    "pathtraver": "路径穿越", "hash": "哈希算法", "trustbound": "信任边界",
    "crypto": "弱加密", "cmdi": "命令注入", "sqli": "SQL 注入", "weakrand": "弱随机数",
    "xss": "跨站脚本", "ldapi": "LDAP 注入", "xpathi": "XPath 注入", "securecookie": "Cookie 安全属性",
}


def display_zh(value: object) -> str:
    if value is None or value == "":
        return "暂无"
    text = str(value)
    return DISPLAY_ZH.get(text, DISPLAY_ZH.get(text.lower(), text))


def create_app() -> Flask:
    root_dir = Path(__file__).resolve().parents[1]
    frontend_dir = root_dir / "frontend"
    app = Flask(
        __name__,
        static_folder=str(frontend_dir / "static"),
        template_folder=str(frontend_dir / "templates"),
    )
    secret_key = os.environ.get("XIEZHI_SECRET_KEY")
    if not secret_key and os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("XIEZHI_SECRET_KEY must be set in production")
    app.config["SECRET_KEY"] = secret_key or os.urandom(32)
    request_limit = MAX_ARCHIVE_SIZE + 16 * 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("XIEZHI_MAX_UPLOAD_BYTES", str(request_limit)))
    app.jinja_env.filters["zh"] = display_zh
    init_database()
    reconcile_interrupted_scan_jobs(scan_jobs.active_ids())

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "main.login"
    login_manager.login_message = None

    @login_manager.user_loader
    def load_user(user_id: str):
        return get_user_by_id(int(user_id))

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error: RequestEntityTooLarge):
        if request.path.startswith("/attack/project"):
            flash("ZIP 项目包不能超过 1 GB。")
            return redirect(url_for("attack.project_scan"), code=303)
        return "Upload request is too large.", 413

    app.register_blueprint(main_bp)
    app.register_blueprint(attack_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("XIEZHI_PORT", "3000"))
    print("\n" + "=" * 60)
    print("  Xiezhi CodeGuard: AI Code Risk Detection Workbench")
    print("=" * 60)
    print(f"  URL: http://127.0.0.1:{port}")
    print("  Login user: admin")
    print("  Password: set XIEZHI_ADMIN_PASSWORD, or use the local development default")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)

