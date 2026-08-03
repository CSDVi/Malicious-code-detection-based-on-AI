"""Rule library for source-code supply-chain risk detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from attack_detection.source_masking import mask_non_executable_text


@dataclass(frozen=True)
class DetectionRule:
    rule_id: str
    risk_type: str
    category: str
    severity: int
    pattern: str
    description: str
    repair_advice: str
    cwe: str
    languages: tuple[str, ...] = (
        "php", "python", "javascript", "java", "go", "bash", "powershell",
        "batch", "config", "unknown",
    )


RULES: tuple[DetectionRule, ...] = (
    DetectionRule("SQL-001", "vulnerable", "SQL Injection", 8, r"(?i)(select|union|insert|update|delete).+(from|where).*(\+|\$|request|args|get|post|params)", "发现疑似将用户输入直接拼接进 SQL 的行为。", "使用参数化查询或 ORM 参数绑定。", "CWE-89"),
    DetectionRule("SQL-002", "vulnerable", "SQL Injection", 9, r"(?i)(?:\b(?:or|and)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|\bunion\s+select\b|\b(?:select|union|where|and|or)\b.{0,160}\b(?:sleep|benchmark)\s*\()", "发现常见 SQL 注入载荷特征。", "校验输入类型，并为数据库账户配置最小权限。", "CWE-89"),
    DetectionRule("XSS-001", "vulnerable", "XSS", 8, r"(?i)<script|javascript:|onerror\s*=|onload\s*=", "发现脚本标签、事件属性或 javascript: URL。", "按输出上下文编码内容，并启用内容安全策略。", "CWE-79"),
    DetectionRule("XSS-002", "vulnerable", "XSS", 7, r"(?i)(innerHTML|document\.write|dangerouslySetInnerHTML).*(request|location|cookie|params|query|hash)", "不可信数据可能被直接写入 HTML 或 DOM。", "使用 textContent 或可信的模板净化组件。", "CWE-79"),
    DetectionRule("WS-001", "malicious", "WebShell", 10, r"(?i)(eval|assert|preg_replace)\s*\(\s*(\$_POST|\$_GET|\$_REQUEST|request\.)", "请求数据被动态执行，符合网页后门入口特征。", "删除动态执行入口，并审查访问日志。", "CWE-94", ("php", "python", "javascript", "unknown")),
    DetectionRule("WS-002", "malicious", "WebShell", 10, r"(?i)(base64_decode|gzinflate|str_rot13|atob)\s*\(.{0,120}(eval|assert|system|exec|Function)", "解码或解压后的内容被直接执行。", "删除解码后执行逻辑，并人工复核代码历史。", "CWE-94", ("php", "javascript", "unknown")),
    DetectionRule("CMD-001", "vulnerable", "Command Execution", 10, r"(?i)(os\.system|subprocess\.Popen|subprocess\.call|Runtime\.getRuntime\(\)\.exec|child_process\.exec|shell_exec|passthru|system)\s*\(", "代码调用了系统命令执行接口。", "优先使用安全 API；确需执行时使用参数数组并禁用 Shell。", "CWE-78"),
    DetectionRule("CMD-002", "vulnerable", "Command Execution", 8, r"(?i)(cmd|command|shell)\s*=.*(request|args|get|post|params|\$_GET|\$_POST|req\.)", "命令参数可能直接来自用户输入。", "使用严格白名单，并拒绝分隔符、管道符和重定向符。", "CWE-78"),
    DetectionRule(
        "PYEXEC-001",
        "vulnerable",
        "Command Execution",
        10,
        r"(?is)\b(?:subprocess\.)?(?:Popen|run|call|check_call|check_output)\s*\([^)]*\bshell\s*=\s*True\b",
        "Python 子进程接口启用了 Shell，命令字符串可能被解释为额外指令。",
        "关闭 shell=True，并使用参数数组和严格白名单传入命令参数。",
        "CWE-78",
        ("python", "unknown"),
    ),
    DetectionRule(
        "PYEXEC-002",
        "vulnerable",
        "Command Execution",
        9,
        r"(?i)\b(?:eval|exec)\s*\(\s*(?:input\s*\(|request\.|sys\.argv|user_?(?:input|data)|payload|cmd|code)\b",
        "Python 动态执行接口可能直接处理外部输入或未经验证的代码。",
        "移除 eval/exec；如确需解析表达式，请使用受限解析器并校验允许的语法。",
        "CWE-95",
        ("python", "unknown"),
    ),
    DetectionRule("SSRF-001", "vulnerable", "SSRF", 8, r"(?i)(requests\.get|urllib\.request\.urlopen|fetch|axios\.get|curl_exec|http\.Get)\s*\(.{0,120}(url|request|args|params|\$_GET|\$_POST)", "服务端请求地址可能受外部输入控制。", "使用 URL 白名单，并阻断内网与云元数据地址。", "CWE-918"),
    DetectionRule("SSRF-002", "vulnerable", "SSRF", 9, r"(?i)(?:(?:requests?\.(?:get|post)|urllib\.request\.urlopen|fetch|axios\.(?:get|post)|curl_exec|http\.(?:Get|Post)|Invoke-WebRequest).{0,160}(?:169\.254\.169\.254|metadata\.google\.internal|localhost|127\.0\.0\.1)|(?:169\.254\.169\.254|metadata\.google\.internal|localhost|127\.0\.0\.1).{0,160}(?:requests?\.(?:get|post)|urllib\.request\.urlopen|fetch|axios\.(?:get|post)|curl_exec|http\.(?:Get|Post)|Invoke-WebRequest))", "出站请求直接访问云元数据地址或回环地址。", "在出站请求中阻断元数据网段和本地地址。", "CWE-918"),
    DetectionRule("PATH-001", "vulnerable", "Path Traversal", 8, r"(?i)(\.\./|\.\.\\|send_file|open\s*\(|readFileSync|FileInputStream).{0,120}(request|args|params|get|post|\$_GET|\$_POST)", "文件路径可能受用户输入控制。", "在固定根目录下解析路径，并验证最终路径仍位于根目录内。", "CWE-22"),
    DetectionRule("DESER-001", "vulnerable", "Unsafe Deserialization", 9, r"(?i)(pickle\.loads|yaml\.load\s*\(|ObjectInputStream|unserialize\s*\(|readObject\s*\()", "不安全反序列化可能实例化攻击者控制的对象。", "改用安全数据格式，或强制校验签名与类型白名单。", "CWE-502"),
    DetectionRule("SECRET-001", "vulnerable", "Secret Exposure", 7, r"(?i)(api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}", "发现疑似硬编码密钥或令牌。", "将密钥迁移到环境变量或密钥管理服务，并轮换已暴露凭据。", "CWE-798"),
    DetectionRule("DL-001", "suspicious", "Download or Remote Load", 8, r"(?i)(curl|wget|Invoke-WebRequest|urllib\.request|requests\.get|fetch|axios\.get)\s*\(?[\"']https?://", "代码从远程地址下载或加载内容，单独不足以证明恶意意图。", "限制可信域名，验证哈希或签名，并禁止下载后立即执行。", "CWE-494"),
    DetectionRule("DL-002", "malicious", "Download and Execute", 9, r"(?i)(?:chmod\s+\+x\b|Start-Process\b|ProcessBuilder\s*\(|subprocess(?:\.\w+)?|(?<![\w.])exec\s*(?:\(|\s)).{0,120}(?:https?://|\bdownload\w*\b|\btmp\b|\bcurl\b|\bwget\b)", "发现下载后执行或修改执行权限的调用链。", "分离下载与执行流程，强制签名验证并在沙箱中运行。", "CWE-494"),
    DetectionRule("OBF-001", "suspicious", "Obfuscated Payload", 6, r"(?i)(chr\s*\(|String\.fromCharCode|atob\s*\(|btoa\s*\(|base64_decode|fromCharCode)", "代码主动执行字符拼接或 Base64 转换，可能用于隐藏载荷；单独不足以证明恶意意图。", "发布前解码并人工审查载荷内容。", "CWE-506"),
    DetectionRule("SUPPLY-001", "malicious", "Install Hook Execution", 9, r'(?i)["\']?(preinstall|postinstall|prepare)["\']?\s*:\s*["\'][^"\']*(curl|wget|powershell|child_process|node\s+-e|python\s+-c)', "软件包安装钩子启动了下载器或解释器。", "删除安装阶段的执行逻辑，并审查发布版本差异。", "CWE-506", ("javascript", "config", "unknown")),
    DetectionRule("EXFIL-001", "malicious", "Credential Exfiltration", 10, r"(?i)(discord(app)?\.com/api/webhooks|api\.telegram\.org|webhook).{0,180}(token|secret|password|cookie|process\.env|\.ssh|\.npmrc|pypirc)", "代码疑似将凭据或本地密钥发送到 Webhook 地址。", "删除外传路径，轮换已暴露凭据，并审计软件包使用方。", "CWE-522"),
    DetectionRule("EXFIL-002", "malicious", "Credential Collection", 9, r"(?i)(process\.env|os\.environ|\.npmrc|pypirc|id_rsa|login data|local state).{0,160}(fetch|axios|requests|urllib|http|socket|webhook)", "本地凭据收集行为与出站网络操作同时出现。", "删除凭据收集与外传逻辑，并轮换受影响密钥。", "CWE-522"),
    DetectionRule("PERSIST-001", "malicious", "Persistence", 9, r"(?i)(crontab|schtasks|startup|launchagents|systemd).{0,160}(write|copy|exec|spawn|command|payload)", "代码尝试创建自动启动的持久化机制。", "删除持久化修改，并检查受影响主机中的未授权启动项。", "CWE-506"),
    DetectionRule(
        "AUTHZ-001",
        "suspicious",
        "Insecure Direct Object Reference",
        6,
        r"(?i)\b(?:findById|getById|findOne|find|load|fetch)\s*\(\s*(?:req|request)\.(?:params|query|body)\b",
        "对象查询直接使用了客户端提交的标识，当前文件中需要继续确认是否执行了对象级授权。",
        "查询对象时同时校验当前用户、租户或组织范围，不能只依赖对象编号。",
        "CWE-639",
        ("javascript", "typescript", "unknown"),
    ),
    DetectionRule(
        "MISCONFIG-001",
        "vulnerable",
        "Debug Mode Enabled",
        6,
        r"(?i)\b(?:debug\s*=\s*true|app\.run\s*\([^)]*debug\s*=\s*true|useDeveloperExceptionPage\s*\()",
        "代码显式开启了调试模式或开发异常页面，生产环境可能泄露内部信息。",
        "将调试开关放入分环境配置，并在生产启动时强制校验为关闭。",
        "CWE-489",
    ),
    DetectionRule(
        "MISCONFIG-002",
        "vulnerable",
        "TLS Verification Disabled",
        8,
        r"(?i)(?:verify\s*=\s*false|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true|CURLOPT_SSL_VERIFYPEER\s*,\s*(?:false|0))",
        "网络客户端显式关闭了证书或主机身份验证。",
        "恢复证书链与主机名验证，并修复信任库或证书配置问题。",
        "CWE-295",
    ),
    DetectionRule(
        "MISCONFIG-003",
        "vulnerable",
        "Permissive CORS",
        7,
        r"(?i)(?:allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\].*allow_credentials\s*=\s*true|origin\s*:\s*['\"]\*['\"].*credentials\s*:\s*true|Access-Control-Allow-Origin['\"]?\s*[:,]\s*['\"]\*['\"].{0,100}Access-Control-Allow-Credentials)",
        "跨域配置同时允许任意来源和凭据，可能使受保护响应被非预期站点读取。",
        "将来源限制为经过审核的精确列表，并禁止星号来源与凭据组合。",
        "CWE-942",
    ),
    DetectionRule(
        "CRYPTO-001",
        "vulnerable",
        "Weak Cryptographic Hash",
        7,
        r"(?i)(?:md5|sha1)\s*\([^)]*(?:password|passwd|secret|token)|(?:password|passwd|secret|token).{0,80}(?:md5|sha1)\s*\(",
        "安全敏感数据使用了不适合当前用途的弱哈希算法。",
        "密码改用专用口令哈希算法；签名或完整性校验改用当前认可的算法或 HMAC。",
        "CWE-327",
    ),
    DetectionRule(
        "CRYPTO-002",
        "vulnerable",
        "ECB Cipher Mode",
        7,
        r"(?i)(?:AES|DES|Cipher).{0,80}(?:/ECB\b|MODE_ECB\b|ECBMode\b)",
        "加密配置使用 ECB 模式，重复明文结构可能在密文中保留。",
        "改用成熟库提供的认证加密模式，并为每次加密生成符合要求的唯一 nonce。",
        "CWE-327",
    ),
    DetectionRule(
        "CRYPTO-003",
        "vulnerable",
        "Insecure Randomness",
        7,
        r"(?i)(?:token|session|nonce|reset|otp|api[_-]?key)\w*\s*[:=].{0,80}(?:Math\.random|random\.random|rand\s*\(|srand\s*\()",
        "安全令牌或验证码由可预测的普通伪随机数生成。",
        "使用操作系统或语言平台提供的密码学安全随机源生成安全令牌。",
        "CWE-338",
    ),
    DetectionRule(
        "DESIGN-001",
        "vulnerable",
        "Client Side Only Authorization",
        7,
        r"(?i)(?:localStorage|sessionStorage)\.(?:getItem|\w+).{0,100}(?:role|isAdmin|permission|authorized)",
        "前端可修改存储中的角色或权限值被用于安全判断，需要确认服务端是否重复授权。",
        "关键权限必须由服务端根据已认证主体和可信数据重新计算。",
        "CWE-602",
        ("javascript", "typescript", "unknown"),
    ),
    DetectionRule(
        "DESIGN-002",
        "suspicious",
        "Missing Abuse Controls",
        5,
        r"(?i)(?:while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\)).{0,120}(?:request|fetch|axios|http|socket|query|execute)",
        "发现可能无边界重复执行高成本或外部操作的循环。",
        "为循环、重试、并发和外部请求设置次数、时间与资源上限。",
        "CWE-799",
    ),
    DetectionRule(
        "AUTHN-001",
        "vulnerable",
        "JWT Verification Disabled",
        9,
        r"(?i)(?:verify_signature['\"]?\s*[:=]\s*false|algorithms?\s*[:=]\s*\[\s*['\"]none['\"]|jwt\.decode\s*\([^)]*(?:verify\s*=\s*false|options\s*=\s*\{[^}]*verify_signature[^}]*false))",
        "JWT 解码显式关闭签名验证或接受 none 算法。",
        "固定允许的签名算法，并验证签名、签发者、受众和有效期。",
        "CWE-347",
    ),
    DetectionRule(
        "AUTHN-002",
        "vulnerable",
        "Plaintext Password Handling",
        8,
        r"(?i)(?:password|passwd|pwd)\s*={2,3}\s*['\"][^'\"]+['\"]|['\"][^'\"]+['\"]\s*={2,3}\s*(?:password|passwd|pwd)",
        "密码与硬编码明文直接比较，可能形成固定凭据或明文认证逻辑。",
        "使用受审查的口令哈希验证接口，并从安全配置中移除固定口令。",
        "CWE-256",
    ),
    DetectionRule(
        "LOG-001",
        "vulnerable",
        "Sensitive Data Logging",
        7,
        r"(?i)(?:log(?:ger)?\.(?:debug|info|warn|error|critical)|console\.log|print)\s*\([^)]*(?:password|passwd|access[_-]?token|refresh[_-]?token|private[_-]?key|session[_-]?id)",
        "日志调用可能记录密码、令牌、私钥或会话标识。",
        "移除敏感字段，或在集中日志封装中执行不可逆标记化和受控脱敏。",
        "CWE-532",
    ),
    DetectionRule(
        "LOG-002",
        "vulnerable",
        "Log Injection",
        6,
        r"(?i)(?:log(?:ger)?\.(?:debug|info|warn|error|critical)|console\.log|print)\s*\(.{0,120}(?:\+\s*(?:req|request)\.(?:params|query|body|headers)|\+\s*request\.(?:args|form|values)|\$_(?:GET|POST|REQUEST))",
        "不可信请求数据被直接拼接到日志文本中，换行或控制字符可能伪造日志事件。",
        "使用结构化参数化日志，并对换行、回车和格式控制字符编码或删除。",
        "CWE-117",
    ),
    DetectionRule(
        "ERROR-001",
        "vulnerable",
        "Stack Trace Disclosure",
        7,
        r"(?i)(?:return|send|json|write|render).{0,80}(?:traceback\.format_exc|exception\.stackTrace|err(?:or)?\.stack|getStackTrace\s*\()",
        "异常堆栈可能被直接写入外部响应。",
        "外部只返回通用错误和关联编号，详细堆栈写入受保护且脱敏的内部日志。",
        "CWE-209",
    ),
    DetectionRule(
        "ERROR-002",
        "suspicious",
        "Unbounded Resource Consumption",
        6,
        r"(?i)(?:requests?|axios|fetch|http\.(?:get|post)|Invoke-WebRequest).{0,160}(?:timeout\s*[:=]\s*(?:none|0)|maxRetries\s*[:=]\s*-1)",
        "外部请求显式取消超时或使用无边界重试，可能长期占用资源。",
        "设置连接、读取和总时限，并限制重试次数与并发量。",
        "CWE-770",
    ),
)
_COMPILED_RULES = tuple(
    (rule, re.compile(rule.pattern))
    for rule in RULES
)


def detect_by_rules(content: str, language: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    executable_content = mask_non_executable_text(content, language)
    lines = executable_content.splitlines()
    for rule, regex in _COMPILED_RULES:
        if language not in rule.languages and "unknown" not in rule.languages:
            continue
        # A failed whole-source search proves that no individual line can
        # match, avoiding one regex call per line for the common absent-rule
        # case.  Positive rules retain the established line-by-line matching
        # and therefore the same findings and line numbers.
        if not regex.search(executable_content):
            continue
        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append({
                    "source": "rule_engine",
                    "rule_id": rule.rule_id,
                    "risk_type": rule.risk_type,
                    "category": rule.category,
                    "severity": rule.severity,
                    "description": rule.description,
                    "repair_advice": rule.repair_advice,
                    "cwe": rule.cwe,
                    "line": line_no,
                    "snippet": line.strip()[:180],
                })
    if not any(item.get("category") == "SQL Injection" for item in matches):
        dataflow = _sql_injection_dataflow(executable_content, language)
        if dataflow:
            matches.append(dataflow)
    if language == "java" and not any(item.get("category") == "Path Traversal" for item in matches):
        dataflow = _java_path_traversal_dataflow(executable_content)
        if dataflow:
            matches.append(dataflow)
    matches.extend(_exception_handling_findings(executable_content, language))
    _restore_original_snippets(matches, content)
    return matches


def _restore_original_snippets(
    findings: list[dict[str, object]],
    content: str,
) -> None:
    """Restore report snippets after detection on coordinate-preserving masks."""

    original_lines = content.splitlines()
    for finding in findings:
        _restore_snippet(finding, original_lines)
        trace_steps = finding.get("trace_steps")
        if isinstance(trace_steps, list):
            for step in trace_steps:
                if isinstance(step, dict):
                    _restore_snippet(step, original_lines)


def _restore_snippet(item: dict[str, object], original_lines: list[str]) -> None:
    try:
        line = int(item.get("line") or 0)
    except (TypeError, ValueError):
        return
    if 1 <= line <= len(original_lines):
        item["snippet"] = original_lines[line - 1].strip()[:180]


_ASSIGNMENT = re.compile(
    r"^\s*(?:(?:const|let|var|final|String|Object|def|char\s*\*|auto)\s+)*"
    r"(?P<name>\$?[A-Za-z_][\w$]*)\s*(?::=|=)\s*(?P<value>.+?);?\s*$",
)
_SQL_TEXT = re.compile(r"(?i)\b(?:select|insert|update|delete|replace|merge)\b")
_LANGUAGE_SQL_SOURCES = {
    "python": re.compile(r"(?i)\brequest\.(?:args|form|values|json)\b|\brequest\.get_json\s*\(|\binput\s*\("),
    "java": re.compile(r"(?i)\b(?:getParameter|getHeader|getCookies|getQueryString)\s*\("),
    "javascript": re.compile(r"(?i)\b(?:req|request)\.(?:query|body|params|headers)\b|\blocation\.(?:search|hash)\b"),
    "typescript": re.compile(r"(?i)\b(?:req|request)\.(?:query|body|params|headers)\b|\blocation\.(?:search|hash)\b"),
    "php": re.compile(r"(?i)\$_(?:GET|POST|REQUEST|COOKIE|SERVER)\b"),
    "go": re.compile(r"(?i)\b(?:FormValue|PostFormValue)\s*\(|\.URL\.Query\(\)\.Get\s*\("),
}
_LANGUAGE_SQL_SINKS = {
    "python": re.compile(r"(?i)\b(?:execute|executemany)\s*\("),
    "java": re.compile(r"(?i)\b(?:execute|executeQuery|executeUpdate|createQuery|createNativeQuery)\s*\("),
    "javascript": re.compile(r"(?i)\b(?:query|execute|raw)\s*\("),
    "typescript": re.compile(r"(?i)\b(?:query|execute|raw)\s*\("),
    "php": re.compile(r"(?i)\b(?:mysqli_query|mysql_query|query|exec)\s*\("),
    "go": re.compile(r"(?i)\b(?:db|tx|stmt|conn|database)\s*\.\s*(?:Query|QueryRow|Exec)(?:Context)?\s*\("),
}


def _sql_injection_dataflow(content: str, language: str) -> dict[str, object] | None:
    """Locate a conservative request-input -> dynamic SQL -> database sink flow."""

    lines = content.splitlines()
    source_pattern = _LANGUAGE_SQL_SOURCES.get(language)
    sink_pattern = _LANGUAGE_SQL_SINKS.get(language)
    if source_pattern is None or sink_pattern is None:
        return None

    assignments: list[tuple[int, str, str, str]] = []
    for line_number, line in enumerate(lines, 1):
        match = _ASSIGNMENT.search(line)
        if match:
            assignments.append((line_number, match.group("name"), match.group("value"), line.strip()))

    tainted: set[str] = set()
    query_values: set[str] = set()
    for _ in range(4):
        changed = False
        for _line_number, variable, expression, _snippet in assignments:
            references_tainted = any(_references_name(expression, name) for name in tainted)
            if source_pattern.search(expression) or references_tainted:
                if variable not in tainted:
                    tainted.add(variable)
                    changed = True
            if _SQL_TEXT.search(expression) and (
                source_pattern.search(expression)
                or references_tainted
                or _looks_dynamically_built(expression)
            ):
                if variable not in query_values:
                    query_values.add(variable)
                    changed = True
            elif any(_references_name(expression, name) for name in query_values):
                if variable not in query_values:
                    query_values.add(variable)
                    changed = True
        if not changed:
            break

    for line_number, line in enumerate(lines, 1):
        sink_match = sink_pattern.search(line)
        if not sink_match:
            continue
        direct_dynamic_sql = _SQL_TEXT.search(line) and (
            source_pattern.search(line) or _looks_dynamically_built(line)
        )
        referenced_query = any(_references_name(line, name) for name in query_values)
        first_argument = line[sink_match.end():].split(",", 1)[0].strip()
        direct_tainted_query = any(
            re.fullmatch(rf"{re.escape(name)}\s*\)?\s*;?", first_argument)
            for name in tainted
        )
        if not (direct_dynamic_sql or referenced_query or direct_tainted_query):
            continue
        trace_steps = _sql_trace_steps(
            assignments,
            source_pattern,
            tainted,
            query_values,
            line_number,
            line,
        )
        return {
            "source": "rule_engine",
            "rule_id": "SQL-003",
            "risk_type": "vulnerable",
            "category": "SQL Injection",
            "severity": 9,
            "description": "外部输入经过变量传播参与动态 SQL 构造，并到达数据库执行接口。",
            "repair_advice": "使用参数化查询，并对无法绑定的动态标识符采用固定白名单映射。",
            "cwe": "CWE-89",
            "line": line_number,
            "snippet": line.strip()[:180],
            "trace_steps": trace_steps,
        }
    return None


def _references_name(expression: str, name: str) -> bool:
    return bool(re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", expression))


def _looks_dynamically_built(expression: str) -> bool:
    return bool(
        "+" in expression
        or re.search(r"\$\{[^}]+\}", expression)
        or re.search(r"(?i)\b(?:format|sprintf|Sprintf)\s*\(", expression)
        or re.search(r"(?<!%)%(?!%)", expression)
    )


def _sql_trace_steps(
    assignments: list[tuple[int, str, str, str]],
    source_pattern: re.Pattern[str],
    tainted: set[str],
    query_values: set[str],
    sink_line: int,
    sink_snippet: str,
) -> list[dict[str, object]]:
    """Summarize the conservative static SQL flow without claiming runtime proof."""

    candidates = []
    for line_number, variable, expression, snippet in assignments:
        if line_number > sink_line:
            continue
        if source_pattern.search(expression):
            kind, stage = "source", "外部输入进入变量"
        elif variable in query_values:
            kind, stage = "propagation", "动态 SQL 构造或传播"
        elif variable in tainted:
            kind, stage = "propagation", "不可信数据传播"
        else:
            continue
        candidates.append({
            "kind": kind,
            "stage": stage,
            "line": line_number,
            "snippet": snippet[:180],
        })
    sources = [item for item in candidates if item["kind"] == "source"]
    propagation = [item for item in candidates if item["kind"] == "propagation"]
    selected = sources[:1] + propagation[-3:]
    selected.append({
        "kind": "sink",
        "stage": "数据库执行接口",
        "line": sink_line,
        "snippet": sink_snippet.strip()[:180],
    })
    return _deduplicate_trace_steps(selected)


def _deduplicate_trace_steps(steps: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    seen = set()
    for step in sorted(steps, key=lambda item: int(item.get("line") or 0)):
        key = (step.get("line"), step.get("snippet"))
        if key in seen:
            continue
        output.append(step)
        seen.add(key)
    return output


def _java_path_traversal_dataflow(content: str) -> dict[str, object] | None:
    """Locate a simple Java source -> path concatenation -> file sink flow."""
    lines = content.splitlines()
    tainted: set[str] = set()
    path_values: set[str] = set()
    assignment = re.compile(r"\b([A-Za-z_$][\w$]*)\s*=\s*(.+)")
    external_source = re.compile(
        r"\b(?:getParameter|getHeader|getCookies|getValue|URLDecoder\.decode)\s*\(", re.IGNORECASE,
    )
    file_sink = re.compile(r"\b(?:FileInputStream|FileReader|RandomAccessFile)\s*\(|\bnew\s+File\s*\(")

    assignments: list[tuple[int, str, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = assignment.search(line)
        if not match:
            continue
        variable, expression = match.group(1), match.group(2)
        assignments.append((line_number, variable, expression, line.strip()))
        if external_source.search(expression) or any(re.search(rf"\b{re.escape(name)}\b", expression) for name in tainted):
            tainted.add(variable)

    for line in lines:
        match = assignment.search(line)
        if not match:
            continue
        variable, expression = match.group(1), match.group(2)
        if "+" in expression and any(re.search(rf"\b{re.escape(name)}\b", expression) for name in tainted):
            path_values.add(variable)

    if not tainted or not path_values:
        return None
    for line_number, line in enumerate(lines, start=1):
        if file_sink.search(line) and any(re.search(rf"\b{re.escape(name)}\b", line) for name in path_values):
            candidates = []
            for candidate_line, variable, expression, snippet in assignments:
                if candidate_line > line_number:
                    continue
                if external_source.search(expression):
                    kind, stage = "source", "外部输入进入路径变量"
                elif variable in path_values:
                    kind, stage = "propagation", "路径拼接或传播"
                elif variable in tainted:
                    kind, stage = "propagation", "不可信数据传播"
                else:
                    continue
                candidates.append({
                    "kind": kind,
                    "stage": stage,
                    "line": candidate_line,
                    "snippet": snippet[:180],
                })
            sources = [item for item in candidates if item["kind"] == "source"]
            propagation = [item for item in candidates if item["kind"] == "propagation"]
            trace_steps = sources[:1] + propagation[-3:] + [{
                "kind": "sink",
                "stage": "文件读取接口",
                "line": line_number,
                "snippet": line.strip()[:180],
            }]
            return {
                "source": "rule_engine",
                "rule_id": "PATH-002",
                "risk_type": "vulnerable",
                "category": "Path Traversal",
                "severity": 8,
                "description": "外部输入经变量传播参与文件路径拼接，并到达文件读取接口。",
                "repair_advice": "在固定根目录下规范化路径，并验证解析后的路径仍位于该根目录内。",
                "cwe": "CWE-22",
                "line": line_number,
                "snippet": line.strip()[:180],
                "trace_steps": _deduplicate_trace_steps(trace_steps),
            }
    return None


def _exception_handling_findings(
    content: str,
    language: str,
) -> list[dict[str, object]]:
    """Detect a small set of high-signal exception-handling failures."""

    lines = content.splitlines()
    findings: list[dict[str, object]] = []
    python_except = re.compile(r"^\s*except(?:\s+[^:]+)?\s*:\s*(?P<body>.*)$")
    block_catch = re.compile(r"(?i)\bcatch\s*(?:\([^)]*\))?\s*\{\s*(?P<body>.*?)\s*\}")
    fail_open = re.compile(
        r"(?i)\b(?:return\s+true|(?:allow|allowed|authorized|authenticated|valid)\w*\s*[:=]\s*true)\b",
    )

    for index, line in enumerate(lines):
        body = None
        if language == "python":
            match = python_except.search(line)
            if match:
                body = match.group("body").strip()
                if not body:
                    for following in lines[index + 1:index + 4]:
                        stripped = following.strip()
                        if stripped:
                            body = stripped
                            break
        else:
            match = block_catch.search(line)
            if match:
                body = match.group("body").strip()
            elif re.search(r"(?i)\bcatch\s*(?:\([^)]*\))?\s*\{\s*$", line):
                body_lines: list[str] = []
                for following in lines[index + 1:index + 4]:
                    stripped = following.strip()
                    if stripped == "}":
                        break
                    if stripped:
                        body_lines.append(stripped)
                body = " ".join(body_lines)

        if body is None:
            continue
        if not body or re.fullmatch(r"(?:pass|continue|;|//.*|/\*.*\*/)", body):
            findings.append({
                "source": "rule_engine",
                "rule_id": "ERROR-003",
                "risk_type": "vulnerable",
                "category": "Empty Exception Handler",
                "severity": 6,
                "description": "异常处理器为空或仅跳过错误，程序可能带着不完整状态继续执行。",
                "repair_advice": "只捕获能够处理的异常，并明确记录、回滚、拒绝或安全终止当前操作。",
                "cwe": "CWE-390",
                "line": index + 1,
                "snippet": line.strip()[:180],
            })
        elif fail_open.search(body):
            findings.append({
                "source": "rule_engine",
                "rule_id": "DESIGN-003",
                "risk_type": "vulnerable",
                "category": "Fail Open Security Decision",
                "severity": 9,
                "description": "异常路径把安全判断设置为允许，安全依赖故障时可能绕过控制。",
                "repair_advice": "认证、授权和完整性校验异常时默认拒绝，并记录可关联的安全事件。",
                "cwe": "CWE-636",
                "line": index + 1,
                "snippet": line.strip()[:180],
            })
    return findings
