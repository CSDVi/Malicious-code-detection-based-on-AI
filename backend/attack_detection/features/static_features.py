"""Stable structured feature schema for the calibrated XGBoost engines."""

from __future__ import annotations

import ast
import math
import re
import warnings
from collections import Counter

from attack_detection.features.high_confidence_behaviors import (
    go_high_confidence_behavior_count,
    powershell_high_confidence_behavior_count,
    python_high_confidence_behavior_count,
    ruby_high_confidence_behavior_count,
    rust_high_confidence_behavior_count,
)

DANGEROUS_APIS = (
    "eval", "exec", "system", "popen", "subprocess", "child_process", "pickle.loads",
    "yaml.load", "unserialize", "requests.get", "urllib.request", "fetch", "curl", "wget",
)
LANGUAGE_BUCKETS = (
    "python", "javascript", "typescript", "java", "kotlin", "php", "bash", "config", "go",
    "powershell", "batch", "c", "cpp", "csharp", "ruby", "rust", "scala", "lua",
    "perl", "html", "sql",
)
NATIVE_BEHAVIOR_GROUP_PATTERNS = (
    r"\b(?:createprocess|winexec|shellexecute|system|popen|execve|/bin/sh|cmd\.exe)\b",
    r"\b(?:socket|connect|bind|listen|accept|sendto|recvfrom|wsastartup|wininet|"
    r"curl|wget|invoke-webrequest|https?://)\b",
    r"\b(?:writeprocessmemory|createremotethread|virtualallocex|ptrace|process_vm_writev)\b",
    r"\b(?:regsetvalue|createservice|schtasks|currentversion\\run|crontab|persistence)\b",
    r"\b(?:getasynckeystate|setwindowshookex|keylog|credential|password|passwd|id_rsa)\b",
    r"\b(?:isdebuggerpresent|ntqueryinformationprocess|virtualbox|vmware|sandbox|wireshark)\b",
    r"\b(?:ransom|encryptfile|deletefile|unlink|wipe|shred|master boot record)\b",
    r"\b(?:botnet|backdoor|rootkit|shellcode|command.?and.?control|reverse.?shell)\b",
    r"\b(?:irc\.|privmsg|user-agent|c2_server|cnc_server|beacon(?:ing)?|gate\.php)\b",
)


def _native_executable_text(content: str) -> str:
    """Remove C/C++ comments so provenance prose is not model evidence."""

    executable = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    return re.sub(r"(?m)(?<!:)//[^\r\n]*", " ", executable).lower()


def extract_static_features(
    content: str,
    language: str,
    *,
    include_rules: bool = True,
) -> dict[str, float]:
    lowered = content.lower()
    lines = content.splitlines()
    line_lengths = [len(line) for line in lines] or [0]
    quoted_strings = re.findall(r"(['\"])(.*?)\1", content, re.DOTALL)
    string_values = [value for _, value in quoted_strings]
    # Maximum entropy is unchanged when duplicate literals are removed.
    # Web projects often repeat the same short HTML/PHP literals thousands of
    # times, so deduplicating only this max calculation avoids redundant
    # Counter/log work without changing any trained feature value.
    unique_string_values = set(string_values)
    ast_metrics = _python_ast_metrics(content) if language == "python" else {}
    rule_metrics = _rule_metrics(content, language) if include_rules else {}
    line_count = max(1, len(lines))
    byte_count = max(1, len(content.encode("utf-8", errors="ignore")))
    from attack_detection.features.behavior_tokens import behavior_tokens

    semantic_tokens = set(behavior_tokens(content, language))

    # These features are deliberately language-agnostic proxies for the
    # source-to-sink and install-time behaviors that a tree model can learn
    # without requiring a full parser for every supported language.
    source_count = len(re.findall(
        r"(?:\$_(?:get|post|request|cookie|server|files)|"
        r"\b(?:request|req|argv|stdin|input|query|params?|payload|body|form|"
        r"process\.env|os\.environ|getenv)\b)",
        lowered,
    ))
    sql_sink_count = len(re.findall(
        r"\b(?:select|insert|update|delete|union)\b.{0,160}"
        r"\b(?:query|execute|exec|cursor|sqlite|mysql|pgsql)\b|"
        r"\b(?:query|execute|exec)\b.{0,160}"
        r"\b(?:select|insert|update|delete)\b",
        lowered,
        re.DOTALL,
    ))
    command_sink_count = len(re.findall(
        r"\b(?:system|popen|shell_exec|passthru|exec|spawn|subprocess|"
        r"child_process|runtime\.getruntime|processbuilder|winexec|"
        r"createprocess(?:a|w)?|shellexecute(?:a|w)?|execv(?:e|p)?|execlp?)\b",
        lowered,
    ))
    file_sink_count = len(re.findall(
        r"\b(?:open|writefile|file_put_contents|fopen|unlink|rename|chmod|"
        r"copy|move_uploaded_file)\b",
        lowered,
    ))
    network_sink_count = len(re.findall(
        r"\b(?:requests?\.(?:get|post|request)|urllib|httpx?|fetch|curl|"
        r"socket|connect|bind|listen|accept|send|recv|webclient|download|"
        r"urlopen|wininet|internetopen|internetconnect|httpopenrequest|"
        r"urldownloadtofile|wsastartup)\b",
        lowered,
    ))
    decode_count = len(re.findall(
        r"\b(?:base64|b64decode|atob|fromcharcode|unescape|urldecode|"
        r"hex2bin|bytes\.fromhex|decode)\b",
        lowered,
    ))
    sanitizer_count = len(re.findall(
        r"\b(?:htmlspecialchars|escape|sanitize|prepared?statement|"
        r"parameteri[sz]e|quote|allowlist|whitelist|validate|strip_tags)\b",
        lowered,
    ))
    import_hook_count = len(re.findall(
        r"\b(?:preinstall|postinstall|prepare|setup\.py|setup_requires|"
        r"cmdclass|build_ext|install_requires|package\.json)\b",
        lowered,
    ))
    native_process_api_count = len(re.findall(
        r"\b(?:createprocess(?:a|w)?|winexec|shellexecute(?:a|w)?|"
        r"execv(?:e|p)?|execlp?|fork|daemon|ptrace|kill)\s*\(",
        lowered,
    ))
    native_network_api_count = len(re.findall(
        r"\b(?:socket|connect|bind|listen|accept|send|sendto|recv|recvfrom|"
        r"wsastartup|internetopen|internetconnect|httpopenrequest|"
        r"urldownloadtofile)\s*(?:a|w)?\s*\(",
        lowered,
    ))
    process_injection_api_count = len(re.findall(
        r"\b(?:openprocess|virtualallocex|writeprocessmemory|"
        r"createremotethread|ntwritevirtualmemory|ntcreatethreadex|"
        r"queueuserapc|setthreadcontext|createtoolhelp32snapshot|"
        r"process_vm_writev|ptrace)\b",
        lowered,
    ))
    persistence_api_count = len(re.findall(
        r"\b(?:regsetvalue|regcreatekey|createservice|startservice|"
        r"schtasks|currentversion\\\\run|startup|crontab|systemd|"
        r"launchagents?|autorun|persistence)\b",
        lowered,
    ))
    credential_access_api_count = len(re.findall(
        r"\b(?:cryptunprotectdata|lsass|samlib|sam hive|keylog|"
        r"getasynckeystate|setwindowshookex|credential|password|passwd|"
        r"id_rsa|wallet\.dat|login data)\b",
        lowered,
    ))
    anti_analysis_api_count = len(re.findall(
        r"\b(?:isdebuggerpresent|checkremotedebuggerpresent|"
        r"ntqueryinformationprocess|outputdebugstring|rdtsc|"
        r"virtualbox|vmware|sandbox|wireshark|procmon|ollydbg|x64dbg)\b",
        lowered,
    ))
    destructive_api_count = len(re.findall(
        r"\b(?:deletefile|removefile|unlink|rmdir|formatvolume|"
        r"deviceiocontrol|master boot record|encrypt(?:file|directory)|"
        r"ransom|wipe|shred)\b",
        lowered,
    ))
    dynamic_winapi_resolution_count = len(re.findall(
        r"\b(?:newlazydll|newproc|getprocaddress|loadlibrary(?:a|w)?|"
        r"getmodulehandle(?:a|w)?)\b",
        lowered,
    ))
    privilege_escalation_count = len(re.findall(
        r"\b(?:bypassuac|uac.?bypass|elevat(?:e|ed|ion)|fodhelper|"
        r"computerdefaults|sdclt|token.?elevation|sedebugprivilege)\b",
        lowered,
    ))
    memory_execution_api_count = len(re.findall(
        r"\b(?:virtualprotect|virtualalloc(?:ex)?|page_execute(?:_readwrite)?|"
        r"writeprocessmemory|createremotethread|ntprotectvirtualmemory)\b",
        lowered,
    ))
    wifi_credential_api_count = len(re.findall(
        r"\b(?:wlangetprofile|wlan_profile_get_plaintext_key|wlanprofile|"
        r"wireless.?password|wifi.?password)\b",
        lowered,
    ))
    self_delete_api_count = len(re.findall(
        r"\b(?:filedispositioninfo|filerenameinfo|movefileex(?:a|w)?|"
        r"deletefile(?:a|w)?|self.?delete|self.?erase)\b",
        lowered,
    ))
    system_identity_api_count = len(re.findall(
        r"\b(?:netwkstagetinfo|netwkstausergetinfo|getusername(?:a|w)?|"
        r"whoami|computername|os\.getlogin|userprofile)\b",
        lowered,
    ))
    sensitive_collection_count = len(re.findall(
        r"(?:\bfilepath\.(?:walk|walkdir)\b|\bwalkdir\b|"
        r"\bcapture(?:screen)?\b|\bscreenshot\b|"
        r"\.(?:pdf|docx?|xlsx?|pptx?)\b|"
        r"\b(?:login data|local state|browser.?cookie|discord.?token|"
        r"wallet|wlangetprofile|outlook|mailbox)\b)",
        lowered,
    ))
    exfiltration_channel_count = len(re.findall(
        r"(?:\b(?:sftp|scp|ftp|webhook|multipart|smtp)\b|"
        r"\bapi/webhooks\b|\bupload(?:file)?\b|"
        r"\brequests?\.(?:post|webhook)\b)",
        lowered,
    ))
    network_service_structure_count = len(re.findall(
        r"\b(?:listenaddress|listenandserve|httpaddr|tls\.config|x509|"
        r"mqtt|paho|router|gateway|handlerfunc|mflag|stringvar|boolvar|"
        r"intvar|server)\b",
        lowered,
    ))
    extended_command_channel_api_count = len(re.findall(
        r"\b(?:command::new|std::process::command|open3|io\.popen|"
        r"kernel\.exec)\b",
        lowered,
    ))
    extended_network_channel_api_count = len(re.findall(
        r"\b(?:tcpsocket|tcpserver|udpsocket|tcpstream|tcplistener|"
        r"tokio::net|wlangetprofile|wlanopenhandle)\b",
        lowered,
    ))
    rust_offensive_behavior_count = (
        rust_high_confidence_behavior_count(content)
        if language == "rust"
        else 0
    )
    ruby_offensive_behavior_count = (
        ruby_high_confidence_behavior_count(content)
        if language == "ruby"
        else 0
    )
    python_offensive_behavior_count = (
        python_high_confidence_behavior_count(content)
        if language == "python"
        else 0
    )
    go_offensive_behavior_count = (
        go_high_confidence_behavior_count(content)
        if language == "go"
        else 0
    )
    powershell_offensive_behavior_count = (
        powershell_high_confidence_behavior_count(content)
        if language == "powershell"
        else 0
    )
    go_reverse_shell_proxy = (
        language == "go"
        and bool(re.search(r"\bnet\.(?:dial|dialtcp)\s*\(", lowered))
        and bool(re.search(
            r"\bexec\.command\s*\(\s*[`'\"]"
            r"(?:/bin/(?:ba)?sh|cmd(?:\.exe)?|powershell(?:\.exe)?)",
            lowered,
        ))
        and sum(
            bool(re.search(rf"\.\s*{stream}\s*=\s*\w+", lowered))
            for stream in ("stdin", "stdout", "stderr")
        ) >= 2
    )
    go_download_execute_proxy = (
        language == "go"
        and bool(re.search(r"\b(?:http\.get|client\.do)\s*\(", lowered))
        and bool(re.search(r"\b(?:os\.create|os\.openfile|io\.copy)\s*\(", lowered))
        and bool(re.search(r"\bexec\.command\s*\(", lowered))
    )
    go_local_development_tool_proxy = (
        language == "go"
        and bool(re.search(
            r"\bexec\.command\s*\(\s*[`'\"]"
            r"(?:git|go|gofmt|golangci-lint|make|cmake|docker|kubectl)",
            lowered,
        ))
        and not bool(re.search(
            r"\b(?:net\.(?:dial|dialtcp)|http\.get|client\.do)\s*\(",
            lowered,
        ))
    )
    powershell_download_execute_persist_proxy = (
        language == "powershell"
        and bool(re.search(
            r"\b(?:invoke-webrequest|invoke-restmethod|downloadstring|"
            r"downloadfile|net\.webclient)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:invoke-expression|iex|start-process|start-job|"
            r"powershell(?:\.exe)?)\b",
            lowered,
        ))
        and bool(re.search(
            r"(?:currentversion\\\\run|currentversion\\run|new-itemproperty|"
            r"register-scheduledtask|schtasks|startup|wmi.*eventconsumer)",
            lowered,
        ))
    )
    powershell_encoded_remote_loader_proxy = (
        language == "powershell"
        and bool(re.search(
            r"\b(?:downloadstring|downloadfile|invoke-webrequest|"
            r"net\.webclient)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:frombase64string|gzipstream|deflatestream)\b",
            lowered,
        ))
        and bool(re.search(r"\b(?:invoke-expression|iex)\b", lowered))
    )
    java_socket_command_shell_proxy = (
        language == "java"
        and bool(re.search(r"\b(?:socket|serversocket)\s*\(", lowered))
        and bool(re.search(
            r"\b(?:processbuilder|runtime\.getruntime\(\)\.exec)\b",
            lowered,
        ))
        and bool(re.search(
            r"(?:/bin/(?:ba)?sh|cmd(?:\.exe)?|powershell(?:\.exe)?)",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:getinputstream|getoutputstream|transferto)\b",
            lowered,
        ))
    )
    java_download_execute_proxy = (
        language == "java"
        and bool(re.search(
            r"\b(?:url|httpclient|openstream|openconnection)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:files\.(?:copy|write)|fileoutputstream)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:processbuilder|runtime\.getruntime\(\)\.exec)\b",
            lowered,
        ))
    )
    java_local_development_tool_proxy = (
        language == "java"
        and bool(re.search(
            r"\bprocessbuilder\s*\(\s*[\"']"
            r"(?:git|mvn|mvnw|gradle|gradlew|java|javac|docker)",
            lowered,
        ))
        and not bool(re.search(
            r"\b(?:socket|serversocket|url|httpclient|openstream)\b",
            lowered,
        ))
    )
    javascript_remote_eval_loader_proxy = (
        language in {"javascript", "typescript"}
        and bool(re.search(
            r"\b(?:fetch|axios\.(?:get|request)|https?\.get)\s*\(",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:eval|function|vm\.runin(?:new|this)context)\s*\(",
            lowered,
        ))
    )
    javascript_credential_exfiltration_proxy = (
        language in {"javascript", "typescript"}
        and bool(re.search(r"\bprocess\.env\b", lowered))
        and bool(re.search(
            r"\b(?:fetch|axios\.(?:post|request)|https?\.request|webhook)\b",
            lowered,
        ))
    )
    javascript_local_development_tool_proxy = (
        language in {"javascript", "typescript"}
        and bool(re.search(
            r"\b(?:spawn|execfile|execfilesync)\s*\(\s*[\"']"
            r"(?:git|npm|npx|yarn|pnpm|node|docker)",
            lowered,
        ))
        and not bool(re.search(
            r"\b(?:fetch|axios|https?\.get|https?\.request)\b",
            lowered,
        ))
    )
    php_prepared_query_proxy = (
        language == "php"
        and bool(re.search(r"(?:->prepare|mysqli_prepare)\s*\(", lowered))
        and bool(re.search(r"(?:->execute|mysqli_stmt_execute)\s*\(", lowered))
        and not bool(re.search(
            r"\b(?:eval|assert|shell_exec|passthru|system)\s*\(",
            lowered,
        ))
    )
    php_encoded_webshell_proxy = (
        language == "php"
        and bool(re.search(
            r"\$_(?:post|get|request|cookie)\s*\[",
            lowered,
        ))
        and bool(re.search(r"\bbase64_decode\s*\(", lowered))
        and bool(re.search(r"\b(?:eval|assert)\s*\(", lowered))
    )
    php_credential_exfiltration_proxy = (
        language == "php"
        and bool(re.search(
            r"\bfile_get_contents\s*\([^)]{0,160}"
            r"(?:id_rsa|credentials|login data|wallet|passwd|shadow)",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:curl_exec|curl_setopt|http_build_query)\s*\(",
            lowered,
        ))
    )
    bash_build_script_structure_count = (
        sum((
            bool(re.search(
                r"\b(?:mvn|gradle|cmake|make|configure|autogen)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:docker-compose|docker\s+compose|buildx)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:cflags|ldflags|target_arch|host_compiler|"
                r"cross[_-]?compile)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:test_description|test_expect_success|pytest|"
                r"integration\.test|github_actions|ci)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:apt-get|debconf-set-selections|yum|dnf|apk\s+add)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:aclocal|automake|libtoolize|autoreconf)\b",
                lowered,
            )),
            bool(re.search(
                r"\b(?:clean\s+package|build[_-]?dir|dist-build|"
                r"make\s+install)\b",
                lowered,
            )),
        ))
        if language == "bash"
        else 0
    )
    bash_android_cross_compile_proxy = (
        language == "bash"
        and bool(re.search(
            r"\b(?:android|ndk|armv7|armv8|aarch64|x86_64)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:cflags|ldflags|target_arch|host_compiler|"
            r"cross[_-]?compile)\b",
            lowered,
        ))
    )
    bash_local_archive_structure_proxy = (
        language == "bash"
        and bool(re.search(
            r"\b(?:tar\s+[^\r\n]{0,100}(?:-[a-z]*c[a-z]*|-czf|--create)|"
            r"zip\s+(?:-[a-z]+\s+)*[^\r\n]+)",
            lowered,
        ))
        and bool(re.search(
            r"(?:\bmkdir\s+-p\b|\bbackup\b|\barchive\b|\./|"
            r"\$\{?(?:src|source|dest|target|backup|archive))",
            lowered,
        ))
        and not bool(re.search(
            r"\b(?:curl|wget|nc|ncat|netcat|socat|ssh|scp|ftp)\b|"
            r"/dev/(?:tcp|udp)/",
            lowered,
        ))
    )
    native_executable = (
        _native_executable_text(content)
        if language in {"c", "cpp"}
        else ""
    )
    native_behavior_group_count = sum(
        bool(re.search(pattern, native_executable))
        for pattern in NATIVE_BEHAVIOR_GROUP_PATTERNS
    )
    # The practiceset curator uses these code-only behavior groups for every
    # language, while the legacy model exposed them only for C/C++.  Keep the
    # legacy native fields unchanged and add separate multilingual fields so
    # previously saved feature schemas remain semantically compatible.
    file_local_executable = _native_executable_text(content)
    file_local_behavior_group_count = sum(
        bool(re.search(pattern, file_local_executable))
        for pattern in NATIVE_BEHAVIOR_GROUP_PATTERNS
    )
    html_password_input_count = len(re.findall(
        r"<input\b[^>]{0,500}\btype\s*=\s*['\"]?password\b",
        lowered,
    ))
    html_form_count = len(re.findall(r"<form\b", lowered))
    html_external_form_action_count = len(re.findall(
        r"<form\b[^>]{0,1000}\baction\s*=\s*['\"]?\s*(?:https?:)?//",
        lowered,
    ))
    html_hidden_element_count = len(re.findall(
        r"(?:\btype\s*=\s*['\"]?hidden\b|display\s*:\s*none|"
        r"visibility\s*:\s*hidden|opacity\s*:\s*0(?:\D|$))",
        lowered,
    ))
    html_login_keyword_count = len(re.findall(
        r"\b(?:sign[\s_-]?in|log[\s_-]?in|verify(?:\s+your)?\s+account|"
        r"confirm(?:\s+your)?\s+(?:identity|account)|password|passcode|"
        r"security\s+check|wallet|banking)\b",
        lowered,
    ))
    html_script_obfuscation_count = len(re.findall(
        r"\b(?:eval|unescape|fromcharcode|atob|document\.write)\s*\(",
        lowered,
    ))
    # Co-occurrence is more useful than raw keyword counts for malicious
    # intent: a single network call is normal, while network+write+execute
    # or decode+execute is a strong behavioral chain.
    behavior_chain_count = sum((
        int(network_sink_count > 0 and file_sink_count > 0),
        int(network_sink_count > 0 and command_sink_count > 0),
        int(decode_count > 0 and command_sink_count > 0),
        int(source_count > 0 and (command_sink_count + sql_sink_count + file_sink_count) > 0),
        int(import_hook_count > 0 and (network_sink_count + command_sink_count) > 0),
        int(process_injection_api_count > 0),
        int(persistence_api_count > 0 and command_sink_count > 0),
        int(credential_access_api_count > 0 and network_sink_count > 0),
        int(anti_analysis_api_count > 0 and (
            command_sink_count + native_network_api_count + process_injection_api_count
        ) > 0),
    ))
    advanced_malicious_behavior_count = sum((
        int(
            sensitive_collection_count > 0
            and exfiltration_channel_count > 0
        ),
        int(process_injection_api_count > 0),
        int(persistence_api_count > 0 and command_sink_count > 0),
        int(
            anti_analysis_api_count > 0
            and (
                command_sink_count
                + native_network_api_count
                + process_injection_api_count
            ) > 0
        ),
        int(
            privilege_escalation_count > 0
            and (
                command_sink_count
                + dynamic_winapi_resolution_count
            ) > 0
        ),
        int(
            dynamic_winapi_resolution_count > 0
            and memory_execution_api_count > 0
        ),
        int(
            wifi_credential_api_count > 0
            and exfiltration_channel_count > 0
        ),
        int(
            destructive_api_count > 0
            and (command_sink_count + network_sink_count) > 0
        ),
    ))
    concatenation_count = len(re.findall(r"(?:\+|\.format\s*\(|f['\"]|sprintf\s*\()", content))
    long_identifier_count = len(re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]{32,}\b", content))
    numeric_escape_count = len(re.findall(r"(?:\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|0x[0-9a-f]{4,})", lowered))
    dangerous_api_count = float(sum(lowered.count(api) for api in DANGEROUS_APIS))
    url_count = float(len(re.findall(r"https?://", lowered)))
    minified_line_ratio = round(sum(len(line) >= 300 for line in lines) / line_count, 4)
    comment_ratio_proxy = round(
        len(re.findall(r"(^\s*#|//|/\*)", content, re.MULTILINE)) / line_count,
        4,
    )
    features = {
        "byte_length": float(len(content.encode("utf-8", errors="ignore"))),
        "character_length": float(len(content)),
        "line_count": float(len(lines)),
        "average_line_length": round(sum(line_lengths) / len(line_lengths), 4),
        "max_line_length": float(max(line_lengths)),
        "blank_line_ratio": round(sum(not line.strip() for line in lines) / max(1, len(lines)), 4),
        "string_entropy": round(_entropy(content), 4),
        "maximum_string_entropy": round(max((_entropy(value) for value in unique_string_values), default=0.0), 4),
        "long_string_count": float(sum(len(value) >= 80 for value in string_values)),
        "dangerous_api_count": dangerous_api_count,
        "dynamic_exec_count": float(len(re.findall(r"\b(eval|exec|assert|compile)\s*\(", lowered))),
        "process_execution_count": float(len(re.findall(r"\b(system|popen|spawn|subprocess|child_process|runtime\.getruntime)\b", lowered))),
        "network_api_count": float(len(re.findall(r"\b(requests\.|urllib|http\.|https\.|fetch\s*\(|socket|curl|wget)\b", lowered))),
        "url_count": url_count,
        "ip_count": float(len(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", lowered))),
        "base64_indicator_count": float(len(re.findall(r"base64|b64decode|atob\s*\(", lowered))),
        "hex_escape_count": float(len(re.findall(r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}", lowered))),
        "encoded_blob_count": float(len(re.findall(r"[a-z0-9+/]{80,}={0,2}", lowered))),
        "install_hook_count": float(len(re.findall(r"\b(preinstall|postinstall|prepare|setup_requires|cmdclass)\b", lowered))),
        "sensitive_file_count": float(len(re.findall(r"(\.ssh|\.npmrc|\.pypirc|id_rsa|passwd|shadow|credentials|wallet\.dat)", lowered))),
        "environment_access_count": float(len(re.findall(r"(os\.environ|process\.env|getenv\s*\(|environment\.getenv)", lowered))),
        "file_write_count": float(len(re.findall(r"(open\s*\([^\n]{0,80}['\"](?:w|a|wb|ab)['\"]|writefile|file_put_contents|\.write\s*\()", lowered))),
        "crypto_hash_count": float(len(re.findall(r"\b(md5|sha1|sha256|aes|rsa|cryptography|crypto\.)\b", lowered))),
        "control_flow_count": float(len(re.findall(r"\b(if|for|while|try|catch|except|switch|case)\b", lowered))),
        "function_definition_count": float(len(re.findall(r"\b(def|function|func|public|private|protected)\s+[a-z_$]", lowered))),
        "import_count": float(len(re.findall(r"^\s*(import|from|require\s*\(|include|#include)", lowered, re.MULTILINE))),
        "dependency_marker_count": float(len(re.findall(r"\b(dependencies|devdependencies|requirements|install_requires|version)\b", lowered))),
        "minified_line_ratio": minified_line_ratio,
        "comment_ratio_proxy": comment_ratio_proxy,
        "python_ast_node_count": float(ast_metrics.get("nodes", 0)),
        "python_ast_call_count": float(ast_metrics.get("calls", 0)),
        "python_ast_branch_count": float(ast_metrics.get("branches", 0)),
        "python_ast_parse_failed": float(ast_metrics.get("failed", 0)),
        "source_input_count": float(source_count),
        "sql_sink_count": float(sql_sink_count),
        "command_sink_count": float(command_sink_count),
        "file_sink_count": float(file_sink_count),
        "network_sink_count": float(network_sink_count),
        "decode_operation_count": float(decode_count),
        "sanitizer_count": float(sanitizer_count),
        "install_hook_signal_count": float(import_hook_count),
        "native_process_api_count": float(native_process_api_count),
        "native_network_api_count": float(native_network_api_count),
        "process_injection_api_count": float(process_injection_api_count),
        "persistence_api_count": float(persistence_api_count),
        "credential_access_api_count": float(credential_access_api_count),
        "anti_analysis_api_count": float(anti_analysis_api_count),
        "destructive_api_count": float(destructive_api_count),
        "dynamic_winapi_resolution_count": float(dynamic_winapi_resolution_count),
        "privilege_escalation_count": float(privilege_escalation_count),
        "memory_execution_api_count": float(memory_execution_api_count),
        "wifi_credential_api_count": float(wifi_credential_api_count),
        "self_delete_api_count": float(self_delete_api_count),
        "system_identity_api_count": float(system_identity_api_count),
        "sensitive_collection_count": float(sensitive_collection_count),
        "exfiltration_channel_count": float(exfiltration_channel_count),
        "collection_exfiltration_proxy": float(
            sensitive_collection_count > 0
            and exfiltration_channel_count > 0
        ),
        "credential_remote_transfer_proxy": float(
            credential_access_api_count > 0
            and exfiltration_channel_count > 0
        ),
        "network_service_structure_count": float(
            network_service_structure_count
        ),
        "heavy_network_service_structure_proxy": float(
            network_service_structure_count >= 5
        ),
        "extended_command_channel_api_count": float(
            extended_command_channel_api_count
        ),
        "extended_network_channel_api_count": float(
            extended_network_channel_api_count
        ),
        "rust_offensive_behavior_count": float(
            rust_offensive_behavior_count
        ),
        "rust_offensive_behavior_proxy": float(
            rust_offensive_behavior_count > 0
        ),
        "rust_multi_offensive_behavior_proxy": float(
            rust_offensive_behavior_count >= 2
        ),
        "ruby_offensive_behavior_count": float(
            ruby_offensive_behavior_count
        ),
        "ruby_offensive_behavior_proxy": float(
            ruby_offensive_behavior_count > 0
        ),
        "ruby_multi_offensive_behavior_proxy": float(
            ruby_offensive_behavior_count >= 2
        ),
        "python_offensive_behavior_count": float(
            python_offensive_behavior_count
        ),
        "python_offensive_behavior_proxy": float(
            python_offensive_behavior_count > 0
        ),
        "python_multi_offensive_behavior_proxy": float(
            python_offensive_behavior_count >= 2
        ),
        "go_offensive_behavior_count": float(
            go_offensive_behavior_count
        ),
        "go_offensive_behavior_proxy": float(
            go_offensive_behavior_count > 0
        ),
        "go_multi_offensive_behavior_proxy": float(
            go_offensive_behavior_count >= 2
        ),
        "go_reverse_shell_proxy": float(
            go_reverse_shell_proxy
        ),
        "go_download_execute_proxy": float(
            go_download_execute_proxy
        ),
        "go_local_development_tool_proxy": float(
            go_local_development_tool_proxy
        ),
        "powershell_offensive_behavior_count": float(
            powershell_offensive_behavior_count
        ),
        "powershell_offensive_behavior_proxy": float(
            powershell_offensive_behavior_count > 0
        ),
        "powershell_multi_offensive_behavior_proxy": float(
            powershell_offensive_behavior_count >= 2
        ),
        "powershell_download_execute_persist_proxy": float(
            powershell_download_execute_persist_proxy
        ),
        "powershell_encoded_remote_loader_proxy": float(
            powershell_encoded_remote_loader_proxy
        ),
        "java_socket_command_shell_proxy": float(
            java_socket_command_shell_proxy
        ),
        "java_download_execute_proxy": float(
            java_download_execute_proxy
        ),
        "java_local_development_tool_proxy": float(
            java_local_development_tool_proxy
        ),
        "javascript_remote_eval_loader_proxy": float(
            javascript_remote_eval_loader_proxy
        ),
        "javascript_credential_exfiltration_proxy": float(
            javascript_credential_exfiltration_proxy
        ),
        "javascript_local_development_tool_proxy": float(
            javascript_local_development_tool_proxy
        ),
        "php_prepared_query_proxy": float(
            php_prepared_query_proxy
        ),
        "php_encoded_webshell_proxy": float(
            php_encoded_webshell_proxy
        ),
        "php_credential_exfiltration_proxy": float(
            php_credential_exfiltration_proxy
        ),
        "bash_build_script_structure_count": float(
            bash_build_script_structure_count
        ),
        "bash_build_script_structure_proxy": float(
            bash_build_script_structure_count >= 2
            or bash_android_cross_compile_proxy
        ),
        "bash_android_cross_compile_proxy": float(
            bash_android_cross_compile_proxy
        ),
        "bash_local_archive_structure_proxy": float(
            bash_local_archive_structure_proxy
        ),
        "native_behavior_group_count": float(native_behavior_group_count),
        "native_multi_behavior_group_proxy": float(
            native_behavior_group_count >= 2
        ),
        "file_local_behavior_group_count": float(
            file_local_behavior_group_count
        ),
        "file_local_multi_behavior_group_proxy": float(
            file_local_behavior_group_count >= 2
        ),
        "html_password_input_count": float(html_password_input_count),
        "html_form_count": float(html_form_count),
        "html_external_form_action_count": float(html_external_form_action_count),
        "html_hidden_element_count": float(html_hidden_element_count),
        "html_login_keyword_count": float(html_login_keyword_count),
        "html_script_obfuscation_count": float(html_script_obfuscation_count),
        "behavior_chain_count": float(behavior_chain_count),
        "advanced_malicious_behavior_count": float(
            advanced_malicious_behavior_count
        ),
        "advanced_malicious_behavior_proxy": float(
            advanced_malicious_behavior_count > 0
        ),
        "advanced_or_three_behavior_groups_proxy": float(
            advanced_malicious_behavior_count > 0
            or file_local_behavior_group_count >= 3
        ),
        "string_concatenation_count": float(concatenation_count),
        "long_identifier_count": float(long_identifier_count),
        "numeric_escape_count": float(numeric_escape_count),
        "source_to_sink_proxy": float(source_count > 0 and (
            sql_sink_count + command_sink_count + file_sink_count
        ) > 0),
        "network_write_execute_proxy": float(
            network_sink_count > 0 and (file_sink_count + command_sink_count) > 0
        ),
        "decode_execute_proxy": float(decode_count > 0 and command_sink_count > 0),
        "native_injection_or_persistence_proxy": float(
            process_injection_api_count > 0 or persistence_api_count > 0
        ),
        "dynamic_winapi_memory_execution_proxy": float(
            dynamic_winapi_resolution_count > 0 and memory_execution_api_count > 0
        ),
        "socket_command_channel_proxy": float(
            (network_sink_count + extended_network_channel_api_count) > 0
            and (command_sink_count + extended_command_channel_api_count) > 0
        ),
        "wifi_credential_access_proxy": float(
            wifi_credential_api_count > 0
        ),
        "privilege_escalation_process_proxy": float(
            privilege_escalation_count > 0
            and (command_sink_count + dynamic_winapi_resolution_count) > 0
        ),
        "self_delete_native_proxy": float(
            self_delete_api_count > 0
            and (
                dynamic_winapi_resolution_count
                + memory_execution_api_count
                + native_process_api_count
            ) > 0
        ),
        "credential_exfiltration_proxy": float(
            credential_access_api_count > 0 and network_sink_count > 0
        ),
        "semantic_behavior_token_count": float(
            sum(token.startswith("__bt_") for token in semantic_tokens)
        ),
        "semantic_process_exec": float("__bt_process_exec__" in semantic_tokens),
        "semantic_download": float("__bt_download__" in semantic_tokens),
        "semantic_network": float("__bt_network__" in semantic_tokens),
        "semantic_persistence": float("__bt_persistence__" in semantic_tokens),
        "semantic_injection": float("__bt_injection__" in semantic_tokens),
        "semantic_security_bypass": float(
            "__bt_security_bypass__" in semantic_tokens
        ),
        "semantic_shellcode": float("__bt_shellcode__" in semantic_tokens),
        "semantic_credentials": float("__bt_credentials__" in semantic_tokens),
        "semantic_obfuscation": float("__bt_obfuscation__" in semantic_tokens),
        "semantic_malicious_chain_count": float(
            sum(token.startswith("__bt_chain_") for token in semantic_tokens)
        ),
        "semantic_ps_benign_structure_count": float(
            sum(
                token in {
                    "__bt_ps_linter_settings__",
                    "__bt_ps_signed_linter_settings__",
                    "__bt_ps_module_manifest__",
                    "__bt_localization_resource__",
                }
                for token in semantic_tokens
            )
        ),
        "html_credential_form_proxy": float(
            html_password_input_count > 0
            and html_form_count > 0
            and html_login_keyword_count > 0
        ),
        "html_suspicious_form_proxy": float(
            html_password_input_count > 0
            and (
                html_external_form_action_count > 0
                or html_script_obfuscation_count > 0
                or html_hidden_element_count >= 2
            )
        ),
        "risk_density": float(
            (float(rule_metrics.get("rule_severity_sum", 0.0)) + behavior_chain_count)
            / line_count
        ),
        "dangerous_api_density": float(dangerous_api_count / line_count),
        "url_density": float(url_count / byte_count * 1000.0),
        "comment_density": float(comment_ratio_proxy),
        "minified_byte_density": float(minified_line_ratio),
        **rule_metrics,
    }
    for bucket in LANGUAGE_BUCKETS:
        features[f"language_{bucket}"] = 1.0 if language == bucket else 0.0
    features["language_unknown"] = 1.0 if language not in LANGUAGE_BUCKETS else 0.0
    return features


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _python_ast_metrics(content: str) -> dict[str, int]:
    try:
        # Third-party source frequently contains regex/docstring escapes that
        # are valid source text but emit SyntaxWarning while being parsed.
        # They are not relevant to the static metrics and should not flood a
        # long training run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
    except (SyntaxError, ValueError, MemoryError):
        return {"nodes": 0, "calls": 0, "branches": 0, "failed": 1}
    nodes = list(ast.walk(tree))
    return {
        "nodes": len(nodes),
        "calls": sum(isinstance(node, ast.Call) for node in nodes),
        "branches": sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in nodes),
        "failed": 0,
    }


def _rule_metrics(content: str, language: str) -> dict[str, float]:
    from attack_detection.rules import RULES, detect_by_rules

    matches = detect_by_rules(content, language)
    metrics = {
        "rule_match_count": float(len(matches)),
        "rule_malicious_count": float(sum(item.get("risk_type") == "malicious" for item in matches)),
        "rule_vulnerable_count": float(sum(item.get("risk_type") == "vulnerable" for item in matches)),
        "rule_suspicious_count": float(sum(item.get("risk_type") == "suspicious" for item in matches)),
        "rule_severity_sum": float(sum(int(item.get("severity") or 0) for item in matches)),
        "rule_max_severity": float(max((int(item.get("severity") or 0) for item in matches), default=0)),
    }
    matched_ids = {str(item.get("rule_id") or "") for item in matches}
    for rule in RULES:
        metrics[f"rule_{rule.rule_id.lower().replace('-', '_')}"] = 1.0 if rule.rule_id in matched_ids else 0.0
    metrics["rule_path_002"] = 1.0 if "PATH-002" in matched_ids else 0.0
    return metrics


FEATURE_NAMES = tuple(extract_static_features("", "unknown").keys())


def feature_vector(
    content: str,
    language: str,
    feature_names: list[str] | tuple[str, ...] = FEATURE_NAMES,
    *,
    include_rules: bool = True,
) -> list[float]:
    values = extract_static_features(content, language, include_rules=include_rules)
    return [float(values.get(name, 0.0)) for name in feature_names]
