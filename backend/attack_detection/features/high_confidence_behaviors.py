"""High-confidence, code-local behavior groups for native offensive tooling."""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings


_RUST_PROCESS_INJECTION = re.compile(
    r"\b(?:ntcreateuserprocess|zwcreateuserprocess|ntwritevirtualmemory|"
    r"ntprotectvirtualmemory|ntcreatethreadex|virtualallocex|"
    r"writeprocessmemory|createremotethread|queueuserapc|setthreadcontext|"
    r"rtlcreateuserthread)\b",
    re.IGNORECASE,
)
_RUST_SECURITY_BYPASS = re.compile(
    r"\b(?:amsiscanbuffer|amsi.?bypass|etweventwrite|nttraceevent|"
    r"patchless|unhook(?:ing)?|edr.?bypass|defender.?bypass|"
    r"ntqueryinformationprocess|isdebuggerpresent)\b",
    re.IGNORECASE,
)
_RUST_PRIVILEGE_TOKEN = re.compile(
    r"\b(?:duplicatetokenex|adjusttokenprivileges|createprocesswithtoken|"
    r"createprocessasuser|sedebugprivilege|token.?elevat(?:e|ion)|"
    r"uac.?bypass)\b",
    re.IGNORECASE,
)
_RUST_CREDENTIAL_ACCESS = re.compile(
    r"\b(?:minidumpwritedump|lsass|samlib|cryptunprotectdata|"
    r"wlangetprofile|credential.?dump|wifi.?dump)\b",
    re.IGNORECASE,
)
_RUST_EXPLICIT_OFFENSIVE_INTENT = re.compile(
    r"\b(?:process.?hollow(?:ing)?|shellcode|reflective.?load|dll.?proxy|"
    r"side.?load|anti.?debug|anti.?vm|syscall.?stub|hell.?s.?gate|"
    r"halo.?s.?gate|dumpmdeconfig)\b",
    re.IGNORECASE,
)


def _comment_stripped(content: str) -> str:
    executable = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    return re.sub(r"(?m)(?<!:)//[^\r\n]*", " ", executable)


def _go_comment_stripped(content: str) -> str:
    executable = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    return re.sub(r"(?m)//[^\r\n]*", " ", executable)


def _powershell_comment_stripped(content: str) -> str:
    executable = re.sub(r"<#.*?#>", " ", content, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*#[^\r\n]*", " ", executable)


def _ruby_comment_stripped(content: str) -> str:
    """Remove Ruby comment-only lines without damaging interpolation."""

    executable = re.sub(r"(?ms)^=begin\b.*?^=end\b", " ", content)
    return re.sub(r"(?m)^\s*#[^\r\n]*", " ", executable)


def _python_executable_text(content: str) -> str:
    """Remove comments and true docstrings while retaining payload strings."""

    lines = content.splitlines(keepends=True)
    docstring_lines: set[int] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
        bodies = [tree.body]
        bodies.extend(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        for body in bodies:
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                start = int(getattr(first, "lineno", 1))
                end = int(getattr(first, "end_lineno", start))
                docstring_lines.update(range(start, end + 1))
    except (SyntaxError, ValueError, TypeError):
        pass
    without_docstrings = "".join(
        "\n" if index in docstring_lines else line
        for index, line in enumerate(lines, start=1)
    )
    try:
        tokens = tokenize.generate_tokens(io.StringIO(without_docstrings).readline)
        return tokenize.untokenize(
            token for token in tokens if token.type != tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"(?m)#.*$", " ", without_docstrings)


def python_high_confidence_behavior_count(content: str) -> int:
    """Count specific file-local malicious behavior chains in Python source."""

    executable = _python_executable_text(content)
    lowered = executable.lower()
    decode = bool(re.search(
        r"\b(?:base64\.(?:b64decode|b85decode|a85decode)|b64decode|"
        r"marshal\.loads|zlib\.decompress|bz2\.decompress|"
        r"lzma\.decompress|codecs\.decode|bytes\.fromhex)\b",
        lowered,
    ))
    execute = bool(re.search(
        r"\b(?:exec|eval|compile)\s*\(|\bos\.system\s*\(|"
        r"(?<![\w.])system\s*\(|\bos\.popen\s*\(|"
        r"(?<![\w.])popen\s*\(|"
        r"\bsubprocess\.(?:run|call|popen|check_output|check_call)\s*\(",
        lowered,
    ))
    fetch = bool(re.search(
        r"\b(?:requests\.(?:get|post)|httpx\.(?:get|post)|"
        r"urllib(?:\.request)?\.(?:urlopen|urlretrieve)|"
        r"urlopen|urlretrieve)\b",
        lowered,
    ))
    write = bool(re.search(
        r"\b(?:open\s*\([^\n]{0,120}['\"](?:w|a|wb|ab)|"
        r"write\s*\(|pathlib\.|shutil\.copy|tempfile\.)",
        lowered,
    ))
    collect = bool(re.search(
        r"\b(?:os\.environ|os\.getenv|login data|local state|discord|"
        r"wallet|seed phrase|browser|cookies?|passwords?|credentials?|"
        r"webcam|screenshot|keylog|clipboard|wifi)\b",
        lowered,
    ))
    exfiltrate = bool(re.search(
        r"\b(?:webhook|requests\.post|httpx\.post|urlopen|"
        r"socket\.socket|smtplib|ftplib|telegram|discord_webhook)\b",
        lowered,
    ))
    install_hook = bool(re.search(
        r"\b(?:cmdclass|class\s+\w+\s*\(\s*(?:install|develop|egg_info)|"
        r"setuptools\.command\.(?:install|develop)|setup\s*\()",
        lowered,
    ))
    shell_marker = bool(re.search(
        r"\b(?:cmd\.exe|powershell(?:\.exe)?|/bin/(?:ba)?sh|"
        r"reverse.?shell|meterpreter)\b",
        lowered,
    ))
    socket_channel = bool(re.search(
        r"\b(?:socket\.socket|create_connection|connect_ex|recv|sendall)\b",
        lowered,
    ))
    traverse = bool(re.search(
        r"\b(?:os\.walk|path\.rglob|glob\.glob|dirpath|filenames)\b",
        lowered,
    ))
    encrypt = bool(re.search(
        r"\b(?:fernet|aes|cipher|encrypt|ransom)\b",
        lowered,
    ))
    persistence = bool(re.search(
        r"\b(?:schtasks|currentversion\\run|startup|appdata|"
        r"crontab|systemd|autorun)\b",
        lowered,
    ))
    explicit_payload = bool(re.search(
        r"\b(?:download_and_run|stealer|grabber|keylogger|ransomware)\b",
        lowered,
    ))
    return sum((
        decode and execute,
        fetch and write and execute,
        collect and exfiltrate,
        install_hook and (
            decode
            or (fetch and (execute or write))
            or exfiltrate
        ),
        socket_channel and execute and (
            shell_marker
            or bool(re.search(r"\b(?:recv|sendall|connect)\b", lowered))
        ),
        traverse and encrypt,
        persistence and (fetch or write or execute),
        explicit_payload and (
            decode or fetch or execute or collect or exfiltrate
        ),
    ))


def go_high_confidence_behavior_count(content: str) -> int:
    """Count specific file-local offensive behavior chains in Go source."""

    lowered = _go_comment_stripped(content).lower()
    socket_channel = bool(re.search(
        r"\b(?:net\.(?:dial|dialtcp)|tcpconn|unixconn)\s*\(",
        lowered,
    ))
    process_execution = bool(re.search(
        r"\b(?:exec\.command|os\.startprocess)\s*\(",
        lowered,
    ))
    shell_process = bool(re.search(
        r"(?:exec\.command|os\.startprocess)\s*\(\s*"
        r"[`'\"](?:/bin/(?:ba)?sh|cmd(?:\.exe)?|powershell(?:\.exe)?)",
        lowered,
    ))
    socket_stdio = sum(
        bool(re.search(rf"\.\s*{stream}\s*=\s*\w+", lowered))
        for stream in ("stdin", "stdout", "stderr")
    ) >= 2
    fetch = bool(re.search(
        r"\b(?:http\.(?:get|newrequest)|client\.do)\s*\(",
        lowered,
    ))
    file_write = bool(re.search(
        r"\b(?:os\.(?:create|openfile)|io\.copy)\s*\(",
        lowered,
    ))
    downloaded_process = bool(re.search(
        r"\bexec\.command\s*\(\s*(?:destination|dest|path|filename|tmp|"
        r"output|outfile|target)\b",
        lowered,
    ))
    persistence = bool(re.search(
        r"\b(?:currentversion\\\\run|schtasks|crontab|systemd|"
        r"launchagents?|startup)\b",
        lowered,
    ))
    collect = bool(re.search(
        r"\b(?:login data|local state|browser.?cookie|discord.?token|"
        r"wallet|id_rsa|credentials?|passwords?|screenshot|keylog)\b",
        lowered,
    ))
    exfiltrate = bool(re.search(
        r"\b(?:multipart\.newwriter|client\.post|http\.post|smtp\.sendmail|"
        r"webhook|upload)\b",
        lowered,
    ))
    return sum((
        socket_channel and process_execution and shell_process and socket_stdio,
        fetch and file_write and process_execution and downloaded_process,
        fetch and file_write and process_execution and persistence,
        collect and exfiltrate,
    ))


def powershell_high_confidence_behavior_count(content: str) -> int:
    """Count specific file-local offensive behavior chains in PowerShell."""

    lowered = _powershell_comment_stripped(content).lower()
    fetch = bool(re.search(
        r"\b(?:invoke-webrequest|invoke-restmethod|downloadstring|"
        r"downloadfile|net\.webclient)\b",
        lowered,
    ))
    execute = bool(re.search(
        r"\b(?:invoke-expression|iex|start-process|start-job|"
        r"powershell(?:\.exe)?\b[^\r\n;]{0,120}(?:-file|-command|-enc))\b",
        lowered,
    ))
    decode = bool(re.search(
        r"\b(?:frombase64string|text\.encoding|gzipstream|deflatestream)\b",
        lowered,
    ))
    persistence = bool(re.search(
        r"(?:currentversion\\\\run|currentversion\\run|new-itemproperty|"
        r"register-scheduledtask|schtasks|startup|wmi.*eventconsumer)",
        lowered,
    ))
    hidden = bool(re.search(
        r"(?:-windowstyle\s+hidden|-w(?:indowstyle)?\s+hidden|"
        r"windowstyle\s*=\s*['\"]?hidden)",
        lowered,
    ))
    credential_access = bool(re.search(
        r"\b(?:lsass|mimikatz|sekurlsa|sam hive|cryptunprotectdata|"
        r"login data|local state|wlangetprofile|credentials?)\b",
        lowered,
    ))
    exfiltrate = bool(re.search(
        r"\b(?:invoke-restmethod|webhook|uploadfile|multipart|send-mailmessage)\b",
        lowered,
    ))
    return sum((
        fetch and execute,
        fetch and persistence,
        decode and execute,
        execute and hidden and persistence,
        credential_access and exfiltrate,
    ))


def ruby_high_confidence_behavior_count(content: str) -> int:
    """Count specific, file-local offensive behavior chains in Ruby source.

    Repository names and comments are deliberately ignored. Generic uses of
    sockets, HTTP, OpenSSL, passwords, or ``system`` are not sufficient by
    themselves because all occur frequently in benign Ruby applications.
    """

    executable = _ruby_comment_stripped(content)
    lowered = executable.lower()
    socket_channel = bool(re.search(
        r"\b(?:tcpsocket|tcpserver|socket\.tcp_server_loop|udpsocket)\b",
        lowered,
    ))
    command_execution = bool(re.search(
        r"\b(?:open3(?:\.|::)|io\.popen|kernel\.(?:exec|system)|"
        r"system\s*\(|exec\s*\(|spawn\s*\(|capture[23]\s*\()",
        lowered,
    )) or bool(re.search(r"`[^`\r\n]{1,240}`", executable))
    socket_io = bool(re.search(
        r"\b(?:recv|recvfrom|readpartial|readline|gets|puts|send|write)\b",
        lowered,
    ))
    shell_marker = bool(re.search(
        r"(?:cmd\.exe|powershell(?:\.exe)?|/bin/(?:ba)?sh|"
        r"reverse.?shell|bind.?shell|rshell_open|command#|meterpreter)",
        lowered,
    ))
    key_capture = bool(re.search(
        r"\b(?:getasynckeystate|getkeystate|getkeyboardstate|"
        r"setwindowshookex|getforegroundwindow|win32api)\b",
        lowered,
    )) and bool(re.search(
        r"\b(?:key(?:board|log|code|press)|log\.txt|net::ftp|"
        r"ftp\.open|storlines|storbinary)\b",
        lowered,
    ))
    file_encryption = (
        bool(re.search(
            r"\b(?:openssl::cipher|cipher\.encrypt|encrypt_file|ransom)",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:dir\.glob|find\.find|file\.read|file\.binread|"
            r"file\.write|file\.open|rename|unlink|delete)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:cipher\.update|cipher\.final|encrypted|ransom|"
            r"readme|decrypt|extension)\b",
            lowered,
        ))
    )
    credential_harvest = (
        bool(re.search(
            r"\bparams\s*\[\s*['\"](?:user|username|email|pass|password)",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:logger|file\.(?:open|write)|net::http|http\.post|"
            r"net::smtp|sqlite|insert)\b",
            lowered,
        ))
    )
    metasploit_action = (
        bool(re.search(
            r"\b(?:metasploitmodule|metasploit3|msf::post)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:cmd_exec|session\.|registry_|meterpreter|"
            r"execute|migrate|persistence|payload)\b",
            lowered,
        ))
    )
    backdoor_protocol = (
        bool(re.search(
            r"\b(?:backdoor|payloadmaker|reverse.?shell|meterpreter)\b",
            lowered,
        ))
        and (
            (socket_channel and (socket_io or command_execution))
            or bool(re.search(
                r"\b(?:websocket|net::http|http\.post|upload|download|"
                r"shellcode|antisandbox)\b",
                lowered,
            ))
        )
    )
    c2_command_channel = (
        socket_channel
        and socket_io
        and (command_execution or shell_marker)
    )
    remote_command_generator = (
        bool(re.search(r"\b(?:net::http|open-uri|uri\.parse)\b", lowered))
        and bool(re.search(
            r"\b(?:reverse.?shell|payload|shellcode|backdoor)\b",
            lowered,
        ))
        and bool(re.search(
            r"\b(?:download|http\.get|net::http\.get|file\.write|eval)\b",
            lowered,
        ))
    )
    return sum((
        c2_command_channel,
        key_capture,
        file_encryption,
        credential_harvest,
        metasploit_action,
        backdoor_protocol,
        remote_command_generator,
    ))


def rust_high_confidence_behavior_count(content: str) -> int:
    """Count distinct Rust behavior chains with low benign prevalence.

    The detector deliberately ignores repository names and file paths.  A
    single generic socket, password, ``system`` identifier, or WinAPI import
    is not enough to trigger any group.
    """

    executable = _comment_stripped(content)
    lowered = executable.lower()
    groups = [
        bool(_RUST_PROCESS_INJECTION.search(executable)),
        bool(_RUST_SECURITY_BYPASS.search(executable)),
        bool(_RUST_PRIVILEGE_TOKEN.search(executable)),
        bool(_RUST_CREDENTIAL_ACCESS.search(executable)),
        bool(_RUST_EXPLICIT_OFFENSIVE_INTENT.search(executable)),
        (
            "setfileinformationbyhandle" in lowered
            and (
                "filedispositioninfo" in lowered
                or "filerenameinfo" in lowered
            )
            and "current_exe" in lowered
        ),
        (
            ("ldrloaddll" in lowered or "ldrunloaddll" in lowered)
            and (
                "createthreadpool" in lowered
                or "addvectoredexceptionhandler" in lowered
                or "getprocaddress" in lowered
            )
        ),
        (
            sum(
                marker in lowered
                for marker in (
                    "psloadedmodulelist",
                    "ktrap_frame",
                    "zwgetnextprocess",
                    "pssetcreatethreadnotifyroutine",
                    "alt syscall",
                    "ssdt",
                )
            )
            >= 2
        ),
        (
            (
                "tcpstream" in lowered
                or "tcplistener" in lowered
                or "tokio::net" in lowered
            )
            and (
                "process::command" in lowered
                or "command::new" in lowered
            )
            and sum(
                marker in lowered
                for marker in ("upload", "download", "persist", "cmd.exe")
            )
            >= 2
        ),
        (
            (
                "tcpstream" in lowered
                or "tcplistener" in lowered
                or "tokio::net" in lowered
            )
            and (
                "process::command" in lowered
                or "command::new" in lowered
            )
            and bool(re.search(
                r"(?:/bin/(?:ba)?sh|cmd(?:\.exe)?|powershell(?:\.exe)?)",
                lowered,
            ))
            and sum(
                marker in lowered
                for marker in (".stdin(", ".stdout(", ".stderr(")
            )
            >= 2
        ),
        (
            (
                "userauth_password" in lowered
                or "bruteforce" in lowered
                or "brute force" in lowered
            )
            and (
                "tcpstream" in lowered
                or "smbcredentials" in lowered
                or "ftp" in lowered
            )
        ),
        (
            (
                "encrypt" in lowered
                or "ransom" in lowered
            )
            and (
                "walkdir" in lowered
                or "read_dir" in lowered
                or "filedispositioninfo" in lowered
            )
        ),
    ]
    return sum(groups)
