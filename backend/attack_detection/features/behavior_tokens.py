"""Deterministic semantic tokens for language-specific text classifiers.

The tokens bridge equivalent behavior APIs across languages while keeping the
original source text intact. No decoding, importing, compilation, or execution
is performed.
"""

from __future__ import annotations

import re
from functools import lru_cache


BEHAVIOR_TOKEN_VERSION = "behavior_tokens_v1"
BEHAVIOR_TOKEN_VERSION_V2 = "behavior_tokens_v2"
BEHAVIOR_TOKEN_VERSION_V3 = "behavior_tokens_v3"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "__bt_process_exec__",
        re.compile(
            r"(?:\b(?:invoke-expression|iex|start-process|cmd(?:\.exe)?|"
            r"powershell(?:\.exe)?|createprocess(?:a|w)?|winexec|shellexecute|"
            r"system|popen|execve|os/exec|exec\.command|command::new|"
            r"std::process::command)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_download__",
        re.compile(
            r"(?:\b(?:downloadfile|downloadstring|invoke-webrequest|iwr|wget|curl|"
            r"urlmon|urldownloadtofile|webclient|http\.get|client\.do)\b|"
            r"https?://)",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_network__",
        re.compile(
            r"\b(?:tcpclient|udpclient|socket|connect|net/http|http\.newrequest|"
            r"tcpstream|tcplistener|reqwest|tokio::net|websocket|reverse.?shell)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_persistence__",
        re.compile(
            r"(?:currentversion[\\/]+run|\b(?:register-scheduledtask|schtasks|"
            r"new-service|createservice|startup|launchagent|crontab|systemd|"
            r"autorun|set-itemproperty)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_injection__",
        re.compile(
            r"\b(?:virtualalloc(?:ex)?|writeprocessmemory|createremotethread|"
            r"ntwritevirtualmemory|queueuserapc|rtlmovememory|openprocess|"
            r"process.?hollow|reflective.?load|ptrace|process_vm_writev)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_security_bypass__",
        re.compile(
            r"\b(?:add-mppreference|set-mppreference|amsi(?:utils|initfailed)?|"
            r"disableantispyware|exclusionpath|etwpatch|bypassuac|uac.?bypass|"
            r"executionpolicy\s+bypass)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_shellcode__",
        re.compile(
            r"(?:\bshellcode\b|\[\s*byte\s*\[\s*\]\s*\].{0,200}"
            r"(?:0x[0-9a-f]{1,2}\s*,\s*){8,}|"
            r"(?:0x[0-9a-f]{1,2}\s*,\s*){16,})",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "__bt_credentials__",
        re.compile(
            r"\b(?:login data|local state|dpapi|cryptunprotectdata|credential|"
            r"password|passwd|browser.?cookie|discord.?token|wallet|seed phrase|"
            r"keylog(?:ger)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_crypto_destructive__",
        re.compile(
            r"\b(?:ransom|encryptfile|decryptfile|deletefile|remove-item|"
            r"unlink|wipe|shred|shadowcopy|vssadmin|cipher::new|aes256|chacha20)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_obfuscation__",
        re.compile(
            r"\b(?:frombase64string|base64decode|invoke-obfuscation|"
            r"encodedcommand|charcode|fromcharcode|gzipstream|deflatestream|"
            r"xor.?decode|assembly\.load)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_capture__",
        re.compile(
            r"\b(?:copyfromscreen|screenshot|system\.windows\.forms\.screen|"
            r"getasynckeystate|setwindowshookex|clipboard)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_antianalysis__",
        re.compile(
            r"\b(?:isdebuggerpresent|checkremotedebuggerpresent|"
            r"ntqueryinformationprocess|virtualbox|vmware|sandbox|anti.?vm|"
            r"anti.?debug)\b",
            re.IGNORECASE,
        ),
    ),
)

_V3_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "__bt_privilege_escalation__",
        re.compile(
            r"\b(?:bypassuac|uac.?bypass|elevat(?:e|ed|ion)|fodhelper|"
            r"computerdefaults|sdclt|token.?elevation|sedebugprivilege)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_dynamic_winapi__",
        re.compile(
            r"\b(?:newlazydll|newproc|getprocaddress|loadlibrary(?:a|w)?|"
            r"getmodulehandle(?:a|w)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_memory_execution__",
        re.compile(
            r"\b(?:virtualprotect|virtualalloc(?:ex)?|page_execute(?:_readwrite)?|"
            r"writeprocessmemory|createremotethread)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_wifi_credentials__",
        re.compile(
            r"\b(?:wlangetprofile|wlan_profile_get_plaintext_key|"
            r"wlanprofile|wireless.?password|wifi.?password)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_self_delete__",
        re.compile(
            r"\b(?:filedispositioninfo|filerenameinfo|movefileex(?:a|w)?|"
            r"deletefile(?:a|w)?|self.?delete|self.?erase)\b",
            re.IGNORECASE,
        ),
    ),
)

_POWERSHELL_STRUCTURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "__bt_ps_linter_settings__",
        re.compile(
            r"(?:psscriptanalyzer|includerules|excluderules|"
            r"psuseapprovedverbs|psavoidusing)",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_ps_signed_linter_settings__",
        re.compile(
            r"(?s)(?:psscriptanalyzer|includerules|excluderules).{0,20000}"
            r"#\s*sig\s*#\s*begin signature block",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_ps_module_manifest__",
        re.compile(
            r"(?:rootmodule|moduleversion|functionstoexport|"
            r"cmdletstoexport|requiredmodules)",
            re.IGNORECASE,
        ),
    ),
    (
        "__bt_localization_resource__",
        re.compile(
            r"(?:convertfrom-stringdata|localizeddata|strings\.psd1|"
            r"cultureinfo|localizations?[\\/])",
            re.IGNORECASE,
        ),
    ),
)


def behavior_tokens(code: str, language: str) -> list[str]:
    """Return stable semantic tokens for a source file."""

    return list(_cached_behavior_tokens(code, language))


@lru_cache(maxsize=256)
def _cached_behavior_tokens(
    code: str,
    language: str,
) -> tuple[str, ...]:
    tokens = [token for token, pattern in _PATTERNS if pattern.search(code)]
    if language.lower() == "powershell":
        tokens.extend(
            token
            for token, pattern in _POWERSHELL_STRUCTURE_PATTERNS
            if pattern.search(code)
        )
    token_set = set(tokens)
    if {"__bt_download__", "__bt_process_exec__"} <= token_set:
        tokens.append("__bt_chain_download_exec__")
    if {"__bt_network__", "__bt_process_exec__"} <= token_set:
        tokens.append("__bt_chain_network_exec__")
    if {"__bt_security_bypass__", "__bt_download__"} <= token_set:
        tokens.append("__bt_chain_bypass_download__")
    if {"__bt_injection__", "__bt_shellcode__"} <= token_set:
        tokens.append("__bt_chain_injection_shellcode__")
    return tuple(sorted(set(tokens)))


def behavior_tokens_v3(code: str, language: str) -> list[str]:
    """Return legacy tokens plus additive multilingual native-code signals."""

    tokens = behavior_tokens(code, language)
    tokens.extend(
        token for token, pattern in _V3_PATTERNS if pattern.search(code)
    )
    lowered = code.lower()
    if re.search(r"\b(?:tcpsocket|tcpserver|udpsocket)\b", lowered):
        tokens.append("__bt_network__")
    if re.search(r"\b(?:open3|io\.popen|kernel\.exec)\b", lowered):
        tokens.append("__bt_process_exec__")
    token_set = set(tokens)
    if {"__bt_network__", "__bt_process_exec__"} <= token_set:
        tokens.append("__bt_chain_network_exec__")
    if {"__bt_dynamic_winapi__", "__bt_memory_execution__"} <= token_set:
        tokens.append("__bt_chain_dynamic_memory_exec__")
    if {"__bt_network__", "__bt_credentials__"} <= token_set:
        tokens.append("__bt_chain_credential_network__")
    if {"__bt_network__", "__bt_wifi_credentials__"} <= token_set:
        tokens.append("__bt_chain_wifi_credential_network__")
    return sorted(set(tokens))


def behavior_token_text(code: str, language: str) -> str:
    """Append semantic tokens to the unchanged source text."""

    tokens = behavior_tokens(code, language)
    if not tokens:
        return code
    return code + "\n" + " ".join(tokens)


def behavior_token_text_v2(code: str, language: str) -> str:
    """Append repeated semantic tokens so hashed text features retain signal."""

    tokens = behavior_tokens(code, language)
    if not tokens:
        return code
    return code + "\n" + " ".join(tokens * 4)


def behavior_token_text_v3(code: str, language: str) -> str:
    """Append repeated additive v3 tokens without changing legacy v1/v2."""

    tokens = behavior_tokens_v3(code, language)
    if not tokens:
        return code
    return code + "\n" + " ".join(tokens * 4)
