"""Download a small, bounded set of popular npm package manifests.

Only public registry JSON is downloaded.  Packages are not installed and no
package lifecycle script is executed.  The output is JSONL so the dataset
builder can use format-matched benign ``package.json`` examples.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


QUERIES = (
    "react",
    "typescript",
    "eslint",
    "webpack",
    "express",
    "testing",
    "babel",
    "vite",
    "cli",
    "node",
    "database",
    "http",
)
USER_AGENT = "Xiezhi-CodeGuard-Dataset-Builder/1.0"


def _json(url: str, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.headers.get("Content-Length") or 0) > 4 * 1024 * 1024:
            raise ValueError("registry response exceeds 4 MiB")
        payload = response.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("registry response exceeds 4 MiB")
    return json.loads(payload.decode("utf-8"))


def acquire(output: Path, limit: int) -> dict[str, Any]:
    names: dict[str, float] = {}
    errors: list[dict[str, str]] = []
    for query in QUERIES:
        url = (
            "https://registry.npmjs.org/-/v1/search?"
            + urllib.parse.urlencode({"text": query, "size": 50, "from": 0})
        )
        try:
            result = _json(url)
            for item in result.get("objects", []):
                package = item.get("package") or {}
                score = item.get("score") or {}
                detail = score.get("detail") or {}
                name = str(package.get("name") or "")
                popularity = float(detail.get("popularity") or 0.0)
                if name and popularity >= 0.35:
                    names[name] = max(names.get(name, 0.0), popularity)
        except Exception as exc:
            errors.append({"query": query, "error": str(exc)})

    selected = sorted(
        names.items(),
        key=lambda item: (-item[1], item[0]),
    )[: max(1, limit)]

    def download(item: tuple[str, float]) -> dict[str, Any]:
        name, popularity = item
        encoded = urllib.parse.quote(name, safe="")
        url = f"https://registry.npmjs.org/{encoded}/latest"
        manifest = _json(url, timeout=10.0)
        scripts = manifest.get("scripts")
        if scripts is not None and not isinstance(scripts, dict):
            raise ValueError("manifest scripts field is not an object")
        return {
            "package_name": name,
            "version": str(manifest.get("version") or ""),
            "popularity": popularity,
            "source_url": url,
            "manifest": manifest,
        }

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(download, item): item[0]
            for item in selected
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"package_name": name, "error": str(exc)})
    rows.sort(key=lambda item: str(item["package_name"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "output": str(output.resolve()),
        "queries": list(QUERIES),
        "candidate_names": len(names),
        "downloaded_manifests": len(rows),
        "errors": errors,
        "packages_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(
        acquire(args.output.resolve(), min(500, max(20, args.limit))),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
