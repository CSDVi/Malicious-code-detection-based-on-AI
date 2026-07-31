"""Small resumable parallel HTTP range downloader for public datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path


def _content_length(url: str) -> int:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Xiezhi-CodeGuard-dataset-prep/1.0", "Range": "bytes=0-0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
        length = response.headers.get("Content-Length")
        if length:
            return int(length)
        raise RuntimeError(f"server did not return a file size: {response.status}")


def _download_part(url: str, path: Path, start: int, end: int) -> tuple[int, int]:
    if path.is_file() and path.stat().st_size == end - start + 1:
        return start, end
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Xiezhi-CodeGuard-dataset-prep/1.0",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as stream:
        remaining = end - start + 1
        while remaining:
            block = response.read(min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"short range response for {start}-{end}")
            stream.write(block)
            remaining -= len(block)
    return start, end


def download(url: str, output: Path, workers: int, chunk_size: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url)
    if output.is_file() and output.stat().st_size == total:
        return {"url": url, "output": str(output), "bytes": total, "resumed": True}
    part_root = Path(tempfile.mkdtemp(prefix=f"{output.stem}.parts-", dir=str(output.parent)))
    ranges = [
        (start, min(total - 1, start + chunk_size - 1))
        for start in range(0, total, chunk_size)
    ]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [
                pool.submit(
                    _download_part,
                    url,
                    part_root / f"{start:012d}-{end:012d}.part",
                    start,
                    end,
                )
                for start, end in ranges
            ]
            for job in concurrent.futures.as_completed(jobs):
                job.result()
        temporary = output.with_suffix(output.suffix + ".partial")
        with temporary.open("wb") as stream:
            for start, end in ranges:
                part = part_root / f"{start:012d}-{end:012d}.part"
                with part.open("rb") as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
        if temporary.stat().st_size != total:
            raise RuntimeError(f"assembled size mismatch: {temporary.stat().st_size} != {total}")
        os.replace(temporary, output)
    finally:
        shutil.rmtree(part_root, ignore_errors=True)
    return {"url": url, "output": str(output), "bytes": total, "workers": workers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(download(args.url, args.output, args.workers, args.chunk_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
