#!/usr/bin/env python3
"""Stage selected official LongBench v1 JSONL files from the dataset archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from huggingface_hub import hf_hub_download

from representation_kv.longbench import DEFAULT_TASKS


def validate_task(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not {"context", "input", "answers", "all_classes", "length"} <= row.keys():
                raise ValueError(f"Malformed LongBench row in {path}")
            count += 1
    if count == 0:
        raise ValueError(f"Empty LongBench task: {path}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--revision", default="main", help="Hugging Face dataset commit or branch")
    parser.add_argument("--hf-cache", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.hf_cache).expanduser().resolve() if args.hf_cache else data_dir / ".hf-cache"
    with_archive = hf_hub_download(
        repo_id="zai-org/LongBench", filename="data.zip", repo_type="dataset",
        revision=args.revision, cache_dir=cache_dir,
    )
    archive = Path(with_archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    counts = {}
    with zipfile.ZipFile(archive) as source:
        for task in args.tasks:
            target = data_dir / f"{task}.jsonl"
            if not target.exists():
                member = f"data/{task}.jsonl"
                if member not in source.namelist():
                    raise FileNotFoundError(f"{member} is missing from {archive}")
                temporary = target.with_suffix(".jsonl.tmp")
                with source.open(member) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                validate_task(temporary)
                temporary.replace(target)
            counts[task] = validate_task(target)
            print(f"{task}: {counts[task]} rows at {target}", flush=True)
    manifest = {
        "source": "zai-org/LongBench", "archive": "data.zip",
        "requested_revision": args.revision, "archive_sha256": digest,
        "tasks": counts,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staging complete: {data_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
