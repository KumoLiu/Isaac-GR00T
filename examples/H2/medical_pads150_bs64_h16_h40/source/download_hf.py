"""Download and verify one pinned Hugging Face dataset subfolder."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("lfs"):
        digest = hashlib.sha256()
        expected = entry["lfs"]["oid"]
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {entry['size']}\0".encode())
        expected = entry["oid"]
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    token = args.token_file.read_text().strip()
    args.output.mkdir(parents=True, exist_ok=True)

    def download(entry):
        relative = Path(entry["path"]).relative_to(manifest["prefix"])
        if ".." in relative.parts:
            raise ValueError("Invalid repository path")
        destination = args.output / relative
        if verified(destination, entry):
            return {"path": str(relative), "size": entry["size"], "verified": True}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        url = (
            f"https://huggingface.co/datasets/{manifest['repo']}/resolve/"
            f"{manifest['revision']}/{urllib.parse.quote(entry['path'])}"
        )
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
                opener = urllib.request.build_opener(SafeRedirect())
                with opener.open(request, timeout=120) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                if not verified(temporary, entry):
                    raise ValueError("Downloaded content does not match repository checksum")
                temporary.replace(destination)
                return {"path": str(relative), "size": entry["size"], "verified": True}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download, entry) for entry in manifest["files"]]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 25 == 0:
                print(
                    json.dumps({"downloaded_files": len(records), "total_files": len(futures)}),
                    flush=True,
                )
    result = {
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "prefix": manifest["prefix"],
        "files": len(records),
        "bytes": sum(entry["size"] for entry in records),
        "all_checksums_verified": True,
        "completed_unix": time.time(),
        "dataset_root": str(args.output),
    }
    record = args.output / "DOWNLOAD_VERIFIED.json"
    temporary = record.with_name(record.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(record)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
