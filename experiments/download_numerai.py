"""Download the exact public Numerai v5.3 inputs used by these experiments."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((Path(__file__).with_name("numerai_v5_3_manifest.json")).read_text())
API = "https://api-tournament.numer.ai"


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected["sha256"]


def download_url(session, filename):
    response = session.post(API, json={
        "query": "query($filename: String!) { dataset(filename: $filename, tournament: 8) }",
        "variables": {"filename": f"v5.3/{filename}"},
    }, timeout=60)
    response.raise_for_status()
    payload = response.json()
    url = (payload.get("data") or {}).get("dataset")
    if payload.get("errors") or not isinstance(url, str) or not url.startswith("https://"):
        raise RuntimeError(f"Numerai did not return a download URL for {filename}")
    return url


def download_file(session, filename, destination, expected):
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size == expected["bytes"]:
        if verify_file(partial, expected):
            partial.replace(destination)
            return
        raise RuntimeError(f"Invalid completed partial file: {partial}; move it aside before retrying")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected["bytes"]:
        raise RuntimeError(f"Oversized partial file: {partial}; move it aside before retrying")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = download_url(session, filename)
    try:
        with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            if response.status_code not in (200, 206):
                raise RuntimeError(f"Download failed for {filename}: HTTP {response.status_code}")
            if response.status_code == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Unexpected range response for {filename}")
                mode = "ab"
            else:
                # Some servers ignore Range; restart this partial download safely.
                offset, mode = 0, "wb"
            written, last_report = offset, offset
            with partial.open(mode) as handle:
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    written += len(block)
                    if written - last_report >= 256 * 1024 * 1024:
                        print(f"{filename}: {written / expected['bytes']:.1%}", flush=True)
                        last_report = written
    except requests.RequestException as error:
        # Keep .part for resumption and never print temporary signed URLs.
        raise RuntimeError(f"Network interruption for {filename} ({type(error).__name__}); rerun to resume") from None
    if not verify_file(partial, expected):
        raise RuntimeError(f"Size or SHA256 mismatch for {partial}; final file was not installed")
    partial.replace(destination)


def run(output_dir, filenames, verify_only=False):
    output_dir = Path(output_dir).resolve()
    if verify_only and not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = 0
    pending = []
    for filename in filenames:
        expected = MANIFEST["files"][filename]
        destination = output_dir / filename
        if destination.exists():
            if not verify_file(destination, expected):
                raise RuntimeError(f"Existing file failed verification: {destination}; move it aside before downloading again")
            print(f"Verified: {filename}", flush=True)
        elif verify_only:
            raise FileNotFoundError(destination)
        else:
            partial = destination.with_name(destination.name + ".part")
            needed += max(0, expected["bytes"] - (partial.stat().st_size if partial.exists() else 0))
            pending.append(filename)
    if pending and shutil.disk_usage(output_dir).free < needed + 512 * 1024 * 1024:
        raise RuntimeError(f"Not enough disk space: downloads need about {needed / 2**30:.2f} GiB plus reserve")
    with requests.Session() as session:
        for filename in pending:
            print(f"Downloading: v5.3/{filename}", flush=True)
            download_file(session, filename, output_dir / filename, MANIFEST["files"][filename])
            print(f"Verified: {filename}", flush=True)
    print(f"Inputs ready: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "downloads/numerai/v5.3")
    parser.add_argument("--files", nargs="+", choices=list(MANIFEST["files"]),
                        default=list(MANIFEST["files"]))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-api", action="store_true",
                        help="Check public download URL availability without downloading files")
    args = parser.parse_args()
    if args.check_api:
        with requests.Session() as session:
            for filename in args.files:
                download_url(session, filename)
                print(f"Public download URL available: v5.3/{filename}", flush=True)
    else:
        run(args.output_dir, args.files, args.verify_only)
