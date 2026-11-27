"""Verify and restore the four original repositories to a new empty folder."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def restore(destination):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    archive = root / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch")
    destination = Path(destination).absolute()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Choose a new or empty destination directory")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Archive contains an unsafe path or unsupported file type")
        destination.mkdir(parents=True, exist_ok=True)
        stream.extractall(destination, members=members)
    for entry in manifest["files"]:
        path = destination / entry["path"]
        if path.stat().st_size != entry["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Restored file checksum mismatch: " + entry["path"])
    print("Restored and verified %d files in %s" % (len(manifest["files"]), destination))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 restore.py NEW_EMPTY_DESTINATION")
    restore(sys.argv[1])
