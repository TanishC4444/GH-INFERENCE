import io
import time
import zipfile

import requests

from .logger import _log
from .repo import API, _headers


def _download_output(token, username, repo_name, run_id, verbose, timeout=60):
    start  = time.time()
    target = None

    while time.time() - start < timeout:
        r         = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}/artifacts",
            headers=_headers(token),
        )
        artifacts = r.json().get("artifacts", [])
        target    = next((a for a in artifacts if a["name"] == "ai-output"), None)
        if target:
            break
        _log("Waiting for artifact...", verbose=verbose)
        time.sleep(5)

    if not target:
        raise RuntimeError("No ai-output artifact found after waiting.")

    r = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/artifacts/{target['id']}/zip",
        headers=_headers(token),
        allow_redirects=False,
    )
    location  = r.headers.get("Location", "")
    zip_bytes = requests.get(location).content if location else r.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        _log(f"Artifact contents: {z.namelist()}", verbose=verbose)
        name = next((n for n in z.namelist() if n.endswith("output.txt")), None)
        if not name:
            raise RuntimeError(f"output.txt not found in artifact. Contents: {z.namelist()}")
        with z.open(name) as f:
            return f.read().decode()