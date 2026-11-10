"""Install checksum-pinned official report compilers into a local tools directory."""

import argparse
import hashlib
import io
import json
import os
import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RELEASES = {
    "Windows": {
        "pandoc": (
            "https://github.com/jgm/pandoc/releases/download/3.13/pandoc-3.13-windows-x86_64.zip",
            "23e1f2e978a3df80db8bf65c9c59534f41a23140480cfb4f783a882ab7dc6a4e",
        ),
        "tectonic": (
            "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip",
            "f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f",
        ),
    },
    "Linux": {
        "pandoc": (
            "https://github.com/jgm/pandoc/releases/download/3.13/pandoc-3.13-linux-amd64.tar.gz",
            "38aa33b325a2bc96a4c39409631212db55702f564250bcdbead180b3bf646389",
        ),
        "tectonic": (
            "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz",
            "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7",
        ),
    },
}


def extract_binary(archive, expected, name, output):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Compiler archive checksum differs")
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as stream:
            matches = [p for p in stream.namelist() if p.split("/")[-1] == name]
            if len(matches) != 1:
                raise ValueError("Compiler archive must contain exactly one matching binary")
            binary = stream.read(matches[0])
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as stream:
            matches = [
                p for p in stream.getmembers() if p.isfile() and p.name.split("/")[-1] == name
            ]
            if len(matches) != 1:
                raise ValueError("Compiler archive must contain exactly one matching binary")
            with stream.extractfile(matches[0]) as handle:
                binary = handle.read()
    # Only write the selected binary; never extract archive paths or symlinks.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    if os.name != "nt":
        output.chmod(0o755)
    return hashlib.sha256(binary).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/report-tools"))
    args = parser.parse_args()
    system = platform.system()
    if system not in RELEASES or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("Portable report tools support Windows/Linux x86_64")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"versions": {"pandoc": "3.13", "tectonic": "0.17.0"}, "archives": {}}
    for tool, (url, digest) in RELEASES[system].items():
        archive = args.output / (tool + (".zip" if system == "Windows" else ".tar.gz"))
        if not archive.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "DACN-report-build"})
            with urllib.request.urlopen(request, timeout=120) as response:
                archive.write_bytes(response.read())
        name = tool + (".exe" if system == "Windows" else "")
        executable = args.output / name
        sha = extract_binary(archive, digest, name, executable)
        receipt[tool] = str(executable.resolve())
        receipt["archives"][tool] = {"url": url, "sha256": digest, "binary_sha256": sha}
    (args.output / "tools.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Pinned Pandoc3.13/Tectonic0.17.0 ready; archives verified.")


if __name__ == "__main__":
    main()
