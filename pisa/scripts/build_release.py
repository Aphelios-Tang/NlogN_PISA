"""Build the PISA wheel and source distribution."""

import gzip
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def normalize_ownership(path):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tar.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with (
            tarfile.open(path, "r:gz") as source,
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as target,
        ):
            for member in source:
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                for field in ("uid", "gid", "uname", "gname"):
                    member.pax_headers.pop(field, None)
                content = source.extractfile(member) if member.isfile() else None
                target.addfile(member, content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    package = Path(__file__).resolve().parents[1]
    output = package / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation",
         "--outdir", str(output), str(package)],
        check=True,
    )
    for path in output.glob("*.tar.gz"):
        normalize_ownership(path)



if __name__ == "__main__":
    main()
