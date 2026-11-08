import ast
import hashlib
import io
import json
import subprocess
import sys
import zipfile

import pytest


@pytest.mark.parametrize(
    "comment", [b"d5889d51a1a19213c6e533412b4582d645354c88", b"", b"not-a-commit"]
)
def test_generated_setup_does_not_claim_inherited_cpu_identity(tmp_path, comment):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.comment = comment
        archive.writestr("src/example.py", "# source fixture\n")
    partition = tmp_path / "partition.json"
    partition.write_text("{}")
    output = tmp_path / "notebook.ipynb"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_controls_colab.py",
            "--source",
            str(source),
            "--partition",
            str(partition),
            "--output",
            str(output),
        ],
        check=True,
    )
    notebook = json.loads(output.read_text())
    for cell in notebook["cells"][1:]:
        compile("".join(cell["source"]), "<generated-cell>", "exec")
    setup = ast.parse("".join(notebook["cells"][3]["source"]))
    start = next(
        i
        for i, node in enumerate(setup.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "manifest" for t in node.targets)
    )
    stop = next(i for i in range(start + 1, len(setup.body)) if isinstance(setup.body[i], ast.For))
    candidate = {
        "execution_source_commit": "old_cpu_commit",
        "environment_scope": "Windows CPU",
        "pending": ["old smoke"],
        "thresholds": {"pixel": 0.25},
        "artifacts": {"checkpoint": "pinned"},
    }
    namespace = {
        "candidate": candidate,
        "json": json,
        "zipfile": zipfile,
        "io": io,
        "SOURCE": source.read_bytes(),
        "hashlib": hashlib,
    }
    exec(  # noqa: S102 - execute only the generated manifest preflight, no runtime/data access.
        compile(
            ast.Module(body=setup.body[start:stop], type_ignores=[]),
            "<generated-manifest-preflight>",
            "exec",
        ),
        namespace,
    )
    manifest = namespace["manifest"]
    assert "execution_source_commit" not in manifest
    assert "pending" not in manifest
    assert manifest["source_archive_commit"] == (comment.decode() if len(comment) == 40 else None)
    assert "Colab" in manifest["environment_scope"]
    assert manifest["thresholds"] == candidate["thresholds"]
    assert manifest["artifacts"] == candidate["artifacts"]
    assert candidate["execution_source_commit"] == "old_cpu_commit"
