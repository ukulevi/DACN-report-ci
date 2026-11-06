import hashlib

import pytest

from dacn.training.checkpoint_artifacts import capture_continuation, write_continuation


def parent(tmp_path):
    files = {name: name.encode() for name in ("ddpm_last.pt", "ddpm_best.pt")}
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    return files, {
        "artifacts": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    }


@pytest.mark.parametrize("case", ["missing_file", "missing_hash", "changed_bytes"])
def test_capture_rejects_unverified_parent(tmp_path, case):
    _, manifest = parent(tmp_path)
    if case == "missing_file":
        (tmp_path / "ddpm_best.pt").unlink()
    elif case == "missing_hash":
        del manifest["artifacts"]["ddpm_best.pt"]
    else:
        (tmp_path / "ddpm_best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        capture_continuation(tmp_path, manifest)


def test_child_uses_captured_bytes_even_after_parent_changes(tmp_path):
    files, manifest = parent(tmp_path)
    captured = capture_continuation(tmp_path, manifest)
    for name in files:
        (tmp_path / name).write_bytes(b"later replacement")
    child = tmp_path / "child"
    child.mkdir()
    write_continuation(child, captured)
    assert {name: (child / name).read_bytes() for name in files} == files


def test_interrupted_write_can_restart_without_loading_temporary_file(tmp_path, monkeypatch):
    from dacn.training import checkpoint_artifacts as module

    files, manifest = parent(tmp_path)
    captured = capture_continuation(tmp_path, manifest)
    child = tmp_path / "child"
    child.mkdir()
    replace = module.os.replace

    def interrupt(*args):
        raise OSError("synthetic interruption")

    monkeypatch.setattr(module.os, "replace", interrupt)
    with pytest.raises(OSError):
        write_continuation(child, captured)
    assert not (child / "ddpm_last.pt").exists()
    monkeypatch.setattr(module.os, "replace", replace)
    write_continuation(child, captured)
    assert (child / "ddpm_last.pt").read_bytes() == files["ddpm_last.pt"]
