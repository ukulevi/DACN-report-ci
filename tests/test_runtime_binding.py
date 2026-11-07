import io
import zipfile

import pytest

from dacn.evaluation.runtime_binding import verify_runtime_inputs


def fixture(tmp_path):
    p = tmp_path / "src/example.py"
    p.parent.mkdir()
    p.write_bytes(b"original")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("src/example.py", b"original")
    return {"source_snapshot": buffer.getvalue(), "dependency_lock": b"numpy==1\r\n"}


def test_source_and_dependency_binding_accepts_crlf(tmp_path):
    captured = fixture(tmp_path)
    assert verify_runtime_inputs(captured, tmp_path, "numpy==1\n")["source_files_verified"] == 1


@pytest.mark.parametrize("change", ["source", "extra", "dependency"])
def test_runtime_changes_rejected(tmp_path, change):
    captured = fixture(tmp_path)
    freeze = "numpy==1\n"
    if change == "source":
        (tmp_path / "src/example.py").write_bytes(b"changed")
    elif change == "extra":
        (tmp_path / "src/extra.py").write_bytes(b"extra")
    else:
        freeze = "numpy==2\n"
    with pytest.raises(ValueError):
        verify_runtime_inputs(captured, tmp_path, freeze)
