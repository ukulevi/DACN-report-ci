import json
import re
from pathlib import Path


def test_anoddpm_colab_notebook_is_clean_and_compilable():
    path = Path(__file__).parents[1] / "notebooks/anoddpm_bmad_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"

    sources = []
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        source = "".join(cell["source"])
        sources.append(source)
        if cell["cell_type"] == "code":
            compile(source, f"notebook-cell-{index}", "exec")

    complete_source = "\n".join(sources)
    assert "feat/anoddpm-notebook-training" in complete_source
    assert "fbb884dfe144be5c71e205a73701507163557ea4" in complete_source
    assert 'max_train_samples": 128' in complete_source
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", complete_source)
