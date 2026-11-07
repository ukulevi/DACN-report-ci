import pytest

from dacn.evaluation.dependency_snapshot import version_snapshot


def test_direct_urls_are_replaced_with_installed_versions(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "1.2.3")
    result = version_snapshot("My_Pkg @ https://user:secret@example.test/archive\nNumPy==2\r\n")
    assert result == "my-pkg==1.2.3\nnumpy==2\n"
    assert "secret" not in result


def test_format_and_order_do_not_change_inventory():
    assert version_snapshot("NumPy==2\r\nPillow==10\r\n") == version_snapshot(
        "pillow==10\nnumpy==2\n"
    )


@pytest.mark.parametrize("value", ["broken", "thing==https://secret", "a==1\na==2"])
def test_unsupported_or_conflicting_versions_rejected(value):
    with pytest.raises(ValueError):
        version_snapshot(value)
