import pytest

from dacn.cloud.config import parse_config_json


def test_parse_config_json_accepts_object():
    assert parse_config_json('{"epochs": 3, "seed": 17}') == {"epochs": 3, "seed": 17}


@pytest.mark.parametrize("value", ["{", "not-json"])
def test_parse_config_json_rejects_malformed_json(value):
    with pytest.raises(ValueError, match="config_json must be a JSON object"):
        parse_config_json(value)


@pytest.mark.parametrize("value", ["[]", "null", '"text"', "42"])
def test_parse_config_json_rejects_non_object_json(value):
    with pytest.raises(TypeError, match="config_json must be a JSON object"):
        parse_config_json(value)
