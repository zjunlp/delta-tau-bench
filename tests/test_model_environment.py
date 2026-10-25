"""Model configuration can use credentials without writing them to YAML."""
import os
from unittest.mock import patch

import pytest

from dtau_bench.utils.config import load_models_config


def test_environment_values_are_not_reparsed_as_yaml_or_shell(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text('models:\n  demo:\n    api_key: "${TEST_MODEL_KEY}"\n    max_tokens: 32\n')
    value = "quotes: ' \"\n$(echo never-execute) ${UNCHANGED}"
    with patch.dict(os.environ, {"TEST_MODEL_KEY": value}):
        loaded = load_models_config(config)
    assert loaded["models"]["demo"]["api_key"] == value
    assert loaded["models"]["demo"]["max_tokens"] == 32


def test_defaults_and_missing_variables(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text('embedding:\n  urls: ["${UNSET_MODEL_URL:-http://localhost:8009/v1}"]\n')
    with patch.dict(os.environ, {}, clear=True):
        assert load_models_config(config)["embedding"]["urls"] == ["http://localhost:8009/v1"]
        config.write_text('models:\n  demo:\n    api_key: "${REQUIRED_MODEL_KEY}"\n')
        with pytest.raises(ValueError, match="REQUIRED_MODEL_KEY"):
            load_models_config(config)
