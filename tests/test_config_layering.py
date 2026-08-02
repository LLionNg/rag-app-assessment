from pathlib import Path

import pytest

from src.core.config import load_settings
from src.core.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_extends_overrides_only_the_keys_it_names():
    base = load_settings(ROOT / "config.yml")
    docker = load_settings(ROOT / "config.docker.yml")

    assert docker.ui.host == "0.0.0.0"
    assert docker.retrieval.strategy == "keyword"
    assert docker.embeddings.provider is None
    # Untouched keys come through from the base file.
    assert docker.llm.provider == base.llm.provider
    assert (
        docker.knowledge_base.chunking.max_chars
        == base.knowledge_base.chunking.max_chars
    )


def test_extends_chains_through_two_levels():
    semantic = load_settings(ROOT / "config.docker.semantic.yml")

    assert semantic.retrieval.strategy == "semantic"
    assert semantic.embeddings.provider == "bge_m3"
    # Still inherits the container binding from the file in the middle.
    assert semantic.ui.host == "0.0.0.0"


def test_circular_extends_is_rejected(tmp_path: Path):
    (tmp_path / "a.yml").write_text("extends: b.yml\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("extends: a.yml\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Circular extends"):
        load_settings(tmp_path / "a.yml")


def test_missing_base_is_reported(tmp_path: Path):
    (tmp_path / "a.yml").write_text("extends: nope.yml\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "a.yml")
