"""
tests/unit/test_rule_engine.py
-------------------------------
Unit tests for run_rule_engine() error paths in rule_engine.py.
Targets the three uncovered branches that bring coverage from 84% -> 90%+:

  - _load_config(): file missing      -> RuleEngineError(code="CONFIG_ERROR")
  - _load_config(): malformed YAML    -> RuleEngineError(code="CONFIG_ERROR")
  - run_rule_engine(): unexpected exc -> RuleEngineError(code="RULE_EVALUATION_FAILED")
"""

from __future__ import annotations

import pytest

from urolens_ai.smart_diagnosis import rule_engine
from urolens_ai.smart_diagnosis.rule_engine import run_rule_engine
from urolens_ai.smart_diagnosis.rules.gout import RuleResult
from urolens_ai.utils.exceptions import RuleEngineError


from pathlib import Path

class TestLoadConfigErrors:

    def test_missing_config_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(rule_engine, "_CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(RuleEngineError) as exc_info:
            run_rule_engine({"crystals": 5})
        assert exc_info.value.code == "CONFIG_ERROR"
        assert "not found" in exc_info.value.message.lower()

    def test_malformed_yaml_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad_yaml = tmp_path / "bad_config.yaml"
        bad_yaml.write_text("conditions: [\ninvalid: yaml: {{\n", encoding="utf-8")
        monkeypatch.setattr(rule_engine, "_CONFIG_PATH", str(bad_yaml))
        with pytest.raises(RuleEngineError) as exc_info:
            run_rule_engine({"crystals": 5})
        assert exc_info.value.code == "CONFIG_ERROR"
        assert "parse" in exc_info.value.message.lower()


class TestRuleEvaluationFailed:
    """Unexpected exceptions inside run_rule_engine() must be wrapped as RULE_EVALUATION_FAILED."""

    def test_unexpected_exception_wrapped_as_rule_evaluation_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Monkeypatch GoutRules.evaluate() to raise a bare RuntimeError.
        The outer except block must catch it and re-raise as RuleEngineError.
        """
        from urolens_ai.smart_diagnosis.rules.gout import GoutRules

        def boom(self: GoutRules, classification: dict[str, int]) -> RuleResult:
            raise RuntimeError("simulated rule crash")

        monkeypatch.setattr(GoutRules, "evaluate", boom)
        with pytest.raises(RuleEngineError) as exc_info:
            run_rule_engine({"crystals": 5})
        assert exc_info.value.code == "RULE_EVALUATION_FAILED"
        assert "simulated rule crash" in exc_info.value.message