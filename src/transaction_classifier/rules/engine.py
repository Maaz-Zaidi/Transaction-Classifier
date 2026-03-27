"""Rules engine for known-merchant classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from transaction_classifier.config import settings


@dataclass(frozen=True, slots=True)
class RuleMatch:
    category: str
    confidence: float
    rule_pattern: str


@dataclass(frozen=True, slots=True)
class _CompiledRule:
    pattern_str: str
    category: str
    rule_type: str  # "contains", "regex", "exact"
    priority: int
    compiled: re.Pattern | None  # None for contains/exact


def _compile_rule(raw: dict) -> _CompiledRule:
    rule_type = raw.get("type", "contains")
    compiled = None
    if rule_type == "regex":
        compiled = re.compile(raw["pattern"], re.IGNORECASE)
    return _CompiledRule(
        pattern_str=raw["pattern"],
        category=raw["category"],
        rule_type=rule_type,
        priority=raw.get("priority", 50),
        compiled=compiled,
    )


class RulesEngine:
    def __init__(self, rules_path: Path | None = None):
        rules_path = rules_path or (
            Path(__file__).parent / "rules.yaml"
        )
        with open(rules_path) as f:
            data = yaml.safe_load(f)

        raw_rules = data.get("rules", [])
        self._rules = sorted(
            [_compile_rule(r) for r in raw_rules],
            key=lambda r: r.priority,
            reverse=True,  # highest priority first
        )
        self._confidence = settings.rules_confidence

    def match(self, cleaned_text: str) -> RuleMatch | None:
        """Match a cleaned transaction string against rules.

        Returns RuleMatch on first hit (highest priority), or None.
        """
        upper = cleaned_text.upper()
        for rule in self._rules:
            if _matches(rule, upper):
                return RuleMatch(
                    category=rule.category,
                    confidence=self._confidence,
                    rule_pattern=rule.pattern_str,
                )
        return None

    def match_batch(self, texts: list[str]) -> list[RuleMatch | None]:
        """Match a batch of cleaned transaction strings."""
        return [self.match(t) for t in texts]


def _matches(rule: _CompiledRule, upper_text: str) -> bool:
    if rule.rule_type == "exact":
        return upper_text == rule.pattern_str.upper()
    if rule.rule_type == "contains":
        return rule.pattern_str.upper() in upper_text
    # regex
    return rule.compiled.search(upper_text) is not None
