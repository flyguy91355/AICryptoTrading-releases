"""Configuration loader."""

import re
from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/settings.yaml") -> dict:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f)


_SECTION_RE = re.compile(r"^([A-Za-z_][\w]*):\s*(#.*)?$")
_KEYVAL_RE = re.compile(r"^(\s+)([A-Za-z_][\w]*):(\s*)([^\s#]*)(.*)$")


def _format_yaml_scalar(value) -> str:
    """Formats a Python value to match this file's existing plain-YAML style
    (lowercase unquoted booleans, unquoted numbers, unquoted simple strings)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if re.fullmatch(r"[\w.\-/]+", text):
        return text
    return yaml.safe_dump(text).strip()


def update_settings_yaml(config_path: str, updates: dict) -> None:
    """Rewrites only the specific "section.key" values in `updates` (a flat dict,
    e.g. {"research.min_conviction_score": 6.5}), leaving every comment, blank line,
    and untouched field exactly as-is. A plain yaml.safe_dump of the whole config
    would silently discard every explanatory comment in this file -- the same
    whole-file-rewrite risk AITrading's own settings.yaml carries a standing rule
    against (see that project's feedback-settings-yaml-live-drift memory) -- so this
    does a targeted line-level edit instead, same principle applied to this
    project's own (much smaller, single-level-nested) settings file."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    remaining = dict(updates)
    current_section = None
    for i, line in enumerate(lines):
        section_match = _SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if current_section is None or not remaining:
            continue
        kv_match = _KEYVAL_RE.match(line)
        if not kv_match:
            continue
        indent, key, spacing, _old_value, rest = kv_match.groups()
        dotted = f"{current_section}.{key}"
        if dotted in remaining:
            new_value = _format_yaml_scalar(remaining.pop(dotted))
            lines[i] = f"{indent}{key}:{spacing}{new_value}{rest}\n"

    if remaining:
        raise KeyError(f"update_settings_yaml: field(s) not found in {config_path}: {sorted(remaining)}")

    path.write_text("".join(lines), encoding="utf-8")
