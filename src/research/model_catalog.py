"""Claude model list for the Settings page's model dropdown (2026-10-13, ported from
AITrading; owner report there: "the program doesnt get the new model options, as they have changed").

The Research Model dropdown used to be a hardcoded 3-option list in settings.html,
so every Anthropic model release needed a code edit + release before it could even
be selected. The list now comes from Anthropic's own Models API
(`client.models.list()` -- free, not billed), cached in data/available_models.json
and refreshed at most once per `MAX_AGE_HOURS` when the Settings page is opened.

Pure helpers only -- web/app.py owns the fetch timing and the cache file (atomic
temp-file + os.replace write). Purely a UI choice list: nothing here changes which model any
dial is actually set to. The dial's current value and its default are always kept
selectable even when the API no longer lists them, so a Settings save can never
silently switch a model."""
from datetime import datetime, timezone

MAX_AGE_HOURS = 24
PER_FAMILY_LIMIT = 3  # newest N per family; older ones only appear as a dial's current/default

FAMILY_ORDER = ("haiku", "sonnet", "opus", "fable")
TIER_HINT = {
    "haiku": "Fastest / Cheapest",
    "sonnet": "Balanced",
    "opus": "Most Powerful",
    "fable": "Flagship / Most Capable",
}

# Used only when no cache exists yet AND the live fetch fails (no key, offline).
# Newest first within each family, same shape as fetch_available_models() output.
FALLBACK_MODELS = [
    {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": ""},
    {"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5", "created_at": ""},
    {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5", "created_at": ""},
    {"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "created_at": ""},
    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": ""},
    {"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8", "created_at": ""},
    {"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1", "created_at": ""},
    {"id": "claude-fable-5", "display_name": "Claude Fable 5", "created_at": ""},
]


def model_family(model_id: str) -> str | None:
    """'haiku'/'sonnet'/'opus'/'fable' from a model id, or None for anything else."""
    mid = (model_id or "").lower()
    for fam in FAMILY_ORDER:
        if fam in mid:
            return fam
    return None


def fetch_available_models(client) -> list[dict]:
    """Blocking Models API call (run via asyncio.to_thread). Raises on failure --
    the caller decides to keep the old cache. The SDK auto-paginates list()."""
    out = []
    for m in client.models.list():
        created = getattr(m, "created_at", None)
        out.append({
            "id": m.id,
            "display_name": getattr(m, "display_name", "") or m.id,
            "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created or ""),
        })
    return out


def catalog_is_stale(fetched_at: str | None, now: datetime | None = None,
                     max_age_hours: float = MAX_AGE_HOURS) -> bool:
    """True when the cache is missing, unparseable, or older than max_age_hours."""
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at)
    except (TypeError, ValueError):
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - ts).total_seconds() > max_age_hours * 3600


def _label(model: dict) -> str:
    name = (model.get("display_name") or model["id"]).removeprefix("Claude ").strip()
    hint = TIER_HINT.get(model_family(model["id"]) or "")
    return f"{name} — {hint}" if hint else name


def build_model_options(models: list[dict], current: str | None,
                        default: str | None,
                        per_family: int = PER_FAMILY_LIMIT) -> list[dict]:
    """Dropdown options [{id, label}] for one dial: the newest `per_family` models
    of each known family (cheapest family first), plus the dial's current value and
    default if they weren't already included. Models outside the four families
    (e.g. access-program-only models) are left out unless they are the current
    value. Ordering within a family is newest first by created_at."""
    by_family: dict[str, list[dict]] = {f: [] for f in FAMILY_ORDER}
    for m in models:
        fam = model_family(m.get("id", ""))
        if fam:
            by_family[fam].append(m)

    options: list[dict] = []
    seen: set[str] = set()
    for fam in FAMILY_ORDER:
        # Stable sort keeps the API's/fallback's own order for equal (blank) dates.
        ranked = sorted(by_family[fam], key=lambda m: m.get("created_at") or "", reverse=True)
        for m in ranked[:per_family]:
            if m["id"] not in seen:
                seen.add(m["id"])
                options.append({"id": m["id"], "label": _label(m)})

    known = {m["id"]: m for m in FALLBACK_MODELS} | {m["id"]: m for m in models}
    for extra in (current, default):
        if extra and extra not in seen:
            seen.add(extra)
            m = known.get(extra, {"id": extra, "display_name": extra})
            options.append({"id": extra, "label": _label(m)})
    return options
