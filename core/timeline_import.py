"""Load a mission timeline from .xlsx or .csv into one internal shape.

The author works in Excel, which is the point: a timeline is a table of
moments, and a spreadsheet is the right tool for that. Both formats
normalize to the same rows so nothing downstream knows or cares which was
used.

VALIDATION FAILS LOUDLY, BEFORE THE EXERCISE STARTS. A timeline that
loaded with a silently-dropped row would run an exercise missing a beat
the author believed was there, and the trainee would be assessed on it.
Every error names the sheet row so a typo takes seconds to find.

Only `openpyxl` is required, and it was already a dependency.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from core.timeline import EventType, Priority, ReportingPolicy, Timeline, TimelineEvent

COLUMNS = (
    "event_id", "start_time", "end_time", "event_type", "description",
    "operator_information", "state_updates", "tags", "entity_ids",
    "reporting_policy", "priority", "instructions", "expires_at",
)

REQUIRED = ("event_id", "start_time")


class TimelineError(Exception):
    """A timeline file could not be loaded. Names the row."""

    def __init__(self, message: str, row: int | None = None,
                 source: str | Path | None = None) -> None:
        """Carry the sheet row number, so an import error points at the row to fix."""
        where = f" (row {row})" if row else ""
        origin = f" in {Path(source).name}" if source else ""
        super().__init__(f"timeline{origin}{where}: {message}")
        self.row = row


def load_timeline(path: str | Path) -> Timeline:
    """Load and validate a timeline from .xlsx or .csv."""
    path = Path(path)
    if not path.exists():
        raise TimelineError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        rows = _read_xlsx(path)
    elif suffix in {".csv", ".tsv"}:
        rows = _read_csv(path)
    else:
        raise TimelineError(
            f"unsupported format {suffix!r}; use .xlsx or .csv", source=path
        )

    events = [_to_event(row, index, path) for index, row in rows]
    _check_cross_row(events, path)
    return Timeline(events)


# -- readers ---------------------------------------------------------------


def _read_xlsx(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read an Excel sheet into (row number, row) pairs."""
    try:
        from openpyxl import load_workbook
    except ImportError as err:                      # pragma: no cover
        raise TimelineError(
            "openpyxl is required to read .xlsx; install it or export to CSV"
        ) from err

    # data_only so a formula cell yields its computed value rather than
    # "=A1*60", which an author would reasonably expect to work.
    workbook = load_workbook(path, data_only=True, read_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()

    if not rows:
        raise TimelineError("file is empty", source=path)

    header = [_norm_header(c) for c in rows[0]]
    _check_header(header, path)

    out: list[tuple[int, dict[str, Any]]] = []
    for offset, raw in enumerate(rows[1:], start=2):
        if _is_not_data(header, raw):
            continue
        out.append((offset, dict(zip(header, raw))))
    return out


def _read_csv(path: Path) -> list[tuple[int, dict[str, Any]]]:
    # utf-8-sig: Excel writes a BOM, which would otherwise corrupt the
    # first header name and produce a baffling "missing event_id".
    """Read a CSV or TSV into (row number, row) pairs."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [r for r in reader]

    if not rows:
        raise TimelineError("file is empty", source=path)

    header = [_norm_header(c) for c in rows[0]]
    _check_header(header, path)

    out: list[tuple[int, dict[str, Any]]] = []
    for offset, raw in enumerate(rows[1:], start=2):
        if _is_not_data(header, raw):
            continue
        out.append((offset, dict(zip(header, raw))))
    return out


def _is_not_data(header: list[str], raw: Any) -> bool:
    """Whether a sheet row should be skipped rather than parsed.

    Skips fully blank spacer rows, and rows carrying only a note in the
    first column -- the generated template ends with such a footnote, and
    the round-trip check caught it being read as an event with an empty
    start_time. An author adding their own notes below the data would hit
    the same thing.
    """
    cells = list(raw)
    values = [("" if c is None else str(c).strip()) for c in cells]
    if not any(values):
        return True

    # Only the first column filled, and no start_time: a note, not a row.
    try:
        start_index = header.index("start_time")
    except ValueError:                               # pragma: no cover
        return False
    has_start = start_index < len(values) and bool(values[start_index])
    filled = sum(1 for v in values if v)
    return filled <= 1 and not has_start


def _norm_header(cell: Any) -> str:
    """Lowercase and strip a header cell so column names tolerate stray spacing and case."""
    return re.sub(r"[\s-]+", "_", str(cell or "").strip().lower())


def _check_header(header: list[str], path: Path) -> None:
    """Reject a missing required column or an unknown one, naming both."""
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise TimelineError(
            f"missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(COLUMNS)}",
            row=1, source=path,
        )
    unknown = [c for c in header if c and c not in COLUMNS]
    if unknown:
        raise TimelineError(
            f"unknown column(s): {', '.join(unknown)}. "
            f"Known columns: {', '.join(COLUMNS)}",
            row=1, source=path,
        )


# -- row -> event ----------------------------------------------------------


def _to_event(row: dict[str, Any], index: int, path: Path) -> TimelineEvent:
    """Turn one sheet row into a validated TimelineEvent."""
    event_id = _text(row.get("event_id"))
    if not event_id:
        raise TimelineError("event_id is empty", row=index, source=path)

    start = _seconds(row.get("start_time"), "start_time", index, path)
    end = (_seconds(row.get("end_time"), "end_time", index, path)
           if _text(row.get("end_time")) else None)
    expires = (_seconds(row.get("expires_at"), "expires_at", index, path)
               if _text(row.get("expires_at")) else None)

    event_type = _enum(row.get("event_type"), EventType, EventType.POINT,
                       "event_type", index, path)
    policy = _enum(row.get("reporting_policy"), ReportingPolicy,
                   ReportingPolicy.ON_REQUEST, "reporting_policy", index, path)
    priority = _enum(row.get("priority"), Priority, Priority.NORMAL,
                     "priority", index, path)

    if event_type is EventType.INTERVAL and end is None:
        raise TimelineError(
            "event_type 'interval' requires end_time", row=index, source=path
        )
    if end is not None and end <= start:
        raise TimelineError(
            f"end_time ({end}) must be after start_time ({start})",
            row=index, source=path,
        )

    return TimelineEvent(
        event_id=event_id,
        start_time=start,
        end_time=end,
        event_type=event_type,
        description=_text(row.get("description")),
        operator_information=_text(row.get("operator_information")),
        state_updates=_json_cell(row.get("state_updates"), index, path),
        tags=_list_cell(row.get("tags")),
        entity_ids=_list_cell(row.get("entity_ids")),
        reporting_policy=policy,
        priority=priority,
        instructions=_text(row.get("instructions")),
        expires_at=expires,
    )


def _check_cross_row(events: list[TimelineEvent], path: Path) -> None:
    """Checks spanning rows: duplicate ids, and a reportable event with nothing to report."""
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            raise TimelineError(
                f"duplicate event_id {event.event_id!r}", source=path
            )
        seen.add(event.event_id)

    # A required or subscription-eligible event with nothing the operator
    # may say is almost always an unfinished row: the author wrote the
    # description and forgot operator_information.
    for event in events:
        if (event.reporting_policy is not ReportingPolicy.ON_REQUEST
                and event.event_type is not EventType.HANDOVER
                and not event.operator_information):
            raise TimelineError(
                f"event {event.event_id!r} has reporting_policy "
                f"'{event.reporting_policy.value}' but no operator_information, "
                f"so there would be nothing to report",
                source=path,
            )


# -- cell parsing ----------------------------------------------------------


def _text(cell: Any) -> str:
    """A cell as clean text, with None and numeric cells handled."""
    if cell is None:
        return ""
    return str(cell).strip()


def _seconds(cell: Any, field: str, index: int, path: Path) -> float:
    """Accept HH:MM:SS, MM:SS, or plain seconds.

    Excel turns a typed "00:04:10" into a time/datetime object, so those
    are handled too -- otherwise the most natural way to write a timestamp
    in a spreadsheet would be the one that fails.
    """
    if cell is None or str(cell).strip() == "":
        raise TimelineError(f"{field} is empty", row=index, source=path)

    # Excel native time/datetime/timedelta
    for attribute in ("hour", "total_seconds"):
        if hasattr(cell, attribute):
            if attribute == "total_seconds":
                return float(cell.total_seconds())
            return float(cell.hour * 3600 + cell.minute * 60 + cell.second)

    text = str(cell).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3 or not all(p.strip().isdigit() for p in parts if p.strip()):
            raise TimelineError(
                f"{field} {text!r} is not HH:MM:SS or MM:SS", row=index, source=path
            )
        numbers = [int(p) for p in parts]
        while len(numbers) < 3:
            numbers.insert(0, 0)
        return float(numbers[0] * 3600 + numbers[1] * 60 + numbers[2])

    try:
        value = float(text)
    except ValueError as err:
        raise TimelineError(
            f"{field} {text!r} is not a number or HH:MM:SS", row=index, source=path
        ) from err
    if value < 0:
        raise TimelineError(f"{field} cannot be negative", row=index, source=path)
    return value


def _enum(cell: Any, enum_type: type, default: Any, field: str,
          index: int, path: Path) -> Any:
    """Parse an enum cell, accepting the author-friendly spellings people type."""
    text = _text(cell).lower().replace(" ", "_")
    if not text:
        return default
    # Tolerate the author-friendly spellings a person actually types.
    aliases = {
        "point_event": "point", "persistent_update": "persistent",
        "crew_handover": "handover", "proactive": "required",
        "required_report": "required", "on_demand": "on_request",
    }
    text = aliases.get(text, text)
    try:
        return enum_type(text)
    except ValueError as err:
        allowed = ", ".join(m.value for m in enum_type)
        raise TimelineError(
            f"{field} {text!r} is not one of: {allowed}", row=index, source=path
        ) from err


def _list_cell(cell: Any) -> tuple[str, ...]:
    """Split a tags or entity_ids cell on commas or semicolons."""
    text = _text(cell)
    if not text:
        return ()
    # Comma or semicolon, since both are natural in a spreadsheet cell.
    parts = re.split(r"[;,]", text)
    return tuple(p.strip() for p in parts if p.strip())


def _json_cell(cell: Any, index: int, path: Path) -> dict[str, Any]:
    """Parse state_updates.

    Accepts JSON (`{"vehicle_count": 3}`) or the simpler `key=value`
    pairs an author is more likely to type by hand.
    """
    text = _text(cell)
    if not text:
        return {}

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            raise TimelineError(
                f"state_updates is not valid JSON: {err.msg}. "
                f'Example: {{"vehicle_count": 3}}',
                row=index, source=path,
            ) from err
        if not isinstance(parsed, dict):
            raise TimelineError(
                "state_updates must be a JSON object", row=index, source=path
            )
        return parsed

    updates: dict[str, Any] = {}
    for pair in re.split(r"[;,]", text):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise TimelineError(
                f"state_updates {pair.strip()!r} is not key=value or JSON",
                row=index, source=path,
            )
        key, _, raw = pair.partition("=")
        updates[key.strip()] = _coerce_scalar(raw.strip())
    return updates


def _coerce_scalar(text: str) -> Any:
    """Turn a key=value string into a bool, int, float or string."""
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
