"""Generate an Excel timeline template.

    python timelines/make_template.py [output.xlsx]

Produces a formatted .xlsx with the column headers, inline notes on each
one, and the synthetic example rows to edit over. Excel is where a
timeline actually gets authored -- it is a table of moments, and a
spreadsheet is the right tool.

The importer (core/timeline_import.py) also reads CSV, which is what the
repository keeps checked in so examples stay diffable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent

# (column, width, note shown on hover)
COLUMNS: list[tuple[str, int, str]] = [
    ("event_id", 16,
     "Required. Stable unique id. Referenced in logs and the debrief, so "
     "avoid renaming once an exercise has been run."),
    ("start_time", 12,
     "Required. Time from the START OF THE RECORDING. HH:MM:SS, MM:SS or "
     "plain seconds. An Excel time cell works too."),
    ("end_time", 12,
     "Interval and handover only. When the window closes. Required for "
     "event_type=interval."),
    ("event_type", 13,
     "point | persistent | interval | handover\n\n"
     "point      an occurrence; never 'current' afterwards\n"
     "persistent true until a later event supersedes it\n"
     "interval   current only inside [start, end)\n"
     "handover   crew rotation; blocks ordinary dialogue"),
    ("description", 42,
     "AUTHOR-FACING ONLY. Never reaches the model. Put intent, the "
     "solution, and what the trainee should notice here."),
    ("operator_information", 42,
     "What the operator may know once this is revealed. THE ONLY COLUMN "
     "THE MODEL SEES. Write it as the operator would say it."),
    ("state_updates", 26,
     'Optional. Facts this event changes.\n\n'
     'key=value pairs:  ראות=ירודה, רכבים=2\n'
     'or JSON:          {"ראות": "ירודה"}'),
    ("tags", 20,
     "Optional. Comma-separated categories for reporting agreements: "
     "vehicle, arrival, person. 'Tell me about every vehicle' matches on "
     "these."),
    ("entity_ids", 16,
     "Optional. Ids for a SPECIFIC thing: veh_1. Lets 'only that vehicle' "
     "narrow an agreement."),
    ("reporting_policy", 17,
     "required     | volunteered as soon as it is known\n"
     "on_request   | only if asked\n"
     "subscription | only if a dialogue agreement covers it"),
    ("priority", 11,
     "normal | high | urgent\n\n"
     "urgent may interrupt trainee speech and can come through a crew "
     "handover. Use it sparingly."),
    ("instructions", 32,
     "Optional. In-character guidance for this moment: 'this is "
     "significant, report at once'. Shapes delivery, not content."),
    ("expires_at", 12,
     "Optional. After this time an undelivered report is stale and is "
     "dropped rather than announced as news."),
]

EXAMPLES: list[list[str]] = [
    ["veh_1", "00:03:00", "", "point",
     "SYNTHETIC. First vehicle arrives and stops.",
     "רכב מגיע מכיוון מערב ועוצר ליד המבנה",
     "", "vehicle,arrival", "veh_1", "required", "normal", "", ""],
    ["veh_2", "00:08:00", "", "point",
     "SYNTHETIC. Second vehicle. Only reported if the trainee asked for every vehicle.",
     "רכב שני מגיע ועוצר ליד הראשון",
     "", "vehicle,arrival", "veh_2", "subscription", "normal", "", ""],
    ["vis_drop", "00:01:30", "00:06:00", "interval",
     "SYNTHETIC. Visibility dips, then recovers.",
     "הראות ירדה, פחות חדה מקודם",
     "ראות=ירודה", "conditions", "", "required", "normal", "", ""],
    ["handover", "00:14:00", "00:16:00", "handover",
     "SYNTHETIC. Crew rotation; ordinary dialogue unavailable.",
     "", "", "", "", "on_request", "normal", "", ""],
]

HEADER_FILL = PatternFill("solid", fgColor="1F2937")
EXAMPLE_FILL = PatternFill("solid", fgColor="FEF3C7")


def build(output: Path) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "timeline"
    # RTL, since operator_information and instructions are written in Hebrew.
    sheet.sheet_view.rightToLeft = True

    for index, (name, width, note) in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=1, column=index, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        # Hover notes rather than a separate legend sheet: guidance that
        # sits on the column is guidance an author actually reads.
        cell.comment = Comment(note, "Maslul", height=160, width=320)
        sheet.column_dimensions[get_column_letter(index)].width = width

    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 26

    for offset, example in enumerate(EXAMPLES, start=2):
        for index, value in enumerate(example, start=1):
            cell = sheet.cell(row=offset, column=index, value=value)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.fill = EXAMPLE_FILL

    note = sheet.cell(row=len(EXAMPLES) + 3, column=1,
                      value="⚠ The shaded rows are SYNTHETIC examples. "
                            "Delete them and author your own.")
    note.font = Font(italic=True, size=9, color="92400E")

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return output


def main(argv: list[str]) -> int:
    output = Path(argv[1]) if len(argv) > 1 else ROOT / "timeline_template.xlsx"
    path = build(output)
    print(f"\n  wrote {path}")
    print(f"  {len(COLUMNS)} columns, {len(EXAMPLES)} example rows")
    print("  Required: event_id, start_time. Everything else is optional.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
