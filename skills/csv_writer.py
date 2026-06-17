import csv
import json
from pathlib import Path
from core.paths import output_dir


def save_csv(events, video_name, context=None):
    out = output_dir() / "csv"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{video_name}.csv"

    fieldnames = [
        "timestamp",
        "scene",
        "phase",
        "score",
        "key_events",
        "event",
        "reasoning",
        "commentary",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            row = {}
            for k in fieldnames:
                val = ev.get(k, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                row[k] = val
            writer.writerow(row)

    # ── aggregation CSV (summary) ──
    if context:
        agg_path = out / f"{video_name}_summary.csv"
        agg_fields = [
            "sport", "video_type", "teams", "location", "league",
            "home_score", "away_score", "final_score", "phase",
            "key_events_count", "key_events_list",
        ]
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields)
            writer.writeheader()
            row = {
                "sport": context.sport,
                "video_type": context.video_type,
                "teams": " vs ".join(context.teams) if context.teams else "",
                "location": context.location,
                "league": context.league,
                "home_score": context.home_score,
                "away_score": context.away_score,
                "final_score": context.score_string(),
                "phase": context.phase,
                "key_events_count": len(context.key_events),
                "key_events_list": json.dumps([
                    {"t": e.get("timestamp", ""),
                     "type": e.get("type", ""),
                     "team": e.get("team", e.get("batsman", e.get("player", "")))}
                    for e in context.key_events
                ]),
            }
            writer.writerow(row)

    return path
