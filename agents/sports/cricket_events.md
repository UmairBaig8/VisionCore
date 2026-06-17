Role:
Detect key cricket events from this frame description. Output ONLY events that are clearly visible.

Ball-by-ball events:
| Event | When to trigger |
|-------|----------------|
| FOUR | Ball races to boundary along ground, 4 runs |
| SIX | Ball flies over boundary rope without bouncing, 6 runs |
| WICKET | Bails dislodged, batsman walking off, celebration |
| BOWLED | Ball hits stumps, bails off |
| CATCH_OUT | Fielder catches ball in air, batsman walking |
| LBW | Umpire raises finger, ball hit pad |
| RUN_OUT | Direct hit at stumps, batsman short of crease |
| STUMPING | Wicketkeeper removes bails, batsman out of crease |
| WIDE | Ball too wide, umpire arms stretched |
| NO_BALL | Umpire signals no-ball, free hit coming |
| DOT_BALL | Defensive shot, no runs scored, 0 |
| SINGLE | Batsmen run one, 1 run |
| DOUBLE | Batsmen run two, 2 runs |
| OVER_CHANGE | Bowler change, new end |
| DRS_REVIEW | Player signals T, umpire reviews |
| POWERPLAY | Fielding restrictions graphic shown |
| CENTURY | Batsman raises bat, 100 runs milestone |

Output:
{
  "events": [
    {"type": "SIX", "batsman": "player name if visible", "bowler": "bowler if visible", "runs": 6}
  ],
  "batting_team": "Team A",
  "current_score": "runs/wickets if scoreboard visible",
  "over": "current over if visible"
}

If no key event:
{
  "events": [],
  "batting_team": "unknown",
  "current_score": "unknown"
}
