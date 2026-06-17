Role:
Detect key football events from this frame description. Output ONLY events that are clearly visible.

Event types to detect:
| Event | When to trigger |
|-------|----------------|
| GOAL | Ball crosses goal line into net, players celebrating |
| GOAL_ATTEMPT | Shot on target, goalkeeper diving save |
| FOUL | Slide tackle, push, shirt pull, referee whistle visible |
| YELLOW_CARD | Referee showing yellow card to player |
| RED_CARD | Referee showing red card, player walking off |
| PENALTY | Foul in penalty box, referee pointing to spot |
| CORNER_KICK | Ball near corner flag, players in box |
| FREE_KICK | Wall forming, referee marking distance |
| OFFSIDE | Linesman flag raised, play stopped |
| SUBSTITUTION | Player leaving/entering, 4th official board up |
| INJURY | Player down on ground, medical staff on pitch |
| VAR_CHECK | Referee holding hand to ear, pitch-side monitor |
| KICK_OFF | Match starting or restarting, players in formation |
| HALF_TIME | Teams walking off, clock at 45:00 or 45+ |
| FULL_TIME | Final whistle, players shaking hands |

Output format:
{
  "events": [
    {"type": "GOAL", "team": "home/away", "player": "scorer if visible", "timestamp_relative": "from frame context"}
  ],
  "possession_team": "home/away",
  "ball_position": "attacking_third/midfield/defensive_third"
}

If no key event detected:
{
  "events": [],
  "possession_team": "unknown",
  "ball_position": "midfield"
}
