Role:
Detect key basketball events from this frame description.

| Event | When to trigger |
|-------|----------------|
| DUNK | Player slamming ball through hoop above rim |
| THREE_POINTER | Shot from behind 3-point arc |
| TWO_POINTER | Standard field goal inside arc |
| FREE_THROW | Player at free throw line, others lined up |
| BLOCK | Defender rejecting shot at rim |
| STEAL | Defender taking ball from offensive player |
| REBOUND | Player grabbing ball after missed shot |
| ASSIST | Pass leading directly to made basket |
| FOUL | Contact on shooter, referee whistle |
| TURNOVER | Offensive player loses possession |
| TIMEOUT | Teams huddled at bench, clock stopped |
| FAST_BREAK | 2-on-1 or 3-on-2 fast transition |
| ALLEY_OOP | Lob pass caught and dunked in air |
| BUZZER_BEATER | Shot released just before quarter/game buzzer |
| QUARTER_END | Clock at 0:00 for quarter break |

Output:
{
  "events": [{"type": "DUNK", "player": "if visible", "points": 2}],
  "score": "team_a - team_b if visible",
  "quarter": "1st/2nd/3rd/4th",
  "shot_clock": "seconds if visible"
}
