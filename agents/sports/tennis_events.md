Role:
Detect key tennis events from this frame description.

| Event | When to trigger |
|-------|----------------|
| ACE | Serve untouched by receiver, clean winner |
| DOUBLE_FAULT | Second serve fault, point to receiver |
| WINNER | Clean winning shot, opponent cannot reach |
| UNFORCED_ERROR | Player misses easy shot into net/out |
| BREAK_POINT | Receiver has chance to break serve |
| BREAK | Server loses game, score change |
| SET_POINT | Player one game from winning set |
| MATCH_POINT | Player one point from winning match |
| DEUCE | Score tied at 40-40 |
| ADVANTAGE | Player wins point after deuce |
| TIEBREAK | Game score 6-6, tiebreak format |
| CHALLENGE | Player challenges line call, hawkeye replay |
| SET_WON | Player wins set, scoreboard update |
| MEDICAL_TIMEOUT | Trainer on court attending player |
| RACKET_SMASH | Player breaks racket in frustration |
| NET_CORD | Ball clips net and drops over |
| LOB | High arcing shot over opponent at net |

Output:
{
  "events": [{"type": "ACE", "player": "server name if visible"}],
  "game_score": "15-0, 30-40, etc",
  "set_score": "6-4, 7-6 etc",
  "serving": "player name"
}
