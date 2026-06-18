"""
Global match context — tracks state across frames.
Updated by agent router, read by orchestrator for report generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MatchContext:
    # ── identity (set once, never changes) ──
    sport: str = "generic"
    video_type: str = "full_match"
    teams: List[str] = field(default_factory=list)
    location: str = ""
    league: str = ""

    # ── score state (updated on GOAL events) ──
    home_score: int = 0
    away_score: int = 0
    last_score_change: Optional[str] = None  # timestamp

    # ── phase tracking (changes the agent strategy) ──
    phase: str = "kickoff"
    # valid phases: kickoff, open_play, attack_building, attack_final_third,
    #               set_piece, dead_ball, half_time, full_time, replay, commercial
    phase_changed: bool = False
    phase_frame_count: int = 0

    # ── momentum ──
    attacking_team: str = "unknown"   # "home" or "away"
    possession_pct: int = 50
    momentum_score: int = 0  # -50 to +50 (negative = away, positive = home)

    # ── event tracking ──
    last_event: Optional[str] = None
    last_event_time: Optional[str] = None
    key_events: List[Dict] = field(default_factory=list)
    consecutive_generic_frames: int = 0

    # ── agent routing hints ──
    force_full_pipeline: bool = False    # set when a goal/card is detected
    skip_event_detection: bool = False   # set after N generic frames
    skip_deep_analysis: bool = True      # default: skip reasoning+commentary

    def add_key_event(self, event: Dict):
        event["global_time"] = event.get("timestamp", "?")
        self.key_events.append(event)
        self.last_event = event.get("type", "unknown")
        self.last_event_time = event.get("timestamp", "?")

        et = event.get("type", "")
        if et in ("GOAL",):
            self.last_score_change = event.get("timestamp")
            self.force_full_pipeline = True
            self.consecutive_generic_frames = 0

        elif et in ("HALF_TIME",):
            self.phase = "half_time"
            self.phase_changed = True

        elif et in ("FULL_TIME",):
            self.phase = "full_time"
            self.phase_changed = True

    def update_phase(self, new_phase: str):
        if new_phase != self.phase:
            self.phase_changed = True
            self.phase = new_phase
            self.phase_frame_count = 0
        else:
            self.phase_changed = False
            self.phase_frame_count += 1

    def update_momentum(self, possession_home: int):
        self.possession_pct = possession_home
        if possession_home > 55:
            self.attacking_team = "home"
            self.momentum_score = min(50, self.momentum_score + 5)
        elif possession_home < 45:
            self.attacking_team = "away"
            self.momentum_score = max(-50, self.momentum_score - 5)

    def score_string(self) -> str:
        return f"{self.home_score}-{self.away_score}"

    def summary(self) -> Dict:
        return {
            "sport": self.sport,
            "type": self.video_type,
            "teams": self.teams,
            "location": self.location,
            "league": self.league,
            "score": self.score_string(),
            "phase": self.phase,
            "momentum": self.momentum_score,
            "key_events_count": len(self.key_events),
        }
