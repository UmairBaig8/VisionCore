"""
Event emitter — callbacks that the orchestrator fires during analysis.
CLI mode uses no-op; API mode wires to WebSocket.
"""


class EventEmitter:
    """Base emitter — no-op. Override in API server."""

    def on_detection(self, sport, video_type, location, league, teams):
        pass

    def on_agent_active(self, agent_name):
        pass

    def on_ball_position(self, ball_position, timestamp):
        pass

    def on_scene(self, timestamp, scene_type, activity, scene_raw):
        pass

    def on_key_event(self, event):
        pass

    def on_clip_generated(self, event_type, timestamp, path, total_clips):
        pass

    def on_score_change(self, home, away):
        pass

    def on_phase_change(self, phase):
        pass

    def on_progress(self, frame, total, pct):
        pass

    def on_complete(self, report_path, csv_path, reel_paths, key_events_count):
        pass

    def on_analysis_complete(self, event_count, final_score):
        """Fired BEFORE reel generation — unblocks UI immediately."""

    def on_reel_progress(self, flavor, idx, total):
        """Fired during reel generation so UI shows activity."""

    def on_error(self, message):
        pass
