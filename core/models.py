from dataclasses import dataclass
from typing import List


@dataclass
class Event:

    timestamp: str
    event_type: str
    description: str
    confidence: float = 0.0


@dataclass
class AnalysisResult:

    timeline: List[Event]
    summary: str