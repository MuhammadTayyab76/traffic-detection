"""
rule_engine.py

Pluggable rule engine for traffic violation and event detection.

Architecture:
  - Rule (abstract base)  : defines the interface every rule must implement
  - Event (dataclass)     : what a triggered rule emits
  - RuleEngine            : manages registered rules, calls them each frame

Adding a new rule (e.g. speed estimation):
  1. Create src/analytics/rules/speed_estimator.py
  2. Subclass Rule
  3. Implement evaluate() — return list[Event]
  4. Register with engine.register(SpeedEstimator())
  5. Done — no other files change

The inference engine, GUI, and API never need to know what
rules are registered. They just receive a list of Events.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.analytics.stats_collector import Detection, FrameStats


# Event dataclass 

@dataclass
class Event:
    """
    Emitted by a Rule when it triggers.
    Consumed by the GUI (show alert) and API (stream to client).
    """
    rule_name:    str           # which rule fired
    severity:     str           # 'info' | 'warning' | 'critical'
    message:      str           # human-readable description
    frame_idx:    int           # which frame triggered it
    timestamp_ms: float         # milliseconds since session start
    detections:   list[Detection] = field(default_factory=list)
                                # the detections that caused the event
    metadata:     dict          = field(default_factory=dict)
                                # rule-specific extra data


# Abstract Rule base 

class Rule(ABC):
    """
    Abstract base class for all traffic rules.

    Subclass this to add any new detection capability.
    Only evaluate() is mandatory to implement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this rule. Used in Event.rule_name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this rule detects."""
        return "No description provided."

    @property
    def enabled(self) -> bool:
        """Rules can be disabled at runtime without unregistering."""
        return True

    @abstractmethod
    def evaluate(
        self,
        detections: list[Detection],
        frame_stats: FrameStats,
    ) -> list[Event]:
        """
        Analyse detections for the current frame and return any Events.

        Args:
            detections:  All detections in the current frame.
            frame_stats: Statistics snapshot for the current frame.

        Returns:
            List of Event objects. Empty list if nothing triggered.
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each new video session.
        Override if your rule maintains state across frames.
        """
        pass


# Rule engine 

class RuleEngine:
    """
    Manages a collection of Rule objects and evaluates them each frame.

    Usage:
        engine = RuleEngine()
        engine.register(RedLightRule())
        engine.register(CrowdingRule(threshold=10))

        for frame in video:
            detections = model.detect(frame)
            stats      = collector.update(detections)
            events     = engine.evaluate(detections, stats)
            for event in events:
                gui.show_alert(event)
    """

    def __init__(self):
        self._rules:  list[Rule]  = []
        self._history: list[dict] = []   # log of all events this session

    # Registration 

    def register(self, rule: Rule) -> None:
        """
        Register a rule with the engine.

        Args:
            rule: Any Rule subclass instance.
        """
        self._rules.append(rule)
        print(f"  RuleEngine: registered '{rule.name}' — {rule.description}")

    def unregister(self, rule_name: str) -> None:
        """Remove a rule by name."""
        self._rules = [r for r in self._rules if r.name != rule_name]

    def list_rules(self) -> list[dict]:
        """Return info about all registered rules."""
        return [
            {
                "name":        r.name,
                "description": r.description,
                "enabled":     r.enabled,
            }
            for r in self._rules
        ]

    # Evaluation 

    def evaluate(
        self,
        detections: list[Detection],
        frame_stats: FrameStats,
    ) -> list[Event]:
        """
        Run all enabled rules against the current frame.

        Args:
            detections:  Detections from the inference engine.
            frame_stats: Frame statistics from StatsCollector.

        Returns:
            All Events emitted by any rule this frame.
        """
        all_events: list[Event] = []

        for rule in self._rules:
            if not rule.enabled:
                continue
            try:
                events = rule.evaluate(detections, frame_stats)
                all_events.extend(events)
            except Exception as e:
                print(f"  RuleEngine WARNING: rule '{rule.name}' raised {e}")

        # Log events
        for event in all_events:
            self._history.append({
                "rule":       event.rule_name,
                "severity":   event.severity,
                "message":    event.message,
                "frame":      event.frame_idx,
                "timestamp":  event.timestamp_ms,
            })

        return all_events

    # Session management 

    def reset(self) -> None:
        """Reset all rules and clear event history for a new session."""
        for rule in self._rules:
            rule.reset()
        self._history = []

    def get_event_history(self) -> list[dict]:
        """Return the full event log for this session."""
        return self._history

    def get_event_summary(self) -> dict:
        """Return aggregate count of events per rule."""
        summary: dict[str, int] = {}
        for entry in self._history:
            summary[entry["rule"]] = summary.get(entry["rule"], 0) + 1
        return summary