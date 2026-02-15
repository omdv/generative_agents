"""
Personality configuration for generative agents.

The personality defines the core identity and behavioral traits of an agent.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonalityConfig:
    """
    Core personality configuration for an agent.

    Attributes:
        name: Full name of the agent (unique identifier).
        first_name: Agent's first name.
        last_name: Agent's last name.
        age: Agent's age in years.
        innate: Core personality traits (L0 - permanent).
            Example: "curious, kind, introverted"
        learned: Biographical information and learned behaviors (L1 - stable).
            Example: "Maria is a software engineer who moved to the city..."
        currently: Current goals and state (L2 - dynamic).
            Example: "Maria is working on a new project and feeling stressed..."
        lifestyle: Daily routines and habits.
            Example: "Maria goes to bed around 11pm, wakes up at 7am..."
        living_area: Where the agent lives in the world.
            Example: "the Ville:artist's co-living space:Maria's room"
        daily_plan_req: Special daily planning requirements.
            Example: "Maria needs to attend a meeting at 2pm"
    """

    name: str
    first_name: str = ""
    last_name: str = ""
    age: int = 25
    innate: str = ""
    learned: str = ""
    currently: str = ""
    lifestyle: str = ""
    living_area: str = ""
    daily_plan_req: str = ""

    # Perception parameters
    vision_r: int = 8  # Vision radius in tiles
    att_bandwidth: int = 8  # Attention bandwidth (max events to perceive)
    retention: int = 8  # How many recent events to retain

    # Reflection parameters
    recency_weight: float = 1.0
    relevance_weight: float = 1.0
    importance_weight: float = 1.0
    recency_decay: float = 0.99
    importance_trigger_max: int = 150
    reflection_count: int = 5

    def __post_init__(self) -> None:
        """Parse first and last name from full name if not provided."""
        if not self.first_name and self.name:
            parts = self.name.split()
            self.first_name = parts[0] if parts else ""
            self.last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

    def get_identity_summary(self) -> str:
        """
        Get a summary of the agent's identity for use in prompts.

        This is the "Identity Stable Set" (ISS) - the bare minimum description
        that gets used in almost all prompts involving this agent.
        """
        summary = f"Name: {self.name}\n"
        summary += f"Age: {self.age}\n"
        summary += f"Innate traits: {self.innate}\n"
        summary += f"Learned traits: {self.learned}\n"
        summary += f"Currently: {self.currently}\n"
        summary += f"Lifestyle: {self.lifestyle}\n"
        if self.daily_plan_req:
            summary += f"Daily plan requirement: {self.daily_plan_req}\n"
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert personality to dictionary for serialization."""
        return {
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "innate": self.innate,
            "learned": self.learned,
            "currently": self.currently,
            "lifestyle": self.lifestyle,
            "living_area": self.living_area,
            "daily_plan_req": self.daily_plan_req,
            "vision_r": self.vision_r,
            "att_bandwidth": self.att_bandwidth,
            "retention": self.retention,
            "recency_weight": self.recency_weight,
            "relevance_weight": self.relevance_weight,
            "importance_weight": self.importance_weight,
            "recency_decay": self.recency_decay,
            "importance_trigger_max": self.importance_trigger_max,
            "reflection_count": self.reflection_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonalityConfig":
        """Create personality from dictionary."""
        return cls(**data)


@dataclass
class ScratchPad:
    """
    Short-term memory and current state for an agent.

    The scratch pad holds transient information about what the agent
    is currently doing, planning to do, and has recently perceived.
    """

    # Current action state
    current_action: str | None = None
    action_address: str | None = None
    action_start_time: str | None = None
    action_duration: int = 0
    action_description: str | None = None
    action_emoji: str | None = None
    action_event: tuple[str, str | None, str | None] = field(
        default_factory=lambda: ("", None, None)
    )

    # Object interaction state
    object_action_description: str | None = None
    object_action_emoji: str | None = None
    object_action_event: tuple[str, str | None, str | None] = field(
        default_factory=lambda: ("", None, None)
    )

    # Conversation state
    chatting_with: str | None = None
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    chatting_buffer: dict[str, int] = field(default_factory=dict)
    chat_end_time: str | None = None

    # Planning state
    daily_requirements: list[str] = field(default_factory=list)
    daily_schedule: list[tuple[str, int]] = field(default_factory=list)
    daily_schedule_hourly: list[tuple[str, int]] = field(default_factory=list)

    # Movement state
    planned_path: list[tuple[int, int]] = field(default_factory=list)
    path_set: bool = False

    # Reflection state
    importance_trigger_current: int = 150
    importance_element_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert scratch pad to dictionary for serialization."""
        return {
            "current_action": self.current_action,
            "action_address": self.action_address,
            "action_start_time": self.action_start_time,
            "action_duration": self.action_duration,
            "action_description": self.action_description,
            "action_emoji": self.action_emoji,
            "action_event": list(self.action_event),
            "object_action_description": self.object_action_description,
            "object_action_emoji": self.object_action_emoji,
            "object_action_event": list(self.object_action_event),
            "chatting_with": self.chatting_with,
            "chat_history": self.chat_history,
            "chatting_buffer": self.chatting_buffer,
            "chat_end_time": self.chat_end_time,
            "daily_requirements": self.daily_requirements,
            "daily_schedule": self.daily_schedule,
            "daily_schedule_hourly": self.daily_schedule_hourly,
            "planned_path": self.planned_path,
            "path_set": self.path_set,
            "importance_trigger_current": self.importance_trigger_current,
            "importance_element_count": self.importance_element_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScratchPad":
        """Create scratch pad from dictionary."""
        # Convert lists back to tuples where needed
        if "action_event" in data:
            data["action_event"] = tuple(data["action_event"])
        if "object_action_event" in data:
            data["object_action_event"] = tuple(data["object_action_event"])
        return cls(**data)
