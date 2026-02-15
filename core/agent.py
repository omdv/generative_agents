"""
Agent class for generative agents.

This module implements the core Agent class that powers individual agents
in the simulation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.memory import MemoryStore
from core.personality import PersonalityConfig, ScratchPad

if TYPE_CHECKING:
    from core.world import World

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents an action an agent will take."""

    address: str  # Location in world (e.g., "the Ville:cafe:counter")
    description: str  # What the agent is doing
    emoji: str  # Visual representation
    duration: int  # Duration in minutes
    event: tuple[str, str | None, str | None]  # Subject, predicate, object
    object_description: str | None = None
    object_emoji: str | None = None
    object_event: tuple[str, str | None, str | None] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)


class Agent:
    """
    A generative agent in the simulation.

    Each agent has:
    - A personality that defines their identity and traits
    - A memory store for long-term memories
    - A scratch pad for short-term state
    - A position in the world
    """

    def __init__(
        self,
        personality: PersonalityConfig,
        db_path: str | Path,
        position: tuple[int, int] = (0, 0),
    ):
        """
        Initialize an agent.

        Args:
            personality: The agent's personality configuration.
            db_path: Path to the SQLite database for memory storage.
            position: Initial position in the world (x, y).
        """
        self.personality = personality
        self.memory = MemoryStore(db_path, agent_id=personality.name)
        self.scratch = ScratchPad()
        self.position = position
        self.current_time: datetime | None = None

        # Spatial memory (knowledge of the world)
        self.spatial_memory: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.personality.name

    async def cognitive_loop(
        self,
        world: "World",
        agents: dict[str, "Agent"],
        current_time: datetime,
    ) -> Action | None:
        """
        Execute one step of the cognitive cycle.

        This is the main cognitive function that processes:
        1. Perceive - What does the agent see/notice?
        2. Retrieve - What memories are relevant?
        3. Plan - What should the agent do?
        4. Reflect - Should the agent form new insights?
        5. Execute - What specific action to take?

        Args:
            world: The world the agent is in.
            agents: Dictionary of all agents in the simulation.
            current_time: Current simulation time.

        Returns:
            The action to execute, or None if no action needed.
        """
        from core.cognitive import execute, perceive, plan, reflect, retrieve

        # Determine if this is a new day BEFORE updating current_time
        new_day = self._check_new_day(current_time)

        # Update current time
        self.current_time = current_time

        # 1. Perceive - Get events visible to the agent
        perceived = await perceive(self, world)

        # 2. Retrieve - Get relevant memories for perceived events
        retrieved = await retrieve(self, perceived)

        # 3. Plan - Determine what to do next
        action_plan = await plan(self, world, agents, new_day, retrieved)
        logger.debug(f"{self.name}: plan returned '{action_plan}'")

        # 4. Reflect - Form new insights if needed
        await reflect(self)

        # 5. Execute - Convert plan to specific action
        if action_plan:
            action = await execute(self, world, agents, action_plan)
            logger.debug(f"{self.name}: execute returned action with path_len={len(action.path) if action else 0}")
            return action

        logger.warning(f"{self.name}: no action plan, returning None")
        return None

    def _check_new_day(self, current_time: datetime) -> str | bool:
        """
        Check if this is a new day in the simulation.

        Returns:
            - "First day" if this is the very first time step
            - "New day" if it's a new day since last step
            - False if it's the same day
        """
        if self.current_time is None:
            return "First day"

        if self.current_time.date() != current_time.date():
            return "New day"

        return False

    def get_identity_summary(self) -> str:
        """Get a summary of the agent's identity for prompts."""
        summary = self.personality.get_identity_summary()
        if self.current_time:
            summary += f"Current Date: {self.current_time.strftime('%A %B %d')}\n"
        return summary

    def get_current_action(self) -> tuple[str, str | None, str | None, str | None]:
        """Get the current action as (subject, predicate, object, description)."""
        if not self.scratch.action_address:
            return (self.name, None, None, None)
        return (
            self.scratch.action_event[0],
            self.scratch.action_event[1],
            self.scratch.action_event[2],
            self.scratch.action_description,
        )

    def set_action(
        self,
        address: str,
        duration: int,
        description: str,
        emoji: str,
        event: tuple[str, str | None, str | None],
        object_description: str | None = None,
        object_emoji: str | None = None,
        object_event: tuple[str, str | None, str | None] | None = None,
        chatting_with: str | None = None,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> None:
        """Set the current action for this agent."""
        self.scratch.action_address = address
        self.scratch.action_duration = duration
        self.scratch.action_description = description
        self.scratch.action_emoji = emoji
        self.scratch.action_event = event
        self.scratch.action_start_time = (
            self.current_time.isoformat() if self.current_time else None
        )
        self.scratch.object_action_description = object_description
        self.scratch.object_action_emoji = object_emoji
        self.scratch.object_action_event = object_event or ("", None, None)
        self.scratch.chatting_with = chatting_with
        self.scratch.chat_history = chat_history or []
        self.scratch.path_set = False

    def is_action_finished(self) -> bool:
        """Check if the current action has finished."""
        if not self.scratch.action_address:
            return True

        if not self.current_time or not self.scratch.action_start_time:
            return True

        # If chatting, check chat end time
        if self.scratch.chatting_with and self.scratch.chat_end_time:
            end_time = datetime.fromisoformat(self.scratch.chat_end_time)
            return self.current_time >= end_time

        # Otherwise check action duration
        start_time = datetime.fromisoformat(self.scratch.action_start_time)
        elapsed_minutes = (self.current_time - start_time).total_seconds() / 60
        return elapsed_minutes >= self.scratch.action_duration

    def get_daily_schedule_index(self, advance: int = 0) -> int:
        """Get the current index in the daily schedule."""
        if not self.current_time:
            return 0

        today_minutes = self.current_time.hour * 60 + self.current_time.minute + advance
        elapsed = 0

        for idx, (_, duration) in enumerate(self.scratch.daily_schedule):
            elapsed += duration
            if elapsed > today_minutes:
                return idx

        return len(self.scratch.daily_schedule)

    def to_dict(self) -> dict[str, Any]:
        """Convert agent state to dictionary for serialization."""
        return {
            "name": self.name,
            "personality": self.personality.to_dict(),
            "scratch": self.scratch.to_dict(),
            "position": list(self.position),
            "current_time": self.current_time.isoformat() if self.current_time else None,
            "spatial_memory": self.spatial_memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], db_path: str | Path) -> "Agent":
        """Create an agent from dictionary."""
        personality = PersonalityConfig.from_dict(data["personality"])
        agent = cls(
            personality=personality,
            db_path=db_path,
            position=tuple(data.get("position", [0, 0])),
        )

        if data.get("scratch"):
            agent.scratch = ScratchPad.from_dict(data["scratch"])

        if data.get("current_time"):
            agent.current_time = datetime.fromisoformat(data["current_time"])

        if data.get("spatial_memory"):
            agent.spatial_memory = data["spatial_memory"]

        return agent
