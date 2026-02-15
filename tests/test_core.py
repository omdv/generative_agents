"""Tests for core simulation components."""

import pytest
from datetime import datetime
from pathlib import Path

from core.personality import PersonalityConfig, ScratchPad
from core.memory import MemoryNode, MemoryStore
from core.agent import Agent


class TestPersonalityConfig:
    """Tests for PersonalityConfig dataclass."""

    def test_create_personality(self):
        """Test creating a basic personality."""
        personality = PersonalityConfig(
            name="John Doe",
            age=30,
            innate="curious, kind",
            learned="John is a software engineer.",
            currently="Working on a project.",
            lifestyle="Goes to bed at 11pm.",
        )

        assert personality.name == "John Doe"
        assert personality.first_name == "John"
        assert personality.last_name == "Doe"
        assert personality.age == 30

    def test_identity_summary(self):
        """Test getting identity summary."""
        personality = PersonalityConfig(
            name="Jane Smith",
            age=25,
            innate="creative",
            learned="Jane is an artist.",
            currently="Painting.",
            lifestyle="Works from home.",
        )

        summary = personality.get_identity_summary()

        assert "Jane Smith" in summary
        assert "25" in summary
        assert "creative" in summary

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = PersonalityConfig(
            name="Test Person",
            age=40,
            innate="test",
        )

        data = original.to_dict()
        restored = PersonalityConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.age == original.age
        assert restored.innate == original.innate


class TestScratchPad:
    """Tests for ScratchPad dataclass."""

    def test_create_empty_scratch(self):
        """Test creating an empty scratch pad."""
        scratch = ScratchPad()

        assert scratch.current_action is None
        assert scratch.daily_schedule == []
        assert scratch.planned_path == []

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = ScratchPad(
            current_action="test",
            action_duration=60,
        )

        data = original.to_dict()
        restored = ScratchPad.from_dict(data)

        assert restored.current_action == original.current_action
        assert restored.action_duration == original.action_duration


class TestMemoryNode:
    """Tests for MemoryNode dataclass."""

    def test_create_memory_node(self):
        """Test creating a memory node."""
        node = MemoryNode(
            id="test_1",
            node_type="event",
            subject="John",
            predicate="is",
            object="walking",
            description="John is walking in the park.",
            keywords={"john", "walking", "park"},
            poignancy=5.0,
        )

        assert node.id == "test_1"
        assert node.node_type == "event"
        assert node.spo_summary() == ("John", "is", "walking")

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = MemoryNode(
            id="test_2",
            node_type="thought",
            subject="Jane",
            predicate="thinks",
            object="about work",
            description="Jane is thinking about work.",
            keywords={"jane", "work"},
            poignancy=7.0,
            created_at=datetime(2023, 2, 13, 10, 0, 0),
        )

        data = original.to_dict()
        restored = MemoryNode.from_dict(data)

        assert restored.id == original.id
        assert restored.node_type == original.node_type
        assert restored.description == original.description


class TestMemoryStore:
    """Tests for MemoryStore."""

    @pytest.fixture
    def memory_store(self, tmp_path):
        """Create a temporary memory store."""
        db_path = tmp_path / "test_memory.db"
        return MemoryStore(db_path, agent_id="test_agent")

    @pytest.mark.asyncio
    async def test_add_event(self, memory_store):
        """Test adding an event memory."""
        node = await memory_store.add_event(
            subject="Test",
            predicate="is",
            object_="testing",
            description="Test is testing the system.",
            keywords={"test", "testing"},
            poignancy=5.0,
        )

        assert node.id.startswith("event_")
        assert node.node_type == "event"

        # Should be retrievable
        recent = memory_store.get_recent_events(10)
        assert len(recent) == 1
        assert recent[0].id == node.id

    @pytest.mark.asyncio
    async def test_add_thought(self, memory_store):
        """Test adding a thought memory."""
        node = await memory_store.add_thought(
            subject="Agent",
            predicate="reflects on",
            object_="experience",
            description="The agent reflects on their experience.",
            keywords={"agent", "reflection"},
            poignancy=7.0,
        )

        assert node.id.startswith("thought_")
        assert node.node_type == "thought"
        assert node.depth >= 1

    @pytest.mark.asyncio
    async def test_retrieve(self, memory_store):
        """Test memory retrieval."""
        # Add some memories
        await memory_store.add_event(
            subject="Alice",
            predicate="met",
            object_="Bob",
            description="Alice met Bob at the cafe.",
            keywords={"alice", "bob", "cafe"},
            poignancy=6.0,
        )

        await memory_store.add_event(
            subject="Alice",
            predicate="had",
            object_="coffee",
            description="Alice had coffee.",
            keywords={"alice", "coffee"},
            poignancy=3.0,
        )

        # Retrieve
        results = await memory_store.retrieve(
            query="meeting Bob",
            limit=10,
        )

        assert len(results) == 2


class TestAgent:
    """Tests for Agent class."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create a test agent."""
        personality = PersonalityConfig(
            name="Test Agent",
            age=25,
            innate="test trait",
        )
        return Agent(
            personality=personality,
            db_path=tmp_path / "test.db",
            position=(10, 10),
        )

    def test_create_agent(self, agent):
        """Test creating an agent."""
        assert agent.name == "Test Agent"
        assert agent.position == (10, 10)
        assert agent.current_time is None

    def test_get_identity_summary(self, agent):
        """Test getting identity summary."""
        summary = agent.get_identity_summary()
        assert "Test Agent" in summary

    def test_is_action_finished_no_action(self, agent):
        """Test action finished check with no action."""
        assert agent.is_action_finished() is True

    def test_to_dict_from_dict(self, agent, tmp_path):
        """Test serialization round-trip."""
        data = agent.to_dict()
        restored = Agent.from_dict(data, tmp_path / "restored.db")

        assert restored.name == agent.name
        assert restored.position == agent.position
