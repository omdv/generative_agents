"""
Retrieve module - recalls relevant memories.

This module handles memory retrieval based on the current context,
using recency, relevance (semantic similarity), and importance.
"""

import logging
from typing import TYPE_CHECKING, Any

from core.memory import MemoryNode
from services.embeddings import get_embedding_service

if TYPE_CHECKING:
    from core.agent import Agent

logger = logging.getLogger(__name__)


async def retrieve(
    agent: "Agent",
    perceived: list[MemoryNode],
) -> dict[str, dict[str, Any]]:
    """
    Retrieve relevant memories for each perceived event.

    This function:
    1. For each perceived event, retrieves related memories
    2. Uses recency, relevance, and importance weighting
    3. Returns both events and thoughts that are relevant

    Args:
        agent: The agent retrieving memories.
        perceived: List of newly perceived events.

    Returns:
        Dictionary mapping event IDs to retrieved context:
        {
            event_id: {
                "curr_event": MemoryNode,
                "events": list[MemoryNode],
                "thoughts": list[MemoryNode],
            }
        }
    """
    if not perceived:
        return {}

    embedding_service = get_embedding_service()
    retrieved: dict[str, dict[str, Any]] = {}

    for event in perceived:
        # Get embedding for this event
        try:
            query_embedding = await embedding_service.embed(event.description)
        except Exception as e:
            logger.warning(f"Failed to get embedding for retrieval: {e}")
            query_embedding = None

        # Retrieve relevant memories
        try:
            relevant_memories = await agent.memory.retrieve(
                query=event.description,
                query_embedding=query_embedding,
                limit=20,
                recency_weight=agent.personality.recency_weight,
                relevance_weight=agent.personality.relevance_weight,
                importance_weight=agent.personality.importance_weight,
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            relevant_memories = []

        # Also retrieve by keywords
        keywords = list(event.keywords)
        keyword_events = agent.memory.get_events_by_keywords(keywords)
        keyword_thoughts = agent.memory.get_thoughts_by_keywords(keywords)

        # Combine and deduplicate
        all_events: list[MemoryNode] = []
        all_thoughts: list[MemoryNode] = []
        seen_ids: set[str] = {event.id}

        for mem in relevant_memories:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                if mem.node_type == "event":
                    all_events.append(mem)
                elif mem.node_type == "thought":
                    all_thoughts.append(mem)

        for mem in keyword_events:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                all_events.append(mem)

        for mem in keyword_thoughts:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                all_thoughts.append(mem)

        # Limit results
        all_events = all_events[:10]
        all_thoughts = all_thoughts[:5]

        retrieved[event.id] = {
            "curr_event": event,
            "events": all_events,
            "thoughts": all_thoughts,
        }

    logger.debug(f"{agent.name} retrieved context for {len(retrieved)} events")
    return retrieved


async def retrieve_for_reflection(
    agent: "Agent",
    count: int = 100,
) -> list[MemoryNode]:
    """
    Retrieve recent memories for reflection.

    Args:
        agent: The agent reflecting.
        count: Number of recent memories to retrieve.

    Returns:
        List of recent memories (events and thoughts).
    """
    recent_events = agent.memory.get_recent_events(count // 2)
    recent_thoughts = agent.memory.get_recent_thoughts(count // 2)

    # Combine and sort by creation time
    all_memories = recent_events + recent_thoughts
    all_memories.sort(key=lambda m: m.created_at, reverse=True)

    return all_memories[:count]


async def retrieve_for_planning(
    agent: "Agent",
    focus: str,
) -> list[MemoryNode]:
    """
    Retrieve memories relevant to planning.

    Args:
        agent: The agent planning.
        focus: Focus area for planning.

    Returns:
        List of relevant memories.
    """
    embedding_service = get_embedding_service()

    try:
        query_embedding = await embedding_service.embed(focus)
    except Exception:
        query_embedding = None

    memories = await agent.memory.retrieve(
        query=focus,
        query_embedding=query_embedding,
        limit=15,
        recency_weight=0.3,
        relevance_weight=0.5,
        importance_weight=0.2,
    )

    return memories
