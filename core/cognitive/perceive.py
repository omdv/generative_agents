"""
Perceive module - processes sensory input from the world.

This module handles what an agent can see and notice in their environment,
filtering based on attention bandwidth and recency.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from core.memory import MemoryNode
from services.embeddings import get_embedding_service
from services.llm import get_llm_service

if TYPE_CHECKING:
    from core.agent import Agent
    from core.world import World

logger = logging.getLogger(__name__)


async def perceive(agent: "Agent", world: "World") -> list[MemoryNode]:
    """
    Perceive the environment and create new memories for notable events.

    This function:
    1. Gets all events within the agent's vision radius
    2. Filters to events the agent hasn't recently perceived
    3. Creates memory nodes for new perceptions
    4. Limits to attention bandwidth

    Args:
        agent: The perceiving agent.
        world: The world to perceive.

    Returns:
        List of newly perceived events as MemoryNodes.
    """
    x, y = agent.position
    vision_r = agent.personality.vision_r
    att_bandwidth = agent.personality.att_bandwidth
    retention = agent.personality.retention

    # Get events within vision radius
    nearby_events = world.get_nearby_events(x, y, vision_r)

    # Get recently perceived events to filter out
    recent_spo = agent.memory.get_summarized_latest_events(retention)

    # Filter to new events
    new_events: list[tuple[tuple[int, int], tuple[str, str | None, str | None, str | None]]] = []
    for pos, event in nearby_events:
        subject, predicate, obj, _ = event
        spo = (subject, predicate, obj)

        # Skip if already recently perceived
        if spo in recent_spo:
            continue

        # Skip self-events (agent's own actions)
        if subject == agent.name:
            continue

        # Skip idle events
        if predicate is None and obj is None:
            continue

        new_events.append((pos, event))

    # Sort by distance (closer events are more likely to be noticed)
    new_events.sort(key=lambda e: abs(e[0][0] - x) + abs(e[0][1] - y))

    # Limit to attention bandwidth
    new_events = new_events[:att_bandwidth]

    # Create memory nodes for perceived events
    perceived_nodes: list[MemoryNode] = []
    llm = get_llm_service()
    embedding_service = get_embedding_service()

    for pos, event in new_events:
        subject, predicate, obj, description = event

        # Create description if not provided
        if not description:
            if predicate and obj:
                description = f"{subject} is {predicate} {obj}"
            elif predicate:
                description = f"{subject} is {predicate}"
            else:
                description = f"{subject} is present"

        # Generate importance score
        try:
            poignancy = await llm.generate_importance(
                description=description,
                agent_identity=agent.get_identity_summary(),
            )
        except Exception as e:
            logger.warning(f"Failed to generate importance: {e}")
            poignancy = 5.0

        # Generate keywords
        try:
            keywords = await llm.generate_keywords(
                description=description,
                agent_name=agent.name,
            )
        except Exception as e:
            logger.warning(f"Failed to generate keywords: {e}")
            keywords = {subject.lower(), agent.name.lower()}
            if predicate:
                keywords.add(predicate.lower())
            if obj:
                keywords.add(obj.lower())

        # Generate embedding
        try:
            embedding = await embedding_service.embed(description)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            embedding = None

        # Create memory node
        try:
            node = await agent.memory.add_event(
                subject=subject,
                predicate=predicate,
                object_=obj,
                description=description,
                keywords=keywords,
                poignancy=poignancy,
                embedding=embedding,
                created_at=agent.current_time or datetime.now(),
            )
            perceived_nodes.append(node)
        except Exception as e:
            logger.error(f"Failed to create memory node: {e}")

    logger.debug(f"{agent.name} perceived {len(perceived_nodes)} new events")
    return perceived_nodes
