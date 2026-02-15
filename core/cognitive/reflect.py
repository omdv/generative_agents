"""
Reflect module - forms higher-level insights from memories.

This module handles the reflection process where agents synthesize
their experiences into new thoughts and insights.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from core.cognitive.retrieve import retrieve_for_reflection
from core.memory import MemoryNode
from services.embeddings import get_embedding_service
from services.llm import get_llm_service

if TYPE_CHECKING:
    from core.agent import Agent

logger = logging.getLogger(__name__)


async def reflect(agent: "Agent") -> list[MemoryNode]:
    """
    Generate reflections based on recent experiences.

    Reflection is triggered when the cumulative importance of recent
    events exceeds a threshold. The agent then synthesizes insights
    from their experiences.

    Args:
        agent: The reflecting agent.

    Returns:
        List of new thought nodes created during reflection.
    """
    # Check if reflection is needed
    if not _should_reflect(agent):
        return []

    # Get recent memories for reflection
    recent_memories = await retrieve_for_reflection(agent, count=100)

    if not recent_memories:
        return []

    # Generate reflection questions
    questions = await _generate_reflection_questions(agent, recent_memories)

    # Generate insights for each question
    new_thoughts: list[MemoryNode] = []

    for question in questions:
        thought = await _generate_insight(agent, question, recent_memories)
        if thought:
            new_thoughts.append(thought)

    # Reset importance trigger
    agent.scratch.importance_trigger_current = agent.personality.importance_trigger_max
    agent.scratch.importance_element_count = 0

    logger.info(f"{agent.name} generated {len(new_thoughts)} reflections")
    return new_thoughts


def _should_reflect(agent: "Agent") -> bool:
    """Determine if the agent should reflect now."""
    # Reflection is triggered when cumulative importance exceeds threshold
    return agent.scratch.importance_trigger_current <= 0


async def _generate_reflection_questions(
    agent: "Agent",
    memories: list[MemoryNode],
) -> list[str]:
    """Generate questions to guide reflection."""
    llm = get_llm_service()

    # Format recent memories for prompt
    memory_summaries = []
    for mem in memories[:20]:  # Limit for prompt size
        memory_summaries.append(f"- {mem.description}")

    memory_text = "\n".join(memory_summaries)

    prompt = f"""Given {agent.name}'s recent experiences:

{memory_text}

What are 3 high-level questions or insights that {agent.name} might reflect on based on these experiences? Focus on patterns, relationships, and self-understanding.

Format as numbered questions:
1.
2.
3."""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.8,
            max_tokens=300,
        )

        # Parse questions
        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering/bullet
                question = line.lstrip("0123456789.-) ").strip()
                if question:
                    questions.append(question)

        return questions[:3]

    except Exception as e:
        logger.warning(f"Failed to generate reflection questions: {e}")
        return []


async def _generate_insight(
    agent: "Agent",
    question: str,
    memories: list[MemoryNode],
) -> MemoryNode | None:
    """Generate an insight/thought in response to a question."""
    llm = get_llm_service()
    embedding_service = get_embedding_service()

    # Find memories relevant to this question
    try:
        question_embedding = await embedding_service.embed(question)
        relevant_memories = await agent.memory.retrieve(
            query=question,
            query_embedding=question_embedding,
            limit=10,
            recency_weight=0.2,
            relevance_weight=0.6,
            importance_weight=0.2,
        )
    except Exception as e:
        logger.warning(f"Failed to retrieve for reflection: {e}")
        relevant_memories = memories[:10]

    # Format relevant memories
    memory_text = "\n".join(
        f"- {mem.description}" for mem in relevant_memories
    )

    prompt = f"""Based on {agent.name}'s experiences:

{memory_text}

Reflecting on: {question}

What insight or conclusion might {agent.name} draw? Provide a single, clear statement (1-2 sentences) that represents this insight.

Insight:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
        )

        insight = response.strip().strip('"')

        if not insight:
            return None

        # Generate embedding for the insight
        try:
            embedding = await embedding_service.embed(insight)
        except Exception:
            embedding = None

        # Generate importance
        try:
            poignancy = await llm.generate_importance(
                description=insight,
                agent_identity=agent.get_identity_summary(),
            )
        except Exception:
            poignancy = 7.0  # Reflections tend to be more important

        # Generate keywords
        try:
            keywords = await llm.generate_keywords(insight, agent.name)
        except Exception:
            keywords = {agent.name.lower()}

        # Create thought node
        thought = await agent.memory.add_thought(
            subject=agent.name,
            predicate="reflects",
            object_=question,
            description=insight,
            keywords=keywords,
            poignancy=poignancy,
            embedding=embedding,
            filling=[mem.id for mem in relevant_memories],
            created_at=agent.current_time or datetime.now(),
        )

        return thought

    except Exception as e:
        logger.error(f"Failed to generate insight: {e}")
        return None


async def reflect_on_conversation(
    agent: "Agent",
    other_agent: str,
    conversation: list[tuple[str, str]],
) -> MemoryNode | None:
    """
    Generate a reflection on a conversation.

    Args:
        agent: The reflecting agent.
        other_agent: Name of the other agent in the conversation.
        conversation: List of (speaker, message) tuples.

    Returns:
        A thought node reflecting on the conversation.
    """
    llm = get_llm_service()
    embedding_service = get_embedding_service()

    # Format conversation
    conv_text = "\n".join(
        f"{speaker}: {message}" for speaker, message in conversation
    )

    prompt = f"""{agent.name} just had a conversation with {other_agent}:

{conv_text}

What is {agent.name}'s main takeaway or feeling about this conversation?
Provide a brief reflection (1-2 sentences).

Reflection:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
        )

        reflection = response.strip().strip('"')

        if not reflection:
            return None

        # Generate embedding
        try:
            embedding = await embedding_service.embed(reflection)
        except Exception:
            embedding = None

        # Generate keywords
        keywords = {agent.name.lower(), other_agent.lower()}

        # Create thought
        thought = await agent.memory.add_thought(
            subject=agent.name,
            predicate="thinks about conversation with",
            object_=other_agent,
            description=reflection,
            keywords=keywords,
            poignancy=6.0,
            embedding=embedding,
            created_at=agent.current_time or datetime.now(),
        )

        return thought

    except Exception as e:
        logger.error(f"Failed to reflect on conversation: {e}")
        return None
