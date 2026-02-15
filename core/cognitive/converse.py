"""
Converse module - handles agent conversations.

This module manages dialogue between agents, including
deciding when to talk, generating responses, and
handling conversation flow.
"""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from core.memory import MemoryNode
from services.embeddings import get_embedding_service
from services.llm import get_llm_service

if TYPE_CHECKING:
    from core.agent import Agent

logger = logging.getLogger(__name__)


async def should_converse(
    agent: "Agent",
    other_agent: "Agent",
) -> bool:
    """
    Determine if two agents should start a conversation.

    Args:
        agent: The initiating agent.
        other_agent: The potential conversation partner.

    Returns:
        True if they should converse.
    """
    # Check if already in conversation
    if agent.scratch.chatting_with or other_agent.scratch.chatting_with:
        return False

    # Check buffer - don't talk to same person repeatedly
    if other_agent.name in agent.scratch.chatting_buffer:
        buffer_val = agent.scratch.chatting_buffer[other_agent.name]
        if buffer_val > 0:
            return False

    llm = get_llm_service()

    # Get last conversation with this person
    last_chat = agent.memory.get_last_chat_with(other_agent.name)
    last_chat_info = ""
    if last_chat:
        last_chat_info = f"\nTheir last conversation was about: {last_chat.description}"

    prompt = f"""{agent.name} is {agent.scratch.action_description or 'doing something'}.
{other_agent.name} is {other_agent.scratch.action_description or 'doing something'}.
{last_chat_info}

Would {agent.name} initiate a conversation with {other_agent.name}?
Consider their current activities and relationship.

Answer with just "yes" or "no":"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.5,
            max_tokens=10,
        )

        return response.strip().lower().startswith("yes")

    except Exception as e:
        logger.warning(f"Failed to determine conversation: {e}")
        return False


async def open_conversation(
    agent: "Agent",
    other_agent: "Agent",
) -> str:
    """
    Generate an opening line for a conversation.

    Args:
        agent: The agent starting the conversation.
        other_agent: The other agent.

    Returns:
        The opening line.
    """
    llm = get_llm_service()

    # Get relevant memories about the other agent
    memories = agent.memory.get_events_by_keywords([other_agent.name.lower()])
    memory_context = ""
    if memories:
        memory_context = "\n".join(
            f"- {mem.description}" for mem in list(memories)[:5]
        )
        memory_context = f"\n{agent.name}'s memories of {other_agent.name}:\n{memory_context}"

    prompt = f"""{agent.get_identity_summary()}

{agent.name} is currently: {agent.scratch.action_description or 'doing something'}
{other_agent.name} is currently: {other_agent.scratch.action_description or 'doing something'}
{memory_context}

{agent.name} wants to start a conversation with {other_agent.name}.
What would {agent.name} say to initiate the conversation?

{agent.name}:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.8,
            max_tokens=100,
        )

        return response.strip().strip('"')

    except Exception as e:
        logger.error(f"Failed to generate opening: {e}")
        return f"Hi {other_agent.personality.first_name}!"


async def generate_response(
    agent: "Agent",
    other_agent: "Agent",
    conversation: list[tuple[str, str]],
) -> str:
    """
    Generate a response in an ongoing conversation.

    Args:
        agent: The responding agent.
        other_agent: The other agent.
        conversation: The conversation so far.

    Returns:
        The agent's response.
    """
    llm = get_llm_service()

    # Format conversation history
    conv_text = "\n".join(
        f"{speaker}: {message}" for speaker, message in conversation
    )

    prompt = f"""{agent.get_identity_summary()}

{agent.name} is having a conversation with {other_agent.name}.

Conversation so far:
{conv_text}

What does {agent.name} say next? Keep it natural and in character.

{agent.name}:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.8,
            max_tokens=150,
        )

        return response.strip().strip('"')

    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "I see..."


async def generate_conversation(
    agent: "Agent",
    other_agent: "Agent",
    max_turns: int = 8,
) -> list[tuple[str, str]]:
    """
    Generate a complete conversation between two agents.

    Args:
        agent: The first agent (initiator).
        other_agent: The second agent.
        max_turns: Maximum conversation turns.

    Returns:
        List of (speaker, message) tuples.
    """
    conversation: list[tuple[str, str]] = []

    # Opening
    opening = await open_conversation(agent, other_agent)
    conversation.append((agent.name, opening))

    # Generate conversation turns
    current_speaker = other_agent
    other_speaker = agent

    for _ in range(max_turns - 1):
        response = await generate_response(
            current_speaker,
            other_speaker,
            conversation,
        )
        conversation.append((current_speaker.name, response))

        # Check if conversation should end
        if await _should_end_conversation(conversation):
            break

        # Swap speakers
        current_speaker, other_speaker = other_speaker, current_speaker

    return conversation


async def _should_end_conversation(
    conversation: list[tuple[str, str]],
) -> bool:
    """Determine if a conversation should end naturally."""
    if len(conversation) < 3:
        return False

    last_message = conversation[-1][1].lower()

    # Check for farewell indicators
    farewell_words = ["goodbye", "bye", "see you", "later", "gotta go", "nice talking"]
    return any(word in last_message for word in farewell_words)


async def end_conversation(
    agent: "Agent",
    other_agent: "Agent",
    conversation: list[tuple[str, str]],
) -> None:
    """
    End a conversation and create memories.

    Args:
        agent: One of the agents.
        other_agent: The other agent.
        conversation: The conversation that occurred.
    """
    embedding_service = get_embedding_service()

    # Create conversation summary
    conv_text = " ".join(msg for _, msg in conversation)

    try:
        embedding = await embedding_service.embed(conv_text)
    except Exception:
        embedding = None

    # Generate summary
    llm = get_llm_service()

    prompt = f"""Summarize this conversation in one sentence:

{chr(10).join(f'{s}: {m}' for s, m in conversation)}

Summary:"""

    try:
        summary = await llm.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=50,
        )
        summary = summary.strip().strip('"')
    except Exception:
        summary = f"Had a conversation with {other_agent.name}"

    # Create chat memories for both agents
    keywords = {agent.name.lower(), other_agent.name.lower()}

    await agent.memory.add_chat(
        subject=agent.name,
        predicate="talked to",
        object_=other_agent.name,
        description=summary,
        keywords=keywords,
        poignancy=6.0,
        embedding=embedding,
        filling=conversation,
        created_at=agent.current_time or datetime.now(),
    )

    await other_agent.memory.add_chat(
        subject=other_agent.name,
        predicate="talked to",
        object_=agent.name,
        description=summary,
        keywords=keywords,
        poignancy=6.0,
        embedding=embedding,
        filling=conversation,
        created_at=other_agent.current_time or datetime.now(),
    )

    # Update chatting buffer
    agent.scratch.chatting_buffer[other_agent.name] = agent.personality.retention
    other_agent.scratch.chatting_buffer[agent.name] = other_agent.personality.retention

    # Clear conversation state
    agent.scratch.chatting_with = None
    agent.scratch.chat_history = []
    other_agent.scratch.chatting_with = None
    other_agent.scratch.chat_history = []

    logger.info(f"Conversation ended between {agent.name} and {other_agent.name}")
