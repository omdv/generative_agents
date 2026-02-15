"""
Execute module - turns plans into concrete actions.

This module handles the execution of planned actions, including
pathfinding and generating action descriptions.
"""

import logging
from typing import TYPE_CHECKING

from core.agent import Action
from services.llm import get_llm_service

if TYPE_CHECKING:
    from core.agent import Agent
    from core.world import World

logger = logging.getLogger(__name__)


async def execute(
    agent: "Agent",
    world: "World",
    agents: dict[str, "Agent"],
    action_address: str,
) -> Action:
    """
    Execute the planned action.

    This function:
    1. Determines the target tile for the action
    2. Generates an action description and emoji
    3. Creates the path to the target
    4. Returns the complete action

    Args:
        agent: The executing agent.
        world: The world state.
        agents: All agents in the simulation.
        action_address: Where to perform the action.

    Returns:
        The action to execute.
    """
    # Get target tiles for this address
    target_tiles = world.get_address_tiles(action_address)

    if not target_tiles:
        # Try to find a partial match
        for addr in world.maze.address_tiles:
            if action_address in addr or addr in action_address:
                target_tiles = world.maze.address_tiles[addr]
                break

    # Pick the closest target tile
    if target_tiles:
        target = min(
            target_tiles,
            key=lambda t: abs(t[0] - agent.position[0]) + abs(t[1] - agent.position[1]),
        )
    else:
        # Stay in place if no valid target
        target = agent.position

    # Generate path to target
    path = world.find_path(agent.position, target)

    # Get action description from scratch
    description = agent.scratch.action_description or "doing something"
    duration = agent.scratch.action_duration

    # Generate emoji for the action
    emoji = await _generate_emoji(description)

    # Generate object interaction if applicable
    object_desc = None
    object_emoji = None
    object_event = None

    tile = world.maze.get_tile(target[0], target[1])
    if tile and tile.game_object:
        object_desc = await _generate_object_description(
            agent, description, tile.game_object
        )
        object_emoji = await _generate_emoji(object_desc)
        object_event = (
            f"{tile.world}:{tile.sector}:{tile.arena}:{tile.game_object}",
            "is being used by",
            agent.name,
        )

    action = Action(
        address=action_address,
        description=description,
        emoji=emoji,
        duration=duration,
        event=(agent.name, "is", description),
        object_description=object_desc,
        object_emoji=object_emoji,
        object_event=object_event,
        path=path,
    )

    logger.debug(f"{agent.name} executing: {description} at {action_address}")
    return action


async def _generate_emoji(description: str) -> str:
    """Generate an emoji to represent the action."""
    llm = get_llm_service()

    prompt = f"""What single emoji best represents this activity?

Activity: {description}

Respond with just one emoji:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=10,
        )

        # Extract first emoji-like character
        response = response.strip()
        if response:
            return response[0]

    except Exception as e:
        logger.warning(f"Failed to generate emoji: {e}")

    return "🔵"  # Default emoji


async def _generate_object_description(
    agent: "Agent",
    action: str,
    game_object: str,
) -> str:
    """Generate a description of how an object is being used."""
    llm = get_llm_service()

    prompt = f"""{agent.name} is {action} using a {game_object}.

Describe what the {game_object} is doing (from the object's perspective).
Keep it brief (5-10 words).

Example: "The stove is being used to cook dinner."

Description:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.5,
            max_tokens=30,
        )

        return response.strip().strip('"')

    except Exception as e:
        logger.warning(f"Failed to generate object description: {e}")

    return f"being used by {agent.name}"


def get_next_tile(agent: "Agent") -> tuple[int, int] | None:
    """
    Get the next tile the agent should move to.

    Returns:
        Next tile coordinates, or None if at destination.
    """
    if not agent.scratch.planned_path:
        return None

    # Pop the next tile from the path
    next_tile = agent.scratch.planned_path.pop(0)
    return next_tile


def update_agent_position(
    agent: "Agent",
    world: "World",
    new_position: tuple[int, int],
) -> None:
    """
    Update the agent's position in the world.

    Args:
        agent: The agent to move.
        world: The world state.
        new_position: New (x, y) position.
    """
    old_position = agent.position

    # Remove agent events from old position
    world.remove_agent_events(old_position[0], old_position[1], agent.name)

    # Update position
    agent.position = new_position

    # Add agent events to new position
    event = agent.scratch.action_event
    world.add_agent_event(
        new_position[0],
        new_position[1],
        agent.name,
        event[1] if len(event) > 1 else None,
        event[2] if len(event) > 2 else None,
        agent.scratch.action_description,
    )
