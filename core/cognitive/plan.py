"""
Plan module - decides what the agent should do.

This module handles both long-term planning (daily schedule) and
short-term planning (what to do right now).
"""

import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from services.llm import get_llm_service

if TYPE_CHECKING:
    from core.agent import Agent
    from core.world import World

logger = logging.getLogger(__name__)


async def plan(
    agent: "Agent",
    world: "World",
    agents: dict[str, "Agent"],
    new_day: str | bool,
    retrieved: dict[str, dict[str, Any]],
) -> str | None:
    """
    Determine what the agent should do next.

    This function handles:
    1. Creating a daily plan if it's a new day
    2. Decomposing hourly plans into finer-grained actions
    3. Determining when to start the next action
    4. Deciding on interaction with other agents

    Args:
        agent: The planning agent.
        world: The world state.
        agents: All agents in the simulation.
        new_day: "First day", "New day", or False.
        retrieved: Retrieved memories for context.

    Returns:
        The target action address, or None if no action needed.
    """
    # Generate daily plan if new day OR if schedule is empty (first run)
    if new_day or not agent.scratch.daily_schedule:
        await _generate_daily_plan(agent, new_day == "First day" or not agent.scratch.daily_schedule)

    # Check if current action is finished
    if not agent.is_action_finished():
        return agent.scratch.action_address

    # Decompose current hour if needed
    await _decompose_hourly_plan(agent)

    # Determine next action
    action_address = await _determine_next_action(agent, world, agents, retrieved)

    return action_address


async def _generate_daily_plan(agent: "Agent", first_day: bool) -> None:
    """Generate a daily schedule for the agent."""
    llm = get_llm_service()

    # Build the prompt
    identity = agent.get_identity_summary()

    prompt = f"""Here is a brief description of {agent.name}:
{identity}

Today is {agent.current_time.strftime('%A %B %d, %Y') if agent.current_time else 'a new day'}.

What would {agent.name}'s daily schedule look like today?
Create a schedule from wake up to sleep with activities and approximate durations.

Format each line as: [Activity] (duration in minutes)
Example:
- Wake up and morning routine (60)
- Have breakfast (30)
- Work on project (180)
- Lunch break (45)
- Continue work (120)
- Exercise (60)
- Dinner (45)
- Relaxation and hobbies (90)
- Get ready for bed (30)
- Sleep (480)

{agent.name}'s schedule for today:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.8,
            max_tokens=500,
        )

        # Parse the schedule
        schedule = _parse_schedule(response)

        if schedule:
            agent.scratch.daily_schedule = schedule
            agent.scratch.daily_schedule_hourly = schedule.copy()
            logger.info(f"{agent.name} created daily schedule with {len(schedule)} items")
        else:
            # Default schedule
            agent.scratch.daily_schedule = _default_schedule()
            agent.scratch.daily_schedule_hourly = agent.scratch.daily_schedule.copy()

    except Exception as e:
        logger.error(f"Failed to generate daily plan: {e}")
        agent.scratch.daily_schedule = _default_schedule()
        agent.scratch.daily_schedule_hourly = agent.scratch.daily_schedule.copy()


def _parse_schedule(text: str) -> list[tuple[str, int]]:
    """Parse a schedule from LLM response."""
    schedule: list[tuple[str, int]] = []

    # Match patterns like "- Activity (60)" or "Activity (60 minutes)"
    pattern = r"[-•]?\s*(.+?)\s*\((\d+)(?:\s*(?:min(?:ute)?s?))?\)"

    for match in re.finditer(pattern, text, re.IGNORECASE):
        activity = match.group(1).strip()
        duration = int(match.group(2))

        # Clean up activity text
        activity = activity.strip("- ").strip()

        if activity and duration > 0:
            schedule.append((activity, duration))

    return schedule


def _default_schedule() -> list[tuple[str, int]]:
    """Return a default daily schedule."""
    return [
        ("sleeping", 420),  # 7 hours
        ("waking up and morning routine", 60),
        ("breakfast", 30),
        ("working", 180),
        ("lunch break", 60),
        ("working", 180),
        ("relaxation", 90),
        ("dinner", 45),
        ("evening activities", 120),
        ("getting ready for bed", 30),
        ("sleeping", 225),  # Until midnight
    ]


async def _decompose_hourly_plan(agent: "Agent") -> None:
    """Decompose the current hourly plan into finer-grained actions."""
    if not agent.current_time:
        return

    schedule_index = agent.get_daily_schedule_index()

    if schedule_index >= len(agent.scratch.daily_schedule):
        return

    current_task, duration = agent.scratch.daily_schedule[schedule_index]

    # Skip decomposition for certain activities
    skip_decomposition = ["sleeping", "sleep", "nap"]
    if any(skip in current_task.lower() for skip in skip_decomposition):
        return

    # Check if already decomposed
    if duration <= 30:
        return

    llm = get_llm_service()

    prompt = f"""{agent.name} is planning to: {current_task}
This activity is scheduled to last {duration} minutes.

Break this down into smaller 5-30 minute subtasks.
Format: [Subtask] (duration in minutes)

Subtasks:"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.7,
            max_tokens=300,
        )

        subtasks = _parse_schedule(response)

        if subtasks:
            # Replace the current task with subtasks
            new_schedule = (
                agent.scratch.daily_schedule[:schedule_index]
                + subtasks
                + agent.scratch.daily_schedule[schedule_index + 1 :]
            )
            agent.scratch.daily_schedule = new_schedule

    except Exception as e:
        logger.warning(f"Failed to decompose hourly plan: {e}")


async def _determine_next_action(
    agent: "Agent",
    world: "World",
    agents: dict[str, "Agent"],
    retrieved: dict[str, dict[str, Any]],
) -> str | None:
    """Determine the next action based on schedule and context."""
    if not agent.current_time:
        logger.warning(f"{agent.name}: no current_time, cannot determine action")
        return None

    schedule_index = agent.get_daily_schedule_index()
    logger.debug(f"{agent.name}: schedule_index={schedule_index}, schedule_len={len(agent.scratch.daily_schedule)}")

    if schedule_index >= len(agent.scratch.daily_schedule):
        logger.warning(f"{agent.name}: schedule exhausted")
        return None

    current_task, duration = agent.scratch.daily_schedule[schedule_index]
    logger.debug(f"{agent.name}: current_task='{current_task}', duration={duration}")

    # Determine where to perform this action
    action_address = await _determine_action_location(agent, world, current_task)
    logger.info(f"{agent.name}: action='{current_task}' at '{action_address}' for {duration}min")

    if action_address:
        # Set up the action in scratch
        agent.scratch.action_address = action_address
        agent.scratch.action_duration = duration
        agent.scratch.action_description = current_task
        agent.scratch.action_event = (agent.name, "is", current_task)
        agent.scratch.action_start_time = agent.current_time.isoformat()

    return action_address


async def _determine_action_location(
    agent: "Agent",
    world: "World",
    task: str,
) -> str | None:
    """Determine where in the world to perform an action."""
    llm = get_llm_service()

    # Get available locations from spatial memory
    available_locations = list(world.maze.address_tiles.keys())

    # Filter to relevant locations
    relevant_locations = [
        loc for loc in available_locations
        if not loc.startswith("<spawn_loc>")
    ][:20]  # Limit for prompt size

    if not relevant_locations:
        return agent.personality.living_area

    prompt = f"""{agent.name} wants to: {task}

Available locations:
{chr(10).join(f'- {loc}' for loc in relevant_locations)}

{agent.name}'s home is: {agent.personality.living_area}

Which location is most appropriate for this activity?
Respond with just the location path (e.g., "the Ville:cafe:counter"):"""

    try:
        response = await llm.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=100,
        )

        # Clean and validate response
        location = response.strip().strip('"').strip("'")

        # Check if it's a valid location
        if location in world.maze.address_tiles:
            return location

        # Try partial match
        for loc in relevant_locations:
            if location.lower() in loc.lower() or loc.lower() in location.lower():
                return loc

    except Exception as e:
        logger.warning(f"Failed to determine action location: {e}")

    # Default to living area
    return agent.personality.living_area
