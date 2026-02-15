"""
Simulation Manager - coordinates the generative agents simulation.

This module provides the central SimulationManager class that
orchestrates all agents, manages time, and broadcasts state changes.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.agent import Agent, Action
from core.cognitive.execute import get_next_tile, update_agent_position
from core.personality import PersonalityConfig
from core.world import Maze, World

logger = logging.getLogger(__name__)


class SimulationState(str, Enum):
    """Possible states of the simulation."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class SimulationManager:
    """
    Singleton manager for the generative agents simulation.

    Handles:
    - Starting/stopping/pausing the simulation
    - Managing agents and the world
    - Running the simulation loop
    - Broadcasting state changes via WebSocket
    """

    _instance: "SimulationManager | None" = None

    def __new__(cls) -> "SimulationManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.state = SimulationState.STOPPED
        self.world: World | None = None
        self.agents: dict[str, Agent] = {}
        self.step = 0
        self.speed = 1.0  # Steps per second
        self.seconds_per_step = 300  # Game seconds per step (5 min = faster for testing)

        self.simulation_id: str | None = None
        self.simulation_name: str | None = None
        self.game_time: datetime | None = None

        # Callbacks for state changes
        self._on_update: list[Callable[[dict[str, Any]], None]] = []
        self._loop_task: asyncio.Task | None = None

        # Database path
        self._db_path: Path | None = None

    @classmethod
    def get_instance(cls) -> "SimulationManager":
        """Get the singleton instance."""
        return cls()

    def add_update_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback to be called on state updates."""
        self._on_update.append(callback)

    def remove_update_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove an update callback."""
        if callback in self._on_update:
            self._on_update.remove(callback)

    async def start(
        self,
        world_name: str,
        agent_configs: list[dict[str, Any]],
        simulation_name: str | None = None,
        start_time: datetime | None = None,
        db_path: str | Path | None = None,
    ) -> str:
        """
        Start a new simulation.

        Args:
            world_name: Name of the world to load.
            agent_configs: List of agent personality configurations.
            simulation_name: Optional name for this simulation.
            start_time: Starting game time.
            db_path: Path to the database file.

        Returns:
            Simulation ID.
        """
        if self.state == SimulationState.RUNNING:
            await self.stop()

        # Generate simulation ID
        import uuid
        self.simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        self.simulation_name = simulation_name or f"Simulation {self.simulation_id}"

        # Set up database
        if db_path:
            self._db_path = Path(db_path)
        else:
            from django.conf import settings
            self._db_path = settings.BASE_DIR / "db.sqlite3"

        # Load world
        from django.conf import settings
        maze_path = (
            settings.BASE_DIR
            / "environment"
            / "frontend_server"
            / "static_dirs"
            / "assets"
            / world_name
            / "matrix"
        )

        try:
            maze = Maze(world_name, maze_path)
            self.world = World(maze)
        except Exception as e:
            logger.error(f"Failed to load world: {e}")
            raise ValueError(f"Failed to load world '{world_name}': {e}")

        # Initialize agents
        self.agents = {}
        for config in agent_configs:
            personality = PersonalityConfig.from_dict(config)
            agent = Agent(
                personality=personality,
                db_path=self._db_path,
            )

            # Set initial position from spawn location or config
            if "position" in config:
                agent.position = tuple(config["position"])
            elif personality.living_area:
                spawn_tiles = self.world.get_address_tiles(personality.living_area)
                if spawn_tiles:
                    agent.position = next(iter(spawn_tiles))

            self.agents[agent.name] = agent

            # Add agent to world at initial position
            self.world.add_agent_event(
                agent.position[0],
                agent.position[1],
                agent.name,
                "is",
                "idle",
                f"{agent.name} is starting their day",
            )

            logger.info(f"Agent {agent.name} positioned at {agent.position}")

        # Initialize time
        self.game_time = start_time or datetime(2023, 2, 13, 0, 0, 0)
        self.step = 0

        # Update state
        self.state = SimulationState.RUNNING

        # Start simulation loop
        self._loop_task = asyncio.create_task(self._run_loop())

        await self._broadcast({
            "type": "simulation_started",
            "simulation_id": self.simulation_id,
            "world": world_name,
            "agents": list(self.agents.keys()),
        })

        logger.info(f"Started simulation {self.simulation_id} with {len(self.agents)} agents")
        return self.simulation_id

    async def stop(self) -> None:
        """Stop the current simulation."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        self.state = SimulationState.STOPPED

        await self._broadcast({
            "type": "simulation_stopped",
            "simulation_id": self.simulation_id,
            "step": self.step,
        })

        logger.info(f"Stopped simulation {self.simulation_id}")

    async def pause(self) -> None:
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED

            await self._broadcast({
                "type": "simulation_paused",
                "simulation_id": self.simulation_id,
                "step": self.step,
            })

            logger.info(f"Paused simulation at step {self.step}")

    async def resume(self) -> None:
        """Resume a paused simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING

            await self._broadcast({
                "type": "simulation_resumed",
                "simulation_id": self.simulation_id,
                "step": self.step,
            })

            logger.info(f"Resumed simulation at step {self.step}")

    async def step_once(self) -> dict[str, Any]:
        """
        Execute a single simulation step.

        Returns:
            State update dictionary.
        """
        if self.state == SimulationState.STOPPED:
            raise RuntimeError("Simulation is not started")

        return await self._execute_step()

    async def _run_loop(self) -> None:
        """Main simulation loop."""
        logger.info("Simulation loop started")
        while self.state in (SimulationState.RUNNING, SimulationState.PAUSED):
            if self.state == SimulationState.RUNNING:
                try:
                    await self._execute_step()
                    logger.debug(f"Step {self.step} completed")
                except Exception as e:
                    logger.error(f"Error in simulation step: {e}", exc_info=True)

            # Wait based on speed
            await asyncio.sleep(1.0 / self.speed)
        logger.info("Simulation loop ended")

    async def _execute_step(self) -> dict[str, Any]:
        """Execute a single simulation step."""
        self.step += 1

        # Advance game time
        if self.game_time:
            self.game_time += timedelta(seconds=self.seconds_per_step)

        # Process each agent
        agent_updates: dict[str, dict[str, Any]] = {}

        for agent_name, agent in self.agents.items():
            try:
                update = await self._process_agent(agent)
                agent_updates[agent_name] = update
            except Exception as e:
                logger.error(f"Error processing agent {agent_name}: {e}")
                agent_updates[agent_name] = {"error": str(e)}

        # Build state update
        state_update = {
            "type": "step",
            "step": self.step,
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "agents": agent_updates,
        }

        await self._broadcast(state_update)

        return state_update

    async def _process_agent(self, agent: Agent) -> dict[str, Any]:
        """Process a single agent for one step."""
        if not self.world:
            return {"error": "No world loaded"}

        # Track if this is the first step for this agent (for new day detection)
        first_step = agent.current_time is None

        # Update agent's current time BEFORE checking if action is finished
        agent.current_time = self.game_time

        # Log agent state every 10 steps
        if self.step % 10 == 0:
            logger.info(f"{agent.name}: pos={agent.position}, action={agent.scratch.action_description}, path_len={len(agent.scratch.planned_path)}, finished={agent.is_action_finished()}")

        # Update agent's position along path
        if agent.scratch.planned_path:
            next_tile = get_next_tile(agent)
            if next_tile:
                update_agent_position(agent, self.world, next_tile)
                logger.debug(f"{agent.name} moved to {next_tile}, {len(agent.scratch.planned_path)} steps remaining")

        # Run cognitive loop if action is finished
        if agent.is_action_finished():
            logger.info(f"{agent.name} action finished, planning next action...")
            action = await agent.cognitive_loop(
                self.world,
                self.agents,
                self.game_time or datetime.now(),
            )

            if action:
                agent.scratch.planned_path = action.path
                agent.scratch.path_set = True
                # Also update scratch with action details
                agent.scratch.action_emoji = action.emoji
                agent.scratch.action_description = action.description
                agent.scratch.action_address = action.address
                logger.info(f"{agent.name} new action: {action.emoji} {action.description} at {action.address} (path: {len(action.path)} steps)")

        # Decrease chatting buffer
        for name in list(agent.scratch.chatting_buffer.keys()):
            agent.scratch.chatting_buffer[name] -= 1
            if agent.scratch.chatting_buffer[name] <= 0:
                del agent.scratch.chatting_buffer[name]

        return {
            "position": list(agent.position),
            "action": agent.scratch.action_description,
            "emoji": agent.scratch.action_emoji,
            "address": agent.scratch.action_address,
            "chatting_with": agent.scratch.chatting_with,
        }

    async def _broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all WebSocket clients via Channels."""
        try:
            from channels.layers import get_channel_layer
            channel_layer = get_channel_layer()
            if channel_layer:
                await channel_layer.group_send(
                    "simulation",
                    {
                        "type": "simulation_update",
                        "data": event,
                    }
                )
                if self.step % 10 == 0:
                    logger.info(f"Broadcast step {self.step} to WebSocket group")
            else:
                logger.warning("No channel layer available for broadcast")
        except Exception as e:
            logger.error(f"Error broadcasting: {e}", exc_info=True)

        # Also call any registered callbacks
        for callback in self._on_update:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")

    def get_state(self) -> dict[str, Any]:
        """Get the current simulation state."""
        agent_states = {}
        for name, agent in self.agents.items():
            agent_states[name] = {
                "position": list(agent.position),
                "action": agent.scratch.action_description,
                "emoji": agent.scratch.action_emoji,
                "address": agent.scratch.action_address,
                "chatting_with": agent.scratch.chatting_with,
            }

        return {
            "simulation_id": self.simulation_id,
            "simulation_name": self.simulation_name,
            "state": self.state.value,
            "step": self.step,
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "speed": self.speed,
            "world": self.world.to_dict() if self.world else None,
            "agents": agent_states,
        }

    def set_speed(self, speed: float) -> None:
        """Set the simulation speed (steps per second)."""
        self.speed = max(0.1, min(10.0, speed))

    async def save_state(self, path: str | Path) -> None:
        """Save the current simulation state to a file."""
        path = Path(path)
        state = {
            "simulation_id": self.simulation_id,
            "simulation_name": self.simulation_name,
            "step": self.step,
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "world_name": self.world.name if self.world else None,
            "agents": {
                name: agent.to_dict()
                for name, agent in self.agents.items()
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved simulation state to {path}")

    async def load_state(self, path: str | Path) -> None:
        """Load simulation state from a file."""
        path = Path(path)

        with open(path) as f:
            state = json.load(f)

        self.simulation_id = state["simulation_id"]
        self.simulation_name = state["simulation_name"]
        self.step = state["step"]
        self.game_time = (
            datetime.fromisoformat(state["game_time"])
            if state["game_time"]
            else None
        )

        # Load world
        if state["world_name"]:
            from django.conf import settings
            maze_path = (
                settings.BASE_DIR
                / "environment"
                / "frontend_server"
                / "static_dirs"
                / "assets"
                / state["world_name"]
                / "matrix"
            )
            maze = Maze(state["world_name"], maze_path)
            self.world = World(maze)

        # Load agents
        self.agents = {}
        for name, agent_data in state["agents"].items():
            agent = Agent.from_dict(agent_data, self._db_path or "db.sqlite3")
            self.agents[name] = agent

        logger.info(f"Loaded simulation state from {path}")


# Module-level convenience function
def get_simulation_manager() -> SimulationManager:
    """Get the global simulation manager instance."""
    return SimulationManager.get_instance()
