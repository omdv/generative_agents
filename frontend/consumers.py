"""WebSocket consumers for real-time simulation updates."""

import json
import logging
from typing import Any

from channels.generic.websocket import AsyncWebsocketConsumer

from core.simulation import SimulationManager

logger = logging.getLogger(__name__)


class SimulationConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time simulation updates.

    Clients connect to /ws/simulation/ to receive live updates
    about agent positions, actions, and simulation state.
    """

    GROUP_NAME = "simulation"

    async def connect(self) -> None:
        """Handle WebSocket connection."""
        # Add to simulation group
        await self.channel_layer.group_add(
            self.GROUP_NAME,
            self.channel_name,
        )

        await self.accept()

        # Send current state immediately
        manager = SimulationManager.get_instance()
        await self.send_json({
            "type": "connection_established",
            "state": manager.get_state(),
        })

        # Register callback for updates
        manager.add_update_callback(self._on_simulation_update)

        logger.info(f"WebSocket client connected: {self.channel_name}")

    async def disconnect(self, close_code: int) -> None:
        """Handle WebSocket disconnection."""
        # Remove from simulation group
        await self.channel_layer.group_discard(
            self.GROUP_NAME,
            self.channel_name,
        )

        # Remove callback
        manager = SimulationManager.get_instance()
        manager.remove_update_callback(self._on_simulation_update)

        logger.info(f"WebSocket client disconnected: {self.channel_name}")

    async def receive(self, text_data: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            message_type = data.get("type", "")

            if message_type == "ping":
                await self.send_json({"type": "pong"})

            elif message_type == "get_state":
                manager = SimulationManager.get_instance()
                await self.send_json({
                    "type": "state",
                    "state": manager.get_state(),
                })

            elif message_type == "get_agent":
                agent_name = data.get("agent")
                if agent_name:
                    manager = SimulationManager.get_instance()
                    if agent_name in manager.agents:
                        agent = manager.agents[agent_name]
                        await self.send_json({
                            "type": "agent_detail",
                            "agent": {
                                "name": agent.name,
                                "position": list(agent.position),
                                "action": agent.scratch.action_description,
                                "emoji": agent.scratch.action_emoji,
                                "chatting_with": agent.scratch.chatting_with,
                            },
                        })

            elif message_type == "set_speed":
                speed = data.get("speed", 1.0)
                manager = SimulationManager.get_instance()
                manager.set_speed(float(speed))
                await self.send_json({
                    "type": "speed_set",
                    "speed": manager.speed,
                })

        except json.JSONDecodeError:
            await self.send_json({
                "type": "error",
                "message": "Invalid JSON",
            })
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_json({
                "type": "error",
                "message": str(e),
            })

    async def send_json(self, data: dict[str, Any]) -> None:
        """Send JSON data to the WebSocket."""
        await self.send(text_data=json.dumps(data))

    def _on_simulation_update(self, event: dict[str, Any]) -> None:
        """Callback for simulation updates (called synchronously)."""
        import asyncio

        # Schedule the async send
        asyncio.create_task(self._send_update(event))

    async def _send_update(self, event: dict[str, Any]) -> None:
        """Send simulation update to the client."""
        await self.send_json(event)

    # Group message handlers

    async def simulation_update(self, event: dict[str, Any]) -> None:
        """Handle simulation update messages from the channel layer."""
        logger.debug(f"Consumer received update, sending to client")
        await self.send_json(event["data"])

    async def simulation_started(self, event: dict[str, Any]) -> None:
        """Handle simulation started event."""
        await self.send_json({
            "type": "simulation_started",
            **event.get("data", {}),
        })

    async def simulation_stopped(self, event: dict[str, Any]) -> None:
        """Handle simulation stopped event."""
        await self.send_json({
            "type": "simulation_stopped",
            **event.get("data", {}),
        })

    async def simulation_paused(self, event: dict[str, Any]) -> None:
        """Handle simulation paused event."""
        await self.send_json({
            "type": "simulation_paused",
            **event.get("data", {}),
        })

    async def simulation_resumed(self, event: dict[str, Any]) -> None:
        """Handle simulation resumed event."""
        await self.send_json({
            "type": "simulation_resumed",
            **event.get("data", {}),
        })
