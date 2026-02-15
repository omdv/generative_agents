"""API views for simulation control and data access."""

import json
import logging
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from core.simulation import SimulationManager, SimulationState

logger = logging.getLogger(__name__)


def get_json_body(request) -> dict[str, Any]:
    """Parse JSON body from request."""
    if request.body:
        return json.loads(request.body)
    return {}


# -----------------
# Simulation Control
# -----------------


@csrf_exempt
@require_POST
async def simulation_start(request) -> JsonResponse:
    """
    Start a new simulation.

    POST /api/simulation/start
    Body: {
        "world": "the_ville",
        "agents": [
            {"name": "John Doe", "age": 30, "innate": "curious", ...},
            ...
        ],
        "name": "My Simulation",  // optional
        "start_time": "2023-02-13T00:00:00"  // optional
    }
    """
    try:
        body = get_json_body(request)

        world_name = body.get("world", "the_ville")
        agent_configs = body.get("agents", [])
        simulation_name = body.get("name")
        start_time = body.get("start_time")

        if start_time:
            from datetime import datetime
            start_time = datetime.fromisoformat(start_time)

        manager = SimulationManager.get_instance()
        sim_id = await manager.start(
            world_name=world_name,
            agent_configs=agent_configs,
            simulation_name=simulation_name,
            start_time=start_time,
        )

        return JsonResponse({
            "status": "ok",
            "simulation_id": sim_id,
            "message": f"Started simulation with {len(agent_configs)} agents",
        })

    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


@csrf_exempt
@require_POST
async def simulation_stop(request) -> JsonResponse:
    """Stop the current simulation."""
    try:
        manager = SimulationManager.get_instance()
        await manager.stop()

        return JsonResponse({
            "status": "ok",
            "message": "Simulation stopped",
        })

    except Exception as e:
        logger.error(f"Failed to stop simulation: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


@csrf_exempt
@require_POST
async def simulation_pause(request) -> JsonResponse:
    """Pause the current simulation."""
    try:
        manager = SimulationManager.get_instance()
        await manager.pause()

        return JsonResponse({
            "status": "ok",
            "message": "Simulation paused",
        })

    except Exception as e:
        logger.error(f"Failed to pause simulation: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


@csrf_exempt
@require_POST
async def simulation_resume(request) -> JsonResponse:
    """Resume a paused simulation."""
    try:
        manager = SimulationManager.get_instance()
        await manager.resume()

        return JsonResponse({
            "status": "ok",
            "message": "Simulation resumed",
        })

    except Exception as e:
        logger.error(f"Failed to resume simulation: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


@csrf_exempt
@require_POST
async def simulation_step(request) -> JsonResponse:
    """Execute a single simulation step."""
    try:
        manager = SimulationManager.get_instance()

        if manager.state == SimulationState.STOPPED:
            return JsonResponse({
                "status": "error",
                "message": "Simulation is not running",
            }, status=400)

        state = await manager.step_once()

        return JsonResponse({
            "status": "ok",
            "state": state,
        })

    except Exception as e:
        logger.error(f"Failed to step simulation: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


@require_GET
def simulation_state(request) -> JsonResponse:
    """Get the current simulation state."""
    manager = SimulationManager.get_instance()
    return JsonResponse(manager.get_state())


@csrf_exempt
@require_POST
def simulation_speed(request) -> JsonResponse:
    """
    Set simulation speed.

    POST /api/simulation/speed
    Body: {"speed": 2.0}
    """
    try:
        body = get_json_body(request)
        speed = float(body.get("speed", 1.0))

        manager = SimulationManager.get_instance()
        manager.set_speed(speed)

        return JsonResponse({
            "status": "ok",
            "speed": manager.speed,
        })

    except Exception as e:
        logger.error(f"Failed to set speed: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e),
        }, status=400)


# -----------------
# Agents
# -----------------


@require_GET
def agents_list(request) -> JsonResponse:
    """List all agents in the simulation."""
    manager = SimulationManager.get_instance()

    agents = []
    for name, agent in manager.agents.items():
        agents.append({
            "name": name,
            "position": list(agent.position),
            "action": agent.scratch.action_description,
            "emoji": agent.scratch.action_emoji,
        })

    return JsonResponse({
        "agents": agents,
        "count": len(agents),
    })


@require_GET
def agent_detail(request, name: str) -> JsonResponse:
    """Get details for a specific agent."""
    manager = SimulationManager.get_instance()

    if name not in manager.agents:
        return JsonResponse({
            "status": "error",
            "message": f"Agent '{name}' not found",
        }, status=404)

    agent = manager.agents[name]

    return JsonResponse({
        "name": agent.name,
        "personality": agent.personality.to_dict(),
        "position": list(agent.position),
        "scratch": agent.scratch.to_dict(),
        "current_time": agent.current_time.isoformat() if agent.current_time else None,
    })


@require_GET
def agent_memory(request, name: str) -> JsonResponse:
    """Get recent memories for an agent."""
    manager = SimulationManager.get_instance()

    if name not in manager.agents:
        return JsonResponse({
            "status": "error",
            "message": f"Agent '{name}' not found",
        }, status=404)

    agent = manager.agents[name]

    limit = int(request.GET.get("limit", 20))
    memory_type = request.GET.get("type")

    memories = []

    if memory_type in (None, "event"):
        for mem in agent.memory.get_recent_events(limit):
            memories.append({
                "id": mem.id,
                "type": "event",
                "description": mem.description,
                "poignancy": mem.poignancy,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })

    if memory_type in (None, "thought"):
        for mem in agent.memory.get_recent_thoughts(limit):
            memories.append({
                "id": mem.id,
                "type": "thought",
                "description": mem.description,
                "poignancy": mem.poignancy,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })

    if memory_type in (None, "chat"):
        for mem in agent.memory.get_recent_chats(limit):
            memories.append({
                "id": mem.id,
                "type": "chat",
                "description": mem.description,
                "poignancy": mem.poignancy,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })

    # Sort by creation time
    memories.sort(key=lambda m: m.get("created_at") or "", reverse=True)

    return JsonResponse({
        "agent": name,
        "memories": memories[:limit],
        "count": len(memories[:limit]),
    })


# -----------------
# World
# -----------------


@require_GET
def world_state(request) -> JsonResponse:
    """Get the current world state."""
    manager = SimulationManager.get_instance()

    if not manager.world:
        return JsonResponse({
            "status": "error",
            "message": "No world loaded",
        }, status=400)

    return JsonResponse(manager.world.to_dict())


@require_GET
def worlds_list(request) -> JsonResponse:
    """List available worlds."""
    assets_path = (
        settings.BASE_DIR
        / "environment"
        / "frontend_server"
        / "static_dirs"
        / "assets"
    )

    worlds = []

    for world_dir in assets_path.iterdir():
        if world_dir.is_dir():
            matrix_dir = world_dir / "matrix"
            if matrix_dir.exists():
                meta_file = matrix_dir / "maze_meta_info.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                        worlds.append({
                            "name": world_dir.name,
                            "display_name": meta.get("world_name", world_dir.name),
                            "width": meta.get("maze_width"),
                            "height": meta.get("maze_height"),
                        })

    return JsonResponse({
        "worlds": worlds,
        "count": len(worlds),
    })


# -----------------
# Personalities
# -----------------


@require_GET
def personalities_list(request) -> JsonResponse:
    """List available personality templates."""
    # Load from base simulation storage
    storage_path = (
        settings.BASE_DIR
        / "environment"
        / "frontend_server"
        / "storage"
    )

    personalities = []

    # Look for persona data in base simulations
    for sim_dir in storage_path.iterdir():
        if sim_dir.is_dir() and sim_dir.name.startswith("base_"):
            personas_dir = sim_dir / "personas"
            if personas_dir.exists():
                for persona_dir in personas_dir.iterdir():
                    if persona_dir.is_dir():
                        scratch_file = persona_dir / "bootstrap_memory" / "scratch.json"
                        if scratch_file.exists():
                            with open(scratch_file) as f:
                                scratch = json.load(f)
                                personalities.append({
                                    "name": scratch.get("name", persona_dir.name),
                                    "age": scratch.get("age"),
                                    "innate": scratch.get("innate"),
                                    "learned": scratch.get("learned"),
                                    "currently": scratch.get("currently"),
                                    "lifestyle": scratch.get("lifestyle"),
                                    "living_area": scratch.get("living_area"),
                                    "source": sim_dir.name,
                                })

    return JsonResponse({
        "personalities": personalities,
        "count": len(personalities),
    })
