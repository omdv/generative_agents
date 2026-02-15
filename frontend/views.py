"""Frontend views for the generative agents simulation."""

import json
import logging
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from core.simulation import SimulationManager, SimulationState

logger = logging.getLogger(__name__)


def landing(request: HttpRequest) -> HttpResponse:
    """Landing page."""
    return render(request, "landing/landing.html", {
        "title": "Generative Agents",
    })


def simulator_home(request: HttpRequest) -> HttpResponse:
    """Main simulator interface."""
    manager = SimulationManager.get_instance()

    # Get available worlds
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
            if matrix_dir.exists() and (matrix_dir / "maze_meta_info.json").exists():
                worlds.append(world_dir.name)

    # Get available personalities
    personalities = _get_available_personalities()

    context = {
        "title": "Generative Agents Simulator",
        "simulation_state": manager.state.value,
        "simulation_id": manager.simulation_id,
        "step": manager.step,
        "game_time": manager.game_time.isoformat() if manager.game_time else None,
        "agents": list(manager.agents.keys()),
        "worlds": worlds,
        "personalities": personalities,
    }

    return render(request, "home/home.html", context)


def demo(request: HttpRequest, sim_code: str, step: int, play_speed: int) -> HttpResponse:
    """Demo replay mode."""
    return render(request, "demo/demo.html", {
        "title": f"Demo - {sim_code}",
        "sim_code": sim_code,
        "step": step,
        "play_speed": play_speed,
    })


def replay(request: HttpRequest, sim_code: str, step: int) -> HttpResponse:
    """Replay mode."""
    return render(request, "demo/demo.html", {
        "title": f"Replay - {sim_code}",
        "sim_code": sim_code,
        "step": step,
        "play_speed": 1,
        "replay_mode": True,
    })


# -----------------
# Legacy API endpoints for compatibility
# -----------------


@csrf_exempt
def process_environment(request: HttpRequest) -> JsonResponse:
    """
    Process environment update from frontend.

    This is a legacy endpoint for compatibility with the original frontend.
    The frontend sends the current state and expects movement updates.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    manager = SimulationManager.get_instance()

    if manager.state == SimulationState.STOPPED:
        return JsonResponse({
            "status": "stopped",
            "message": "Simulation not running",
        })

    # Build response with current agent states
    persona_data = {}
    for name, agent in manager.agents.items():
        persona_data[name] = {
            "x": agent.position[0],
            "y": agent.position[1],
            "description": agent.scratch.action_description or "idle",
            "pronunciatio": agent.scratch.action_emoji or "",
            "chat": agent.scratch.chatting_with,
        }

    return JsonResponse({
        "status": "ok",
        "step": manager.step,
        "personas": persona_data,
    })


@csrf_exempt
def update_environment(request: HttpRequest) -> JsonResponse:
    """
    Send movement updates to frontend.

    Legacy endpoint that provides the next positions for agents.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    manager = SimulationManager.get_instance()

    if manager.state == SimulationState.STOPPED:
        return JsonResponse({
            "status": "stopped",
            "message": "Simulation not running",
        })

    # Build movement data
    movements = {}
    for name, agent in manager.agents.items():
        movements[name] = {
            "movement": list(agent.position),
            "pronunciatio": agent.scratch.action_emoji or "",
            "description": agent.scratch.action_description or "",
            "chat": None,  # Will be populated if in conversation
        }

        if agent.scratch.chatting_with:
            movements[name]["chat"] = [
                agent.name,
                agent.scratch.chatting_with,
                agent.scratch.chat_history[-1][1] if agent.scratch.chat_history else "",
            ]

    return JsonResponse({
        "status": "ok",
        "step": manager.step,
        "movements": movements,
    })


def _get_available_personalities() -> list[dict[str, Any]]:
    """Get available personality templates from storage."""
    storage_path = (
        settings.BASE_DIR
        / "environment"
        / "frontend_server"
        / "storage"
    )

    personalities = []

    for sim_dir in storage_path.iterdir():
        if sim_dir.is_dir() and sim_dir.name.startswith("base_"):
            personas_dir = sim_dir / "personas"
            if personas_dir.exists():
                for persona_dir in personas_dir.iterdir():
                    if persona_dir.is_dir():
                        scratch_file = persona_dir / "bootstrap_memory" / "scratch.json"
                        if scratch_file.exists():
                            try:
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
                                    })
                            except Exception as e:
                                logger.warning(f"Failed to load {scratch_file}: {e}")

    return personalities
