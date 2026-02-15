"""URL configuration for the API."""

from django.urls import path

from api import views

urlpatterns = [
    # Simulation control
    path("simulation/start", views.simulation_start, name="simulation_start"),
    path("simulation/stop", views.simulation_stop, name="simulation_stop"),
    path("simulation/pause", views.simulation_pause, name="simulation_pause"),
    path("simulation/resume", views.simulation_resume, name="simulation_resume"),
    path("simulation/step", views.simulation_step, name="simulation_step"),
    path("simulation/state", views.simulation_state, name="simulation_state"),
    path("simulation/speed", views.simulation_speed, name="simulation_speed"),

    # Agents
    path("agents/", views.agents_list, name="agents_list"),
    path("agents/<str:name>/", views.agent_detail, name="agent_detail"),
    path("agents/<str:name>/memory", views.agent_memory, name="agent_memory"),

    # World
    path("world/", views.world_state, name="world_state"),
    path("worlds/", views.worlds_list, name="worlds_list"),

    # Personalities
    path("personalities/", views.personalities_list, name="personalities_list"),
]
