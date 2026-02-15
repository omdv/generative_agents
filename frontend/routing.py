"""WebSocket URL routing for frontend."""

from django.urls import path

from frontend.consumers import SimulationConsumer

websocket_urlpatterns = [
    path("ws/simulation/", SimulationConsumer.as_asgi()),
]
