"""URL configuration for frontend views."""

from django.urls import path

from frontend import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("simulator_home", views.simulator_home, name="simulator_home"),
    path("demo/<str:sim_code>/<int:step>/<int:play_speed>/", views.demo, name="demo"),
    path("replay/<str:sim_code>/<int:step>/", views.replay, name="replay"),

    # Legacy API endpoints (for compatibility with old frontend)
    path("process_environment/", views.process_environment, name="process_environment"),
    path("update_environment/", views.update_environment, name="update_environment"),
]
