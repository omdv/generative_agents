# Generative Agents

A simulation of autonomous AI agents that live, plan, and interact in a virtual town. Agents have memories, daily schedules, and can perceive and react to their environment and each other.

Based on the research paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442) by Park et al.

## Features

- **Autonomous agents** with personalities, memories, and daily routines
- **Cognitive architecture**: perceive, retrieve, plan, reflect, execute
- **Tile-based world** with A* pathfinding
- **Real-time visualization** using Phaser 3
- **WebSocket updates** for live agent movement
- **LLM-powered** decision making via OpenRouter API

## Quick Start

```bash
# Install dependencies
pip install -e .

# Optional: Set up LLM API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```

Open http://localhost:8000/simulator_home in your browser.

## Controls

| Input | Action |
|-------|--------|
| WASD / Arrow keys | Pan camera |
| Mouse drag | Pan camera |
| Scroll wheel | Zoom in/out |
| Click agent name | Center camera on agent |

## Project Structure

```
generative_agents/
├── config/           # Django settings
├── core/             # Simulation engine
│   ├── agent.py      # Agent class with cognitive loop
│   ├── memory.py     # Memory system with retrieval
│   ├── simulation.py # Simulation manager
│   ├── world.py      # Maze and A* pathfinding
│   └── cognitive/    # Cognitive modules
│       ├── perceive.py
│       ├── retrieve.py
│       ├── plan.py
│       ├── reflect.py
│       └── execute.py
├── services/         # LLM and embedding services
├── api/              # REST API endpoints
└── frontend/         # Django templates and WebSocket
```

## How It Works

Each simulation step, agents:

1. **Perceive** - Notice events within their vision radius
2. **Retrieve** - Recall relevant memories based on perceived events
3. **Plan** - Decide what to do next based on their schedule and context
4. **Reflect** - Form higher-level insights from accumulated experiences
5. **Execute** - Move toward their destination via pathfinding

## Requirements

- Python 3.12+
- Django 5.0+
- Channels (WebSocket support)
- OpenRouter API key (optional, uses mock responses without)

## Credits

The cognitive architecture (perceive, retrieve, plan, reflect, execute) is adapted from the original [Generative Agents repository](https://github.com/joonspk-research/generative_agents) by Joon Sung Park et al. This project is a rewrite with a modernized stack (Django 5, WebSockets, OpenRouter) while preserving the core cognitive loop design.

## Citation

```bibtex
@inproceedings{Park2023GenerativeAgents,
  author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {UIST '23}
}
```

## License

MIT
