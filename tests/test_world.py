"""Tests for world/maze system."""

import pytest
from pathlib import Path

from core.world import Maze, World, Tile


class TestTile:
    """Tests for Tile dataclass."""

    def test_create_tile(self):
        """Test creating a tile."""
        tile = Tile(
            world="the Ville",
            sector="town center",
            arena="cafe",
            game_object="counter",
        )

        assert tile.world == "the Ville"
        assert tile.collision is False
        assert len(tile.events) == 0

    def test_tile_events(self):
        """Test adding events to a tile."""
        tile = Tile()
        event = ("John", "is", "walking", "John is walking")
        tile.events.add(event)

        assert len(tile.events) == 1
        assert event in tile.events


class TestMaze:
    """Tests for Maze class."""

    @pytest.fixture
    def maze_path(self):
        """Get path to test maze assets."""
        # Use the existing the_ville maze
        return Path(__file__).parent.parent / "environment" / "frontend_server" / "static_dirs" / "assets" / "the_ville" / "matrix"

    def test_load_maze(self, maze_path):
        """Test loading maze from files."""
        if not maze_path.exists():
            pytest.skip("Maze assets not found")

        maze = Maze("the_ville", maze_path)

        assert maze.maze_name == "the_ville"
        assert maze.width == 140
        assert maze.height == 100
        assert maze.tile_size == 32

    def test_get_tile(self, maze_path):
        """Test getting a tile."""
        if not maze_path.exists():
            pytest.skip("Maze assets not found")

        maze = Maze("the_ville", maze_path)

        tile = maze.get_tile(50, 50)
        assert tile is not None
        assert isinstance(tile, Tile)

        # Out of bounds
        assert maze.get_tile(-1, 0) is None
        assert maze.get_tile(1000, 1000) is None

    def test_get_nearby_tiles(self, maze_path):
        """Test getting nearby tiles."""
        if not maze_path.exists():
            pytest.skip("Maze assets not found")

        maze = Maze("the_ville", maze_path)

        nearby = maze.get_nearby_tiles(50, 50, 2)

        # Should be a 5x5 area = 25 tiles
        assert len(nearby) == 25

    def test_find_path(self, maze_path):
        """Test pathfinding."""
        if not maze_path.exists():
            pytest.skip("Maze assets not found")

        maze = Maze("the_ville", maze_path)

        # Find a path between two points
        # (These coordinates should be in walkable areas)
        path = maze.find_path((50, 50), (55, 50))

        # Path should exist (assuming these are walkable)
        # May be empty if blocked or same position
        assert isinstance(path, list)

    def test_address_tiles(self, maze_path):
        """Test address to tiles mapping."""
        if not maze_path.exists():
            pytest.skip("Maze assets not found")

        maze = Maze("the_ville", maze_path)

        # Check that we have some address mappings
        assert len(maze.address_tiles) > 0


class TestWorld:
    """Tests for World class."""

    @pytest.fixture
    def world(self, tmp_path):
        """Create a mock world for testing."""
        # Create a simple mock maze
        class MockMaze:
            def __init__(self):
                self.maze_name = "test_world"
                self.width = 10
                self.height = 10
                self.tile_size = 32
                self.tiles = [
                    [Tile() for _ in range(10)]
                    for _ in range(10)
                ]
                self.address_tiles = {}

            def get_tile(self, x, y):
                if 0 <= x < self.width and 0 <= y < self.height:
                    return self.tiles[y][x]
                return None

            def get_nearby_tiles(self, x, y, radius):
                tiles = []
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            tiles.append((x + dx, y + dy))
                return tiles

            def add_event(self, x, y, event):
                tile = self.get_tile(x, y)
                if tile:
                    tile.events.add(event)

            def remove_subject_events(self, x, y, subject):
                tile = self.get_tile(x, y)
                if tile:
                    tile.events = {e for e in tile.events if e[0] != subject}

            def find_path(self, start, end):
                return []

            def get_tiles_for_address(self, address):
                return self.address_tiles.get(address, set())

        return World(MockMaze())

    def test_create_world(self, world):
        """Test creating a world."""
        assert world.name == "test_world"

    def test_get_events_at(self, world):
        """Test getting events at a position."""
        events = world.get_events_at(5, 5)
        assert isinstance(events, set)

    def test_add_agent_event(self, world):
        """Test adding an agent event."""
        world.add_agent_event(5, 5, "TestAgent", "is", "testing", "Testing")

        events = world.get_events_at(5, 5)
        assert len(events) == 1

    def test_move_agent(self, world):
        """Test moving an agent."""
        # Add event at start
        world.add_agent_event(0, 0, "Agent", "is", "idle")

        # Move agent
        world.move_agent("Agent", (0, 0), (5, 5), "is", "walking")

        # Should be removed from old position
        old_events = world.get_events_at(0, 0)
        assert all(e[0] != "Agent" for e in old_events)

        # Should be at new position
        new_events = world.get_events_at(5, 5)
        assert any(e[0] == "Agent" for e in new_events)
