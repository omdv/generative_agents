"""
World and Maze system for generative agents.

This module implements the tile-based world representation with
pathfinding and location management.
"""

import csv
import heapq
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """A single tile in the maze."""

    world: str = ""
    sector: str = ""
    arena: str = ""
    game_object: str = ""
    spawning_location: str = ""
    collision: bool = False
    events: set[tuple[str, str | None, str | None, str | None]] = field(
        default_factory=set
    )


class Maze:
    """
    Represents the tile-based world map.

    The maze is loaded from CSV files exported from Tiled map editor.
    It contains multiple layers:
    - collision: Where agents can/cannot walk
    - sector: Major areas of the world
    - arena: Specific rooms/locations within sectors
    - game_object: Interactive objects
    - spawning_location: Where agents can spawn
    """

    def __init__(self, maze_name: str, maze_path: Path | str):
        """
        Initialize the maze from files.

        Args:
            maze_name: Name of this maze/world.
            maze_path: Path to the directory containing maze CSV files.
        """
        self.maze_name = maze_name
        self.maze_path = Path(maze_path)

        # Load metadata
        meta_path = self.maze_path / "maze_meta_info.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.width = int(meta["maze_width"])
        self.height = int(meta["maze_height"])
        self.tile_size = int(meta["sq_tile_size"])
        self.special_constraint = meta.get("special_constraint", "")

        # Load special blocks (color mappings)
        blocks_path = self.maze_path / "special_blocks"
        self._world_name = self._load_world_blocks(blocks_path / "world_blocks.csv")
        self._sector_blocks = self._load_blocks(blocks_path / "sector_blocks.csv")
        self._arena_blocks = self._load_blocks(blocks_path / "arena_blocks.csv")
        self._object_blocks = self._load_blocks(blocks_path / "game_object_blocks.csv")
        self._spawn_blocks = self._load_blocks(blocks_path / "spawning_location_blocks.csv")

        # Load maze matrices
        maze_folder = self.maze_path / "maze"
        collision_raw = self._load_maze_csv(maze_folder / "collision_maze.csv")
        sector_raw = self._load_maze_csv(maze_folder / "sector_maze.csv")
        arena_raw = self._load_maze_csv(maze_folder / "arena_maze.csv")
        object_raw = self._load_maze_csv(maze_folder / "game_object_maze.csv")
        spawn_raw = self._load_maze_csv(maze_folder / "spawning_location_maze.csv")

        # Convert 1D arrays to 2D matrices
        self.collision_maze = self._to_2d(collision_raw)
        sector_maze = self._to_2d(sector_raw)
        arena_maze = self._to_2d(arena_raw)
        object_maze = self._to_2d(object_raw)
        spawn_maze = self._to_2d(spawn_raw)

        # Build tile grid
        self.tiles: list[list[Tile]] = []
        for row in range(self.height):
            tile_row: list[Tile] = []
            for col in range(self.width):
                tile = Tile(world=self._world_name)

                # Set sector
                sector_code = sector_maze[row][col]
                if sector_code in self._sector_blocks:
                    tile.sector = self._sector_blocks[sector_code]

                # Set arena
                arena_code = arena_maze[row][col]
                if arena_code in self._arena_blocks:
                    tile.arena = self._arena_blocks[arena_code]

                # Set game object
                object_code = object_maze[row][col]
                if object_code in self._object_blocks:
                    tile.game_object = self._object_blocks[object_code]

                # Set spawning location
                spawn_code = spawn_maze[row][col]
                if spawn_code in self._spawn_blocks:
                    tile.spawning_location = self._spawn_blocks[spawn_code]

                # Set collision
                if self.collision_maze[row][col] != "0":
                    tile.collision = True

                tile_row.append(tile)
            self.tiles.append(tile_row)

        # Add default game object events
        for row in range(self.height):
            for col in range(self.width):
                tile = self.tiles[row][col]
                if tile.game_object:
                    object_name = self._make_address(
                        tile.world, tile.sector, tile.arena, tile.game_object
                    )
                    # Give objects a default "idle" state so they can be perceived
                    tile.events.add((object_name, "is", "idle", f"{tile.game_object} is idle"))

        # Build reverse lookup for addresses to tile coordinates
        self.address_tiles: dict[str, set[tuple[int, int]]] = {}
        for row in range(self.height):
            for col in range(self.width):
                tile = self.tiles[row][col]
                addresses = self._get_tile_addresses(tile)
                for addr in addresses:
                    if addr not in self.address_tiles:
                        self.address_tiles[addr] = set()
                    self.address_tiles[addr].add((col, row))

    def _load_world_blocks(self, path: Path) -> str:
        """Load world name from world blocks CSV."""
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    return row[-1].strip()
        return ""

    def _load_blocks(self, path: Path) -> dict[str, str]:
        """Load block mappings from CSV."""
        blocks: dict[str, str] = {}
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # First column is the tile code, last column is the name
                    # Strip whitespace from both
                    key = row[0].strip()
                    value = row[-1].strip()
                    blocks[key] = value
        return blocks

    def _load_maze_csv(self, path: Path) -> list[str]:
        """Load a maze CSV file (single row of tile codes)."""
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # Strip whitespace from each value
                return [val.strip() for val in row]
        return []

    def _to_2d(self, raw: list[str]) -> list[list[str]]:
        """Convert 1D array to 2D matrix."""
        result: list[list[str]] = []
        for i in range(0, len(raw), self.width):
            result.append(raw[i : i + self.width])
        return result

    def _make_address(self, *parts: str) -> str:
        """Create an address string from parts."""
        return ":".join(p for p in parts if p)

    def _get_tile_addresses(self, tile: Tile) -> list[str]:
        """Get all valid addresses for a tile."""
        addresses = []

        if tile.sector:
            addresses.append(f"{tile.world}:{tile.sector}")

        if tile.arena:
            addresses.append(f"{tile.world}:{tile.sector}:{tile.arena}")

        if tile.game_object:
            addresses.append(
                f"{tile.world}:{tile.sector}:{tile.arena}:{tile.game_object}"
            )

        if tile.spawning_location:
            addresses.append(f"<spawn_loc>{tile.spawning_location}")

        return addresses

    def get_tile(self, x: int, y: int) -> Tile | None:
        """
        Get the tile at the given coordinates.

        Args:
            x: X coordinate (column).
            y: Y coordinate (row).

        Returns:
            The tile at (x, y) or None if out of bounds.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None

    def get_tile_path(self, x: int, y: int, level: str = "game_object") -> str:
        """
        Get the address path for a tile at the given level.

        Args:
            x: X coordinate.
            y: Y coordinate.
            level: Level of detail ("world", "sector", "arena", "game_object").

        Returns:
            Address string for the tile.
        """
        tile = self.get_tile(x, y)
        if not tile:
            return ""

        path = tile.world
        if level == "world":
            return path

        path += f":{tile.sector}"
        if level == "sector":
            return path

        path += f":{tile.arena}"
        if level == "arena":
            return path

        path += f":{tile.game_object}"
        return path

    def get_nearby_tiles(
        self, x: int, y: int, radius: int
    ) -> list[tuple[int, int]]:
        """
        Get all tiles within a radius of the given position.

        Args:
            x: Center X coordinate.
            y: Center Y coordinate.
            radius: Radius in tiles.

        Returns:
            List of (x, y) coordinates within the radius.
        """
        left = max(0, x - radius)
        right = min(self.width, x + radius + 1)
        top = max(0, y - radius)
        bottom = min(self.height, y + radius + 1)

        tiles = []
        for col in range(left, right):
            for row in range(top, bottom):
                tiles.append((col, row))
        return tiles

    def pixel_to_tile(self, px: int, py: int) -> tuple[int, int]:
        """Convert pixel coordinates to tile coordinates."""
        return (px // self.tile_size, py // self.tile_size)

    def tile_to_pixel(self, x: int, y: int) -> tuple[int, int]:
        """Convert tile coordinates to pixel coordinates (center of tile)."""
        return (
            x * self.tile_size + self.tile_size // 2,
            y * self.tile_size + self.tile_size // 2,
        )

    def add_event(
        self,
        x: int,
        y: int,
        event: tuple[str, str | None, str | None, str | None],
    ) -> None:
        """Add an event to a tile."""
        tile = self.get_tile(x, y)
        if tile:
            tile.events.add(event)

    def remove_event(
        self,
        x: int,
        y: int,
        event: tuple[str, str | None, str | None, str | None],
    ) -> None:
        """Remove an event from a tile."""
        tile = self.get_tile(x, y)
        if tile and event in tile.events:
            tile.events.remove(event)

    def remove_subject_events(self, x: int, y: int, subject: str) -> None:
        """Remove all events with the given subject from a tile."""
        tile = self.get_tile(x, y)
        if tile:
            tile.events = {e for e in tile.events if e[0] != subject}

    def get_tiles_for_address(self, address: str) -> set[tuple[int, int]]:
        """Get all tile coordinates for a given address."""
        return self.address_tiles.get(address, set())

    def find_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Find a path from start to end using A* pathfinding.

        Args:
            start: Starting position (x, y).
            end: Target position (x, y).

        Returns:
            List of positions from start to end (excluding start, including end).
            Returns empty list if no path found.
        """
        if start == end:
            return []

        # Check if end is walkable
        end_tile = self.get_tile(end[0], end[1])
        if not end_tile or end_tile.collision:
            # Try to find a nearby walkable tile
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    alt_x, alt_y = end[0] + dx, end[1] + dy
                    alt_tile = self.get_tile(alt_x, alt_y)
                    if alt_tile and not alt_tile.collision:
                        end = (alt_x, alt_y)
                        break
                else:
                    continue
                break
            else:
                return []

        # A* pathfinding
        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: list[tuple[float, int, tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0}
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            # Check all 4 neighbors (no diagonal movement)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue

                # Check collision
                neighbor_tile = self.get_tile(neighbor[0], neighbor[1])
                if not neighbor_tile or neighbor_tile.collision:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return []  # No path found


class World:
    """
    High-level world representation that wraps a Maze with simulation state.
    """

    def __init__(self, maze: Maze):
        """
        Initialize the world.

        Args:
            maze: The underlying maze/map.
        """
        self.maze = maze
        self.name = maze.maze_name

        # Track what's happening at each location
        self.location_events: dict[str, set[str]] = {}

    def get_events_at(self, x: int, y: int) -> set[tuple[str, str | None, str | None, str | None]]:
        """Get all events at a tile position."""
        tile = self.maze.get_tile(x, y)
        if tile:
            return tile.events.copy()
        return set()

    def get_nearby_events(
        self, x: int, y: int, radius: int
    ) -> list[tuple[tuple[int, int], tuple[str, str | None, str | None, str | None]]]:
        """
        Get all events within a radius of a position.

        Returns:
            List of ((x, y), event) tuples.
        """
        events = []
        for tile_x, tile_y in self.maze.get_nearby_tiles(x, y, radius):
            tile = self.maze.get_tile(tile_x, tile_y)
            if tile:
                for event in tile.events:
                    events.append(((tile_x, tile_y), event))
        return events

    def add_agent_event(
        self,
        x: int,
        y: int,
        agent_name: str,
        predicate: str | None,
        object_: str | None,
        description: str | None = None,
    ) -> None:
        """Add an event for an agent at a position."""
        event = (agent_name, predicate, object_, description)
        self.maze.add_event(x, y, event)

    def remove_agent_events(self, x: int, y: int, agent_name: str) -> None:
        """Remove all events for an agent at a position."""
        self.maze.remove_subject_events(x, y, agent_name)

    def move_agent(
        self,
        agent_name: str,
        from_pos: tuple[int, int],
        to_pos: tuple[int, int],
        predicate: str | None = None,
        object_: str | None = None,
        description: str | None = None,
    ) -> None:
        """Move an agent from one position to another."""
        self.remove_agent_events(from_pos[0], from_pos[1], agent_name)
        self.add_agent_event(to_pos[0], to_pos[1], agent_name, predicate, object_, description)

    def find_path(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Find a path from start to end."""
        return self.maze.find_path(start, end)

    def get_address_tiles(self, address: str) -> set[tuple[int, int]]:
        """Get all tiles that match an address."""
        return self.maze.get_tiles_for_address(address)

    def to_dict(self) -> dict[str, Any]:
        """Convert world state to dictionary."""
        return {
            "name": self.name,
            "width": self.maze.width,
            "height": self.maze.height,
            "tile_size": self.maze.tile_size,
        }
