"""
Memory system for generative agents.

This module implements the memory stream that stores events, thoughts, and chats
with support for retrieval based on recency, relevance, and importance.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class MemoryNode:
    """
    A single memory node representing an event, thought, or chat.

    Attributes:
        id: Unique identifier for this memory.
        node_type: Type of memory ("event", "thought", "chat").
        subject: The subject of the memory (who/what).
        predicate: The action/relationship.
        object: The object of the action.
        description: Human-readable description of the memory.
        keywords: Keywords associated with this memory for retrieval.
        poignancy: Importance score (1-10).
        embedding: Vector embedding for semantic search.
        created_at: When this memory was created.
        expires_at: When this memory expires (optional).
        last_accessed: When this memory was last retrieved.
        depth: Depth in the reflection hierarchy (0 for events, higher for thoughts).
        filling: References to other memories that contributed to this one.
    """

    id: str
    node_type: str  # "event", "thought", "chat"
    subject: str
    predicate: str | None
    object: str | None
    description: str
    keywords: set[str] = field(default_factory=set)
    poignancy: float = 1.0
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    last_accessed: datetime | None = None
    depth: int = 0
    filling: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        """Hash by unique id for use in sets and dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on id."""
        if isinstance(other, MemoryNode):
            return self.id == other.id
        return False

    def spo_summary(self) -> tuple[str, str | None, str | None]:
        """Return subject-predicate-object tuple."""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "keywords": list(self.keywords),
            "poignancy": self.poignancy,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "depth": self.depth,
            "filling": self.filling,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        """Create from dictionary."""
        data = data.copy()
        data["keywords"] = set(data.get("keywords", []))
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class MemoryStore:
    """
    SQLite-backed memory store with vector search support.

    This implements the associative memory from the original paper,
    with retrieval based on recency, relevance, and importance.
    """

    def __init__(self, db_path: str | Path, agent_id: str):
        """
        Initialize the memory store.

        Args:
            db_path: Path to the SQLite database.
            agent_id: Unique identifier for the agent this memory belongs to.
        """
        self.db_path = Path(db_path)
        self.agent_id = agent_id
        self._ensure_tables()

        # In-memory caches for fast access
        self._id_to_node: dict[str, MemoryNode] = {}
        self._keyword_to_events: dict[str, list[MemoryNode]] = {}
        self._keyword_to_thoughts: dict[str, list[MemoryNode]] = {}
        self._keyword_to_chats: dict[str, list[MemoryNode]] = {}
        self._keyword_strength_events: dict[str, int] = {}
        self._keyword_strength_thoughts: dict[str, int] = {}

        # Sequence lists (most recent first)
        self._seq_events: list[MemoryNode] = []
        self._seq_thoughts: list[MemoryNode] = []
        self._seq_chats: list[MemoryNode] = []

        # Load existing memories from database
        self._load_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    description TEXT NOT NULL,
                    keywords TEXT,
                    poignancy REAL,
                    embedding BLOB,
                    created_at TEXT,
                    expires_at TEXT,
                    last_accessed TEXT,
                    depth INTEGER DEFAULT 0,
                    filling TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent
                ON memories(agent_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(agent_id, node_type)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keyword_strength (
                    agent_id TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    strength INTEGER DEFAULT 0,
                    PRIMARY KEY (agent_id, keyword, memory_type)
                )
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        """Load all memories for this agent from the database."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at DESC",
                (self.agent_id,),
            )
            for row in cursor:
                node = self._row_to_node(row)
                self._add_to_cache(node)

            # Load keyword strengths
            cursor = conn.execute(
                "SELECT * FROM keyword_strength WHERE agent_id = ?",
                (self.agent_id,),
            )
            for row in cursor:
                if row["memory_type"] == "event":
                    self._keyword_strength_events[row["keyword"]] = row["strength"]
                elif row["memory_type"] == "thought":
                    self._keyword_strength_thoughts[row["keyword"]] = row["strength"]

    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
        """Convert a database row to a MemoryNode."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])

        keywords = set()
        if row["keywords"]:
            keywords = set(json.loads(row["keywords"]))

        filling = []
        if row["filling"]:
            filling = json.loads(row["filling"])

        return MemoryNode(
            id=row["id"],
            node_type=row["node_type"],
            subject=row["subject"] or "",
            predicate=row["predicate"],
            object=row["object"],
            description=row["description"],
            keywords=keywords,
            poignancy=row["poignancy"] or 1.0,
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            depth=row["depth"] or 0,
            filling=filling,
        )

    def _add_to_cache(self, node: MemoryNode) -> None:
        """Add a memory node to the in-memory caches."""
        self._id_to_node[node.id] = node

        # Add to type-specific sequence
        if node.node_type == "event":
            self._seq_events.insert(0, node)
        elif node.node_type == "thought":
            self._seq_thoughts.insert(0, node)
        elif node.node_type == "chat":
            self._seq_chats.insert(0, node)

        # Add to keyword indices
        for keyword in node.keywords:
            kw_lower = keyword.lower()
            if node.node_type == "event":
                if kw_lower not in self._keyword_to_events:
                    self._keyword_to_events[kw_lower] = []
                self._keyword_to_events[kw_lower].insert(0, node)
            elif node.node_type == "thought":
                if kw_lower not in self._keyword_to_thoughts:
                    self._keyword_to_thoughts[kw_lower] = []
                self._keyword_to_thoughts[kw_lower].insert(0, node)
            elif node.node_type == "chat":
                if kw_lower not in self._keyword_to_chats:
                    self._keyword_to_chats[kw_lower] = []
                self._keyword_to_chats[kw_lower].insert(0, node)

    def _save_to_db(self, node: MemoryNode) -> None:
        """Save a memory node to the database."""
        with self._get_connection() as conn:
            embedding_blob = json.dumps(node.embedding) if node.embedding else None
            keywords_json = json.dumps(list(node.keywords))
            filling_json = json.dumps(node.filling)

            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, agent_id, node_type, subject, predicate, object, description,
                 keywords, poignancy, embedding, created_at, expires_at, last_accessed,
                 depth, filling)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    self.agent_id,
                    node.node_type,
                    node.subject,
                    node.predicate,
                    node.object,
                    node.description,
                    keywords_json,
                    node.poignancy,
                    embedding_blob,
                    node.created_at.isoformat() if node.created_at else None,
                    node.expires_at.isoformat() if node.expires_at else None,
                    node.last_accessed.isoformat() if node.last_accessed else None,
                    node.depth,
                    filling_json,
                ),
            )
            conn.commit()

    def _update_keyword_strength(self, keywords: set[str], memory_type: str) -> None:
        """Update keyword strength in database and cache."""
        strength_dict = (
            self._keyword_strength_events
            if memory_type == "event"
            else self._keyword_strength_thoughts
        )

        with self._get_connection() as conn:
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in strength_dict:
                    strength_dict[kw_lower] += 1
                else:
                    strength_dict[kw_lower] = 1

                conn.execute(
                    """
                    INSERT OR REPLACE INTO keyword_strength
                    (agent_id, keyword, memory_type, strength)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, kw_lower, memory_type, strength_dict[kw_lower]),
                )
            conn.commit()

    async def add_event(
        self,
        subject: str,
        predicate: str | None,
        object_: str | None,
        description: str,
        keywords: set[str],
        poignancy: float,
        embedding: list[float] | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryNode:
        """Add an event memory."""
        node = MemoryNode(
            id=f"event_{uuid4().hex[:12]}",
            node_type="event",
            subject=subject,
            predicate=predicate,
            object=object_,
            description=description,
            keywords=keywords,
            poignancy=poignancy,
            embedding=embedding,
            created_at=created_at or datetime.now(),
            expires_at=expires_at,
            depth=0,
        )

        self._save_to_db(node)
        self._add_to_cache(node)

        # Update keyword strength for non-idle events
        if predicate != "is" or object_ != "idle":
            self._update_keyword_strength(keywords, "event")

        return node

    async def add_thought(
        self,
        subject: str,
        predicate: str | None,
        object_: str | None,
        description: str,
        keywords: set[str],
        poignancy: float,
        embedding: list[float] | None = None,
        filling: list[str] | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryNode:
        """Add a thought memory (reflection)."""
        # Calculate depth based on filling
        depth = 1
        if filling:
            max_depth = 0
            for node_id in filling:
                if node_id in self._id_to_node:
                    max_depth = max(max_depth, self._id_to_node[node_id].depth)
            depth = max_depth + 1

        node = MemoryNode(
            id=f"thought_{uuid4().hex[:12]}",
            node_type="thought",
            subject=subject,
            predicate=predicate,
            object=object_,
            description=description,
            keywords=keywords,
            poignancy=poignancy,
            embedding=embedding,
            created_at=created_at or datetime.now(),
            expires_at=expires_at,
            depth=depth,
            filling=filling or [],
        )

        self._save_to_db(node)
        self._add_to_cache(node)

        # Update keyword strength
        if predicate != "is" or object_ != "idle":
            self._update_keyword_strength(keywords, "thought")

        return node

    async def add_chat(
        self,
        subject: str,
        predicate: str | None,
        object_: str | None,
        description: str,
        keywords: set[str],
        poignancy: float,
        embedding: list[float] | None = None,
        filling: list[tuple[str, str]] | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryNode:
        """Add a chat memory."""
        node = MemoryNode(
            id=f"chat_{uuid4().hex[:12]}",
            node_type="chat",
            subject=subject,
            predicate=predicate,
            object=object_,
            description=description,
            keywords=keywords,
            poignancy=poignancy,
            embedding=embedding,
            created_at=created_at or datetime.now(),
            expires_at=expires_at,
            depth=0,
            filling=[json.dumps(f) for f in (filling or [])],
        )

        self._save_to_db(node)
        self._add_to_cache(node)

        return node

    async def retrieve(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.5,
        importance_weight: float = 0.2,
        node_types: list[str] | None = None,
    ) -> list[MemoryNode]:
        """
        Retrieve memories based on recency, relevance, and importance.

        Args:
            query: Text query for relevance matching.
            query_embedding: Pre-computed embedding for semantic search.
            limit: Maximum number of memories to return.
            recency_weight: Weight for recency score (0-1).
            relevance_weight: Weight for relevance score (0-1).
            importance_weight: Weight for importance/poignancy score (0-1).
            node_types: Filter by memory types ("event", "thought", "chat").

        Returns:
            List of memories sorted by combined score.
        """
        # Collect all candidate memories
        candidates: list[MemoryNode] = []

        if node_types is None:
            node_types = ["event", "thought", "chat"]

        if "event" in node_types:
            candidates.extend(self._seq_events)
        if "thought" in node_types:
            candidates.extend(self._seq_thoughts)
        if "chat" in node_types:
            candidates.extend(self._seq_chats)

        if not candidates:
            return []

        # Calculate scores for each memory
        now = datetime.now()
        scored_memories: list[tuple[float, MemoryNode]] = []

        for node in candidates:
            # Recency score (exponential decay)
            if node.created_at:
                hours_ago = (now - node.created_at).total_seconds() / 3600
                recency_score = 0.99 ** hours_ago
            else:
                recency_score = 0.0

            # Importance score (normalized poignancy)
            importance_score = node.poignancy / 10.0

            # Relevance score (cosine similarity if embedding available)
            relevance_score = 0.0
            if query_embedding and node.embedding:
                # Cosine similarity
                a = np.array(query_embedding)
                b = np.array(node.embedding)
                if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                    relevance_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

            # Combined score
            total_score = (
                recency_weight * recency_score
                + relevance_weight * relevance_score
                + importance_weight * importance_score
            )

            scored_memories.append((total_score, node))

        # Sort by score (descending) and return top N
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Update last_accessed for retrieved memories
        result = [node for _, node in scored_memories[:limit]]
        for node in result:
            node.last_accessed = now

        return result

    def get_recent_events(self, count: int = 10) -> list[MemoryNode]:
        """Get the most recent events."""
        return self._seq_events[:count]

    def get_recent_thoughts(self, count: int = 10) -> list[MemoryNode]:
        """Get the most recent thoughts."""
        return self._seq_thoughts[:count]

    def get_recent_chats(self, count: int = 10) -> list[MemoryNode]:
        """Get the most recent chats."""
        return self._seq_chats[:count]

    def get_events_by_keywords(self, keywords: list[str]) -> set[MemoryNode]:
        """Get events that match any of the given keywords."""
        result: set[MemoryNode] = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self._keyword_to_events:
                result.update(self._keyword_to_events[kw_lower])
        return result

    def get_thoughts_by_keywords(self, keywords: list[str]) -> set[MemoryNode]:
        """Get thoughts that match any of the given keywords."""
        result: set[MemoryNode] = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self._keyword_to_thoughts:
                result.update(self._keyword_to_thoughts[kw_lower])
        return result

    def get_last_chat_with(self, persona_name: str) -> MemoryNode | None:
        """Get the most recent chat with a specific persona."""
        kw_lower = persona_name.lower()
        if kw_lower in self._keyword_to_chats:
            return self._keyword_to_chats[kw_lower][0]
        return None

    def get_keyword_strength(self, keyword: str, memory_type: str = "event") -> int:
        """Get the strength of a keyword."""
        kw_lower = keyword.lower()
        if memory_type == "event":
            return self._keyword_strength_events.get(kw_lower, 0)
        return self._keyword_strength_thoughts.get(kw_lower, 0)

    def get_summarized_latest_events(self, count: int) -> set[tuple[str, str | None, str | None]]:
        """Get SPO summaries of the latest events."""
        return {node.spo_summary() for node in self._seq_events[:count]}
