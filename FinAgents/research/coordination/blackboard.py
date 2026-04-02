"""Shared Memory Blackboard System for Multi-Agent Coordination.

This module implements a thread-safe blackboard pattern where agents can
read and write structured data to shared memory sections. Each section
represents a different aspect of the trading system state.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union


class Section(Enum):
    """Enumeration of blackboard sections for different data types.

    Each section represents a distinct domain of shared information
    that agents can read from and write to.
    """

    MARKET_STATE = auto()
    """Current market conditions, prices, and indicators."""

    PROPOSALS = auto()
    """Trade proposals submitted by trading agents."""

    RISK_ASSESSMENTS = auto()
    """Risk evaluations and approvals from risk managers."""

    PORTFOLIO_STATE = auto()
    """Current portfolio allocation and positions."""

    DECISIONS = auto()
    """Final execution decisions and orders."""

    VOTES = auto()
    """Voting records and results from agent deliberations."""

    EXECUTION_LOG = auto()
    """Historical log of executed trades and outcomes."""


@dataclass
class BlackboardEntry:
    """A single entry in the blackboard shared memory.

    Attributes
    ----------
    key : str
        Unique identifier for this entry within its section.
    value : Any
        The actual data stored in this entry.
    author : str
        Agent ID that created or last modified this entry.
    timestamp : datetime
        When this entry was created or last modified.
    section : Section
        Which blackboard section this entry belongs to.
    version : int
        Version number, incremented on each update.
    metadata : dict
        Additional metadata for extensibility.
    entry_id : str
        Unique identifier for this specific entry instance.
    """

    key: str
    value: Any
    author: str
    timestamp: datetime
    section: Section
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to a serializable dictionary."""
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": self.value,
            "author": self.author,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "section": self.section.name,
            "version": self.version,
            "metadata": self.metadata,
        }


class Blackboard:
    """Thread-safe shared memory blackboard for agent coordination.

    The blackboard provides a centralized, structured storage system where
    agents can publish information and subscribe to updates from other agents.
    Access control ensures agents only write to appropriate sections.

    Attributes
    ----------
    _storage : dict
        Internal storage mapping sections to dictionaries of entries.
    _subscribers : dict
        Callback functions registered for section updates.
    _access_control : dict
        Maps agent IDs to sets of sections they can write to.
    _history : dict
        Maintains version history for entries.
    _lock : threading.Lock
        Thread safety lock for all operations.
    """

    def __init__(self) -> None:
        """Initialize the blackboard with empty storage and defaults."""
        self._storage: Dict[Section, Dict[str, BlackboardEntry]] = {
            section: {} for section in Section
        }
        self._subscribers: Dict[Section, List[Callable[[BlackboardEntry], None]]] = {
            section: [] for section in Section
        }
        self._access_control: Dict[str, set] = {}
        self._history: Dict[str, List[BlackboardEntry]] = {}
        self._lock = threading.Lock()

    def write(
        self,
        section: Section,
        key: str,
        value: Any,
        author: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BlackboardEntry:
        """Write an entry to the blackboard.

        Validates that the author has write access to the section,
        increments the version if updating an existing entry,
        and notifies all subscribers.

        Parameters
        ----------
        section : Section
            The blackboard section to write to.
        key : str
            Unique key for this entry within the section.
        value : Any
            The data to store.
        author : str
            ID of the agent writing this entry.
        metadata : dict, optional
            Additional metadata to attach to the entry.

        Returns
        -------
        BlackboardEntry
            The created or updated entry.

        Raises
        ------
        PermissionError
            If the author does not have write access to the section.
        """
        with self._lock:
            # Validate access
            if author in self._access_control:
                allowed_sections = self._access_control[author]
                if section not in allowed_sections:
                    raise PermissionError(
                        f"Agent '{author}' does not have write access to section '{section.name}'"
                    )

            # Determine version
            version = 1
            if key in self._storage[section]:
                version = self._storage[section][key].version + 1

            # Create entry
            entry = BlackboardEntry(
                key=key,
                value=value,
                author=author,
                timestamp=datetime.utcnow(),
                section=section,
                version=version,
                metadata=metadata or {},
            )

            # Store entry
            self._storage[section][key] = entry

            # Maintain history (keep last 10 versions)
            history_key = f"{section.name}:{key}"
            if history_key not in self._history:
                self._history[history_key] = []
            self._history[history_key].append(entry)
            if len(self._history[history_key]) > 10:
                self._history[history_key] = self._history[history_key][-10:]

            # Notify subscribers
            for callback in self._subscribers[section]:
                try:
                    callback(entry)
                except Exception as e:
                    # Log but don't fail on subscriber errors
                    print(f"Subscriber error for section {section.name}: {e}")

            return entry

    def read(
        self, section: Section, key: Optional[str] = None
    ) -> Union[BlackboardEntry, Dict[str, BlackboardEntry], None]:
        """Read entries from the blackboard.

        Parameters
        ----------
        section : Section
            The section to read from.
        key : str, optional
            Specific key to read. If None, returns all entries in the section.

        Returns
        -------
        BlackboardEntry or dict or None
            Single entry if key provided, dictionary of all entries if key is None,
            or None if key not found.
        """
        with self._lock:
            if key is not None:
                return self._storage[section].get(key)
            return dict(self._storage[section])

    def subscribe(
        self, section: Section, callback: Callable[[BlackboardEntry], None]
    ) -> None:
        """Register a callback for updates to a section.

        The callback will be invoked whenever any entry in the section
        is created or updated.

        Parameters
        ----------
        section : Section
            The section to subscribe to.
        callback : Callable
            Function to call with the updated entry.
        """
        with self._lock:
            self._subscribers[section].append(callback)

    def unsubscribe(
        self, section: Section, callback: Callable[[BlackboardEntry], None]
    ) -> bool:
        """Remove a callback subscription from a section.

        Parameters
        ----------
        section : Section
            The section to unsubscribe from.
        callback : Callable
            The callback function to remove.

        Returns
        -------
        bool
            True if callback was found and removed, False otherwise.
        """
        with self._lock:
            if callback in self._subscribers[section]:
                self._subscribers[section].remove(callback)
                return True
            return False

    def set_access(self, agent_id: str, writable_sections: set) -> None:
        """Configure write access for an agent.

        Parameters
        ----------
        agent_id : str
            The agent to configure access for.
        writable_sections : set of Section
            Set of sections the agent can write to.
        """
        with self._lock:
            self._access_control[agent_id] = writable_sections

    def get_access(self, agent_id: str) -> set:
        """Get the writable sections for an agent.

        Parameters
        ----------
        agent_id : str
            The agent to check access for.

        Returns
        -------
        set of Section
            Set of sections the agent can write to, or empty set if not configured.
        """
        with self._lock:
            return self._access_control.get(agent_id, set())

    def get_history(
        self, section: Section, key: str, limit: int = 10
    ) -> List[BlackboardEntry]:
        """Get version history for a specific entry.

        Parameters
        ----------
        section : Section
            The section containing the entry.
        key : str
            The entry key.
        limit : int, default 10
            Maximum number of historical versions to return.

        Returns
        -------
        list of BlackboardEntry
            Historical versions, oldest first.
        """
        with self._lock:
            history_key = f"{section.name}:{key}"
            history = self._history.get(history_key, [])
            return history[-limit:] if limit < len(history) else list(history)

    def clear_section(self, section: Section) -> None:
        """Clear all entries from a section.

        Parameters
        ----------
        section : Section
            The section to clear.
        """
        with self._lock:
            self._storage[section].clear()

    def delete_entry(self, section: Section, key: str) -> bool:
        """Delete a specific entry from the blackboard.

        Parameters
        ----------
        section : Section
            The section containing the entry.
        key : str
            The entry key to delete.

        Returns
        -------
        bool
            True if entry was found and deleted, False otherwise.
        """
        with self._lock:
            if key in self._storage[section]:
                del self._storage[section][key]
                return True
            return False

    def snapshot(self) -> Dict[str, Any]:
        """Create a full snapshot of the blackboard state.

        Returns
        -------
        dict
            Serializable representation of all sections and entries.
        """
        with self._lock:
            return {
                section.name: {
                    key: entry.to_dict()
                    for key, entry in section_data.items()
                }
                for section, section_data in self._storage.items()
            }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore blackboard state from a snapshot.

        Parameters
        ----------
        snapshot : dict
            Snapshot data as returned by snapshot().
        """
        with self._lock:
            for section_name, section_data in snapshot.items():
                try:
                    section = Section[section_name]
                    self._storage[section].clear()
                    for key, entry_dict in section_data.items():
                        entry = BlackboardEntry(
                            key=entry_dict["key"],
                            value=entry_dict["value"],
                            author=entry_dict["author"],
                            timestamp=datetime.fromisoformat(entry_dict["timestamp"])
                            if entry_dict["timestamp"]
                            else datetime.utcnow(),
                            section=section,
                            version=entry_dict["version"],
                            metadata=entry_dict.get("metadata", {}),
                            entry_id=entry_dict.get("entry_id", str(uuid.uuid4())),
                        )
                        self._storage[section][key] = entry
                except (KeyError, ValueError) as e:
                    print(f"Error restoring section {section_name}: {e}")

    def get_section_keys(self, section: Section) -> List[str]:
        """Get all keys in a section.

        Parameters
        ----------
        section : Section
            The section to query.

        Returns
        -------
        list of str
            All keys currently in the section.
        """
        with self._lock:
            return list(self._storage[section].keys())

    def has_entry(self, section: Section, key: str) -> bool:
        """Check if an entry exists.

        Parameters
        ----------
        section : Section
            The section to check.
        key : str
            The key to look for.

        Returns
        -------
        bool
            True if entry exists, False otherwise.
        """
        with self._lock:
            return key in self._storage[section]
