"""Experience Replay Buffer for financial agent learning.

This module implements a prioritized experience replay buffer that stores
trade experiences and samples them for learning with configurable
prioritization strategies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from FinAgents.research.memory_learning.trade_memory import TradeRecord


@dataclass
class Experience:
    """A single experience entry in the replay buffer.

    Attributes
    ----------
    experience_id :
        Unique identifier for the experience.
    trade_record :
        The trade record associated with this experience.
    state_before :
        Market state when the decision was made.
    action_taken :
        The decision/action that was taken.
    state_after :
        Market state after the outcome.
    reward :
        Computed reward signal.
    priority :
        Importance weight for replay sampling.
    regime_tag :
        Market regime at time of trade.
    timestamp :
        When the experience was recorded.
    replay_count :
        Number of times this experience has been sampled.
    """

    experience_id: str
    trade_record: TradeRecord
    state_before: Dict[str, Any]
    action_taken: Dict[str, Any]
    state_after: Dict[str, Any]
    reward: float
    priority: float = 1.0
    regime_tag: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    replay_count: int = 0

    @classmethod
    def create(
        cls,
        trade_record: TradeRecord,
        state_before: Dict[str, Any],
        action_taken: Dict[str, Any],
        state_after: Dict[str, Any],
        reward: float,
        regime_tag: Optional[str] = None,
    ) -> "Experience":
        """Factory method to create a new experience.

        Parameters
        ----------
        trade_record :
            Associated trade record.
        state_before :
            Market state before the action.
        action_taken :
            The action that was taken.
        state_after :
            Market state after the outcome.
        reward :
            Computed reward.
        regime_tag :
            Market regime tag. Defaults to trade's regime.

        Returns
        -------
        Experience
            New experience instance.
        """
        return cls(
            experience_id=str(uuid.uuid4()),
            trade_record=trade_record,
            state_before=state_before,
            action_taken=action_taken,
            state_after=state_after,
            reward=reward,
            priority=1.0,  # Will be computed by buffer
            regime_tag=regime_tag or trade_record.market_regime,
            timestamp=datetime.now(),
            replay_count=0,
        )


class ExperienceReplayBuffer:
    """Prioritized experience replay buffer for learning.

    This buffer stores trade experiences and provides prioritized sampling
    for learning, with support for different prioritization strategies.

    Parameters
    ----------
    capacity :
        Maximum number of experiences to store.
    prioritization :
        Prioritization strategy: 'uniform', 'proportional', or 'rank'.
    """

    def __init__(
        self,
        capacity: int = 5000,
        prioritization: str = "proportional",
    ) -> None:
        """Initialize the experience replay buffer."""
        self.capacity = capacity
        self.prioritization = prioritization.lower()

        if self.prioritization not in ["uniform", "proportional", "rank"]:
            raise ValueError(
                f"Invalid prioritization '{prioritization}'. "
                "Must be 'uniform', 'proportional', or 'rank'."
            )

        self._buffer: List[Experience] = []
        self._id_to_index: Dict[str, int] = {}
        self._regime_counts: Dict[str, int] = {}
        self._total_priority: float = 0.0

        # Recency bonus parameters
        self._recency_weight = 0.1
        self._novelty_weight = 0.2

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer.

        If the buffer is at capacity, the lowest priority experience is removed.
        Priority is computed based on reward, recency, and novelty.

        Parameters
        ----------
        experience :
            Experience to add.
        """
        # Compute initial priority
        priority = self._compute_priority(experience)
        experience.priority = priority

        # If at capacity, remove lowest priority
        if len(self._buffer) >= self.capacity:
            self._remove_lowest_priority()

        # Add to buffer
        index = len(self._buffer)
        self._buffer.append(experience)
        self._id_to_index[experience.experience_id] = index
        self._total_priority += priority

        # Update regime counts
        regime = experience.regime_tag
        self._regime_counts[regime] = self._regime_counts.get(regime, 0) + 1

    def _compute_priority(self, experience: Experience) -> float:
        """Compute priority based on reward, recency, and novelty."""
        # Base priority from absolute reward
        base_priority = abs(experience.reward) + 0.1  # Small base to avoid zero

        # Recency bonus: newer experiences get slight bonus
        age = (datetime.now() - experience.timestamp).total_seconds()
        recency_bonus = np.exp(-age / (24 * 3600)) * self._recency_weight  # Decay over 24h

        # Novelty bonus: experiences from underrepresented regimes
        regime_count = self._regime_counts.get(experience.regime_tag, 0)
        total_count = max(1, len(self._buffer))
        regime_ratio = regime_count / total_count
        novelty_bonus = (1 - regime_ratio) * self._novelty_weight

        return base_priority * (1 + recency_bonus + novelty_bonus)

    def _remove_lowest_priority(self) -> None:
        """Remove the lowest priority experience from the buffer."""
        if not self._buffer:
            return

        # Find minimum priority experience
        min_priority = float("inf")
        min_index = 0

        for idx, exp in enumerate(self._buffer):
            if exp.priority < min_priority:
                min_priority = exp.priority
                min_index = idx

        # Remove from buffer
        removed = self._buffer.pop(min_index)
        self._total_priority -= removed.priority

        # Update regime counts
        if removed.regime_tag in self._regime_counts:
            self._regime_counts[removed.regime_tag] -= 1
            if self._regime_counts[removed.regime_tag] <= 0:
                del self._regime_counts[removed.regime_tag]

        # Update index mapping
        del self._id_to_index[removed.experience_id]
        for idx in range(min_index, len(self._buffer)):
            exp = self._buffer[idx]
            self._id_to_index[exp.experience_id] = idx

    def sample(
        self,
        batch_size: int = 32,
        regime: Optional[str] = None,
    ) -> List[Experience]:
        """Sample experiences from the buffer.

        Parameters
        ----------
        batch_size :
            Number of experiences to sample.
        regime :
            If specified, only sample from experiences in this regime.

        Returns
        -------
        List[Experience]
            Sampled experiences.
        """
        if not self._buffer:
            return []

        # Filter by regime if specified
        if regime:
            candidates = [
                exp for exp in self._buffer
                if exp.regime_tag.lower() == regime.lower()
            ]
        else:
            candidates = self._buffer

        if not candidates:
            return []

        # Sample based on prioritization strategy
        if self.prioritization == "uniform":
            indices = self._sample_uniform(candidates, batch_size)
        elif self.prioritization == "proportional":
            indices = self._sample_proportional(candidates, batch_size)
        else:  # rank
            indices = self._sample_rank(candidates, batch_size)

        # Get experiences and increment replay counts
        sampled = []
        for idx in indices:
            exp = candidates[idx]
            exp.replay_count += 1
            sampled.append(exp)

        return sampled

    def _sample_uniform(
        self, candidates: List[Experience], batch_size: int
    ) -> List[int]:
        """Uniform random sampling."""
        n = min(batch_size, len(candidates))
        return list(np.random.choice(len(candidates), size=n, replace=False))

    def _sample_proportional(
        self, candidates: List[Experience], batch_size: int
    ) -> List[int]:
        """Probability proportional to priority."""
        priorities = np.array([exp.priority for exp in candidates])
        probabilities = priorities / priorities.sum()

        n = min(batch_size, len(candidates))
        return list(np.random.choice(
            len(candidates),
            size=n,
            replace=False,
            p=probabilities,
        ))

    def _sample_rank(
        self, candidates: List[Experience], batch_size: int
    ) -> List[int]:
        """Rank-based prioritization (segmented sampling)."""
        # Sort by priority (descending)
        sorted_indices = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i].priority,
            reverse=True,
        )

        n = min(batch_size, len(candidates))
        
        # Probability proportional to rank: P(i) = 1/rank(i)
        ranks = np.arange(1, len(candidates) + 1)
        probabilities = 1.0 / ranks
        probabilities = probabilities / probabilities.sum()

        # Sample from sorted distribution
        sampled_ranks = np.random.choice(
            len(candidates),
            size=n,
            replace=False,
            p=probabilities,
        )

        return [sorted_indices[r] for r in sampled_ranks]

    def get_regime_distribution(self) -> Dict[str, int]:
        """Get the count of experiences per regime.

        Returns
        -------
        Dict[str, int]
            Mapping from regime to count.
        """
        return dict(self._regime_counts)

    def get_high_impact_experiences(
        self, top_k: int = 10
    ) -> List[Experience]:
        """Get experiences with highest absolute reward.

        These include both big wins and big losses - valuable for learning.

        Parameters
        ----------
        top_k :
            Number of high-impact experiences to return.

        Returns
        -------
        List[Experience]
            High-impact experiences sorted by absolute reward.
        """
        sorted_experiences = sorted(
            self._buffer,
            key=lambda e: abs(e.reward),
            reverse=True,
        )
        return sorted_experiences[:top_k]

    def update_priorities(
        self,
        experience_ids: List[str],
        new_priorities: List[float],
    ) -> None:
        """Update priorities for specific experiences.

        This is typically used after learning to adjust priorities based on
        TD-error or other learning signals.

        Parameters
        ----------
        experience_ids :
            List of experience IDs to update.
        new_priorities :
            New priority values (same length as experience_ids).
        """
        if len(experience_ids) != len(new_priorities):
            raise ValueError(
                "experience_ids and new_priorities must have same length"
            )

        for exp_id, new_priority in zip(experience_ids, new_priorities):
            if exp_id in self._id_to_index:
                idx = self._id_to_index[exp_id]
                old_priority = self._buffer[idx].priority
                self._buffer[idx].priority = new_priority
                self._total_priority += new_priority - old_priority

    def get_by_id(self, experience_id: str) -> Optional[Experience]:
        """Retrieve an experience by ID.

        Parameters
        ----------
        experience_id :
            Experience identifier.

        Returns
        -------
        Experience or None
            The experience if found.
        """
        if experience_id in self._id_to_index:
            return self._buffer[self._id_to_index[experience_id]]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics about the buffer state.
        """
        if not self._buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "total_priority": 0.0,
                "avg_priority": 0.0,
                "regime_distribution": {},
            }

        priorities = [e.priority for e in self._buffer]
        rewards = [e.reward for e in self._buffer]
        replay_counts = [e.replay_count for e in self._buffer]

        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "utilization": len(self._buffer) / self.capacity,
            "total_priority": self._total_priority,
            "avg_priority": np.mean(priorities),
            "avg_reward": np.mean(rewards),
            "avg_replay_count": np.mean(replay_counts),
            "regime_distribution": self.get_regime_distribution(),
        }

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._buffer.clear()
        self._id_to_index.clear()
        self._regime_counts.clear()
        self._total_priority = 0.0

    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        """String representation of the buffer."""
        return (
            f"ExperienceReplayBuffer("
            f"size={len(self._buffer)}, "
            f"capacity={self.capacity}, "
            f"prioritization='{self.prioritization}', "
            f"regimes={len(self._regime_counts)})"
        )
