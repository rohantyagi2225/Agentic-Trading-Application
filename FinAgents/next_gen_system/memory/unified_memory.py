"""
Unified Memory System for Multi-Agent Learning
==============================================

Implements three types of memory:
1. Episodic: Past trades, outcomes, market conditions
2. Semantic: Market knowledge, relationships, facts
3. Procedural: Strategies, heuristics, decision rules

Features:
- Experience replay
- Pattern recognition
- Reinforcement learning feedback
- Continuous improvement loop
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Episodic memory of a trading experience"""
    episode_id: str
    timestamp: datetime
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    market_context: Dict[str, Any]
    agent_decisions: List[Dict[str, Any]]
    outcome_rating: float  # -1 to 1
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class KnowledgeNode:
    """Semantic memory node"""
    concept: str
    category: str
    attributes: Dict[str, Any]
    relationships: List[Tuple[str, str]]  # (related_concept, relationship_type)
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)


class UnifiedMemorySystem:
    """
    Unified memory system for multi-agent learning
    
    Enables agents to learn from past experiences and improve
    decision-making over time.
    """
    
    def __init__(
        self,
        memory_id: str = "main",
        max_episodes: int = 10000,
        learning_rate: float = 0.1,
    ):
        """
        Initialize memory system
        
        Args:
            memory_id: Unique identifier
            max_episodes: Maximum episodic memories to store
            learning_rate: Learning rate for updates
        """
        self.memory_id = memory_id
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        
        # Episodic memory
        self.episodes: deque[Episode] = deque(maxlen=max_episodes)
        
        # Semantic memory
        self.semantic_network: Dict[str, KnowledgeNode] = {}
        
        # Procedural memory (strategy parameters)
        self.procedural_memory: Dict[str, Dict[str, Any]] = {
            "strategy_params": {},
            "decision_rules": {},
            "heuristics": {},
        }
        
        # Experience indices
        self.symbol_index: Dict[str, List[int]] = {}
        self.outcome_index: Dict[str, List[int]] = {}
        
        logger.info(f"UnifiedMemorySystem {memory_id} initialized")
    
    def store_episode(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        market_context: Dict[str, Any],
        agent_decisions: List[Dict[str, Any]],
        exit_price: Optional[float] = None,
        size: float = 0,
        pnl: Optional[float] = None,
    ) -> str:
        """Store a trading episode"""
        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
        
        # Calculate outcome rating
        if pnl is not None:
            outcome_rating = np.tanh(pnl / (entry_price * size + 1e-6))
        else:
            outcome_rating = 0.0
        
        episode = Episode(
            episode_id=episode_id,
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            market_context=market_context,
            agent_decisions=agent_decisions,
            outcome_rating=outcome_rating,
        )
        
        self.episodes.append(episode)
        
        # Update indices
        if symbol not in self.symbol_index:
            self.symbol_index[symbol] = []
        self.symbol_index[symbol].append(len(self.episodes) - 1)
        
        outcome_key = "positive" if outcome_rating > 0.1 else "negative" if outcome_rating < -0.1 else "neutral"
        if outcome_key not in self.outcome_index:
            self.outcome_index[outcome_key] = []
        self.outcome_index[outcome_key].append(len(self.episodes) - 1)
        
        # Extract lessons
        self._extract_lessons(episode)
        
        logger.debug(f"Stored episode {episode_id} with outcome {outcome_rating:.3f}")
        
        return episode_id
    
    def retrieve_similar_episodes(
        self,
        symbol: str,
        market_context: Dict[str, Any],
        k: int = 5,
    ) -> List[Episode]:
        """Retrieve k most similar episodes"""
        # Get episodes for this symbol
        symbol_indices = self.symbol_index.get(symbol, [])
        
        if not symbol_indices:
            return []
        
        episodes_list = list(self.episodes)
        symbol_episodes = [episodes_list[i] for i in symbol_indices]
        
        # Score by similarity
        scored = []
        for ep in symbol_episodes:
            similarity = self._calculate_similarity(ep.market_context, market_context)
            scored.append((similarity, ep))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [ep for _, ep in scored[:k]]
    
    def _calculate_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two market contexts"""
        # Simple feature-based similarity
        features = ["volatility", "trend", "volume_ratio", "sentiment"]
        
        sims = []
        for feat in features:
            val1 = context1.get(feat, 0)
            val2 = context2.get(feat, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                sim = 1 / (1 + diff)
                sims.append(sim)
        
        return np.mean(sims) if sims else 0.0
    
    def _extract_lessons(self, episode: Episode):
        """Extract lessons learned from episode"""
        if episode.outcome_rating > 0.5:
            episode.lessons_learned.append("Successful trade pattern identified")
        elif episode.outcome_rating < -0.5:
            episode.lessons_learned.append("Loss pattern to avoid in future")
        
        # Update procedural memory
        if episode.action in ["BUY", "SELL"]:
            strategy_key = f"{episode.action}_{episode.symbol}"
            
            if strategy_key not in self.procedural_memory["strategy_params"]:
                self.procedural_memory["strategy_params"][strategy_key] = {
                    "success_count": 0,
                    "fail_count": 0,
                    "avg_pnl": 0,
                }
            
            params = self.procedural_memory["strategy_params"][strategy_key]
            
            # Exponential moving average update
            alpha = self.learning_rate
            if episode.pnl and episode.pnl > 0:
                params["success_count"] += 1
                params["avg_pnl"] = (1 - alpha) * params["avg_pnl"] + alpha * episode.pnl
            elif episode.pnl and episode.pnl < 0:
                params["fail_count"] += 1
                params["avg_pnl"] = (1 - alpha) * params["avg_pnl"] + alpha * episode.pnl
    
    def add_knowledge(
        self,
        concept: str,
        category: str,
        attributes: Dict[str, Any],
        relationships: Optional[List[Tuple[str, str]]] = None,
        confidence: float = 0.8,
    ):
        """Add semantic knowledge"""
        node = KnowledgeNode(
            concept=concept,
            category=category,
            attributes=attributes,
            relationships=relationships or [],
            confidence=confidence,
        )
        
        self.semantic_network[concept] = node
        logger.debug(f"Added knowledge: {concept} ({category})")
    
    def get_knowledge(self, concept: str) -> Optional[KnowledgeNode]:
        """Retrieve semantic knowledge"""
        return self.semantic_network.get(concept)
    
    def update_strategy_params(
        self,
        strategy_name: str,
        updates: Dict[str, Any],
    ):
        """Update procedural memory for a strategy"""
        if strategy_name not in self.procedural_memory["strategy_params"]:
            self.procedural_memory["strategy_params"][strategy_name] = {}
        
        # Smooth update
        for key, value in updates.items():
            current = self.procedural_memory["strategy_params"][strategy_name].get(key, value)
            
            if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                updated = (1 - self.learning_rate) * current + self.learning_rate * value
                self.procedural_memory["strategy_params"][strategy_name][key] = updated
            else:
                self.procedural_memory["strategy_params"][strategy_name][key] = value
    
    def get_experience_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences"""
        if not self.episodes:
            return {"total_episodes": 0}
        
        episodes_list = list(self.episodes)
        
        total_pnl = sum(ep.pnl for ep in episodes_list if ep.pnl is not None)
        winning = sum(1 for ep in episodes_list if ep.pnl and ep.pnl > 0)
        losing = sum(1 for ep in episodes_list if ep.pnl and ep.pnl < 0)
        
        win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0
        
        return {
            "total_episodes": len(episodes_list),
            "total_pnl": total_pnl,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": win_rate,
            "avg_outcome_rating": np.mean([ep.outcome_rating for ep in episodes_list]),
            "symbols_traded": len(self.symbol_index),
        }
    
    def export_memories(self, filepath: str):
        """Export memories to file"""
        data = {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "timestamp": ep.timestamp.isoformat(),
                    "symbol": ep.symbol,
                    "action": ep.action,
                    "pnl": ep.pnl,
                    "outcome_rating": ep.outcome_rating,
                    "lessons": ep.lessons_learned,
                }
                for ep in self.episodes
            ],
            "semantic_network": {
                k: {
                    "concept": v.concept,
                    "category": v.category,
                    "attributes": v.attributes,
                    "confidence": v.confidence,
                }
                for k, v in self.semantic_network.items()
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported memories to {filepath}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get memory system state"""
        return {
            "memory_id": self.memory_id,
            "episodes_stored": len(self.episodes),
            "semantic_nodes": len(self.semantic_network),
            "strategy_params_count": len(self.procedural_memory["strategy_params"]),
            "experience_stats": self.get_experience_statistics(),
        }
