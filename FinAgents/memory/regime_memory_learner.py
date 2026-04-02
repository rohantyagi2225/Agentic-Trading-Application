"""
Enhanced Memory System for Regime-Specific Pattern Learning

This module extends the existing memory system to enable sophisticated learning
from past trading experiences across different market regimes.

Key Features:
- Regime-tagged memory storage
- Pattern retrieval by market context
- Similarity-based pattern matching
- Performance-weighted memory recall
- Continuous learning from outcomes
- Cross-regime knowledge transfer

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │          Enhanced Memory System                         │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Regime       │  │ Pattern      │  │ Performance  │  │
    │  │ Tagger       │  │ Matcher      │  │ Weighting    │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │         Memory Storage & Retrieval Engine        │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────┐
    │           Memory Applications                           │
    │  - Store successful patterns by regime                  │
    │  - Retrieve similar historical situations               │
    │  - Learn from past mistakes                             │
    │  - Adapt strategies based on memory                     │
    └─────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 1.0.0
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import numpy as np
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("RegimeMemory")


class MemoryType(Enum):
    """Types of trading memories"""
    TRADE_EXECUTION = "trade_execution"
    SIGNAL_GENERATION = "signal_generation"
    REGIME_TRANSITION = "regime_transition"
    STRATEGY_SUCCESS = "strategy_success"
    STRATEGY_FAILURE = "strategy_failure"
    RISK_EVENT = "risk_event"
    MARKET_ANOMALY = "market_anomaly"
    LEARNING_UPDATE = "learning_update"


@dataclass
class TradingMemory:
    """Represents a single trading memory with regime context"""
    memory_id: str
    memory_type: MemoryType
    timestamp: datetime
    regime_context: str  # bull_market, bear_market, high_volatility, etc.
    symbol: str
    strategy_used: str
    outcome: float  # P&L or return
    confidence: float
    decision_details: Dict[str, Any]
    market_conditions: Dict[str, Any]
    signals_present: List[str]
    success: bool
    lessons_learned: str
    similarity_hash: str  # For finding similar situations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            'regime_context': self.regime_context,
            'symbol': self.symbol,
            'strategy_used': self.strategy_used,
            'outcome': self.outcome,
            'confidence': self.confidence,
            'decision_details': {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.decision_details.items()
            },
            'market_conditions': {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.market_conditions.items()
            },
            'signals_present': self.signals_present,
            'success': self.success,
            'lessons_learned': self.lessons_learned,
            'similarity_hash': self.similarity_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingMemory':
        return cls(
            memory_id=data['memory_id'],
            memory_type=MemoryType(data['memory_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            regime_context=data['regime_context'],
            symbol=data['symbol'],
            strategy_used=data['strategy_used'],
            outcome=data['outcome'],
            confidence=data['confidence'],
            decision_details=data['decision_details'],
            market_conditions=data['market_conditions'],
            signals_present=data['signals_present'],
            success=data['success'],
            lessons_learned=data['lessons_learned'],
            similarity_hash=data['similarity_hash']
        )


@dataclass
class PatternCluster:
    """Cluster of similar trading patterns"""
    cluster_id: str
    pattern_type: str
    regime_context: str
    similar_memories: List[str]  # Memory IDs
    avg_outcome: float
    success_rate: float
    key_characteristics: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'pattern_type': self.pattern_type,
            'regime_context': self.regime_context,
            'similar_memories': self.similar_memories,
            'avg_outcome': self.avg_outcome,
            'success_rate': self.success_rate,
            'key_characteristics': self.key_characteristics,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


class RegimeMemoryStorage:
    """Local file-based storage for regime-tagged memories"""
    
    def __init__(self, storage_path: str = "memory_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.memories_file = self.storage_path / "trading_memories.json"
        self.patterns_file = self.storage_path / "pattern_clusters.json"
        self.index_file = self.storage_path / "memory_index.json"
        
        self.memories: List[TradingMemory] = []
        self.pattern_clusters: List[PatternCluster] = []
        self.memory_index: Dict[str, List[str]] = {}  # Regime -> Memory IDs
        
        self._load_storage()
        logger.info(f"✅ Regime memory storage initialized at {self.storage_path}")
    
    def _load_storage(self):
        """Load existing storage from disk"""
        if self.memories_file.exists():
            with open(self.memories_file, 'r') as f:
                data = json.load(f)
                self.memories = [TradingMemory.from_dict(m) for m in data]
        
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                data = json.load(f)
                self.pattern_clusters = [PatternCluster(**p) for p in data]
        
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.memory_index = json.load(f)
        
        logger.debug(f"Loaded {len(self.memories)} memories from storage")
    
    def _save_storage(self):
        """Save storage to disk"""
        with open(self.memories_file, 'w') as f:
            json.dump([m.to_dict() for m in self.memories], f, indent=2)
        
        with open(self.patterns_file, 'w') as f:
            json.dump([p.to_dict() for p in self.pattern_clusters], f, indent=2)
        
        with open(self.index_file, 'w') as f:
            json.dump(self.memory_index, f, indent=2)
    
    def store_memory(self, memory: TradingMemory):
        """Store a new trading memory"""
        self.memories.append(memory)
        
        # Update index
        if memory.regime_context not in self.memory_index:
            self.memory_index[memory.regime_context] = []
        self.memory_index[memory.regime_context].append(memory.memory_id)
        
        self._save_storage()
        logger.debug(f"Stored memory: {memory.memory_id} ({memory.regime_context})")
    
    def get_memories_by_regime(self, regime: str) -> List[TradingMemory]:
        """Retrieve all memories for a specific regime"""
        memory_ids = self.memory_index.get(regime, [])
        return [m for m in self.memories if m.memory_id in memory_ids]
    
    def get_similar_memories(
        self,
        current_conditions: Dict[str, Any],
        regime: str,
        limit: int = 10
    ) -> List[TradingMemory]:
        """Find historically similar situations"""
        # Filter by regime first
        regime_memories = self.get_memories_by_regime(regime)
        
        if not regime_memories:
            return []
        
        # Calculate similarity scores
        similarities = []
        for memory in regime_memories:
            similarity = self._calculate_similarity(current_conditions, memory.market_conditions)
            similarities.append((memory, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in similarities[:limit]]
    
    def _calculate_similarity(
        self,
        conditions1: Dict[str, Any],
        conditions2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two market condition sets"""
        # Simple feature-based similarity
        common_keys = set(conditions1.keys()) & set(conditions2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = conditions1[key]
            val2 = conditions2[key]
            
            # Handle numeric values
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize difference
                max_val = max(abs(val1), abs(val2), 1e-10)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)
            # Handle categorical values
            elif isinstance(val1, str) and isinstance(val2, str):
                sim = 1.0 if val1 == val2 else 0.0
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_successful_patterns(
        self,
        regime: str,
        min_success_rate: float = 0.6,
        min_samples: int = 5
    ) -> List[PatternCluster]:
        """Get successful pattern clusters for a regime"""
        return [
            p for p in self.pattern_clusters
            if p.regime_context == regime and
            p.success_rate >= min_success_rate and
            len(p.similar_memories) >= min_samples
        ]
    
    def add_pattern_cluster(self, cluster: PatternCluster):
        """Add a new pattern cluster"""
        self.pattern_clusters.append(cluster)
        self._save_storage()


class RegimeTagger:
    """Automatically tag memories with regime context"""
    
    def __init__(self):
        self.regime_keywords = {
            'bull_market': ['uptrend', 'golden cross', 'higher highs', 'bullish'],
            'bear_market': ['downtrend', 'death cross', 'lower lows', 'bearish'],
            'high_volatility': ['volatile', 'large swings', 'uncertainty', 'VIX spike'],
            'low_volatility': ['calm', 'stable', 'narrow range', 'consolidation'],
            'crash': ['crash', 'panic', 'flash crash', 'plunge'],
            'recovery': ['recovery', 'bounce back', 'rally', 'rebound'],
            'sideways': ['sideways', 'range-bound', 'consolidation', 'choppy']
        }
    
    def tag_memory(
        self,
        regime_result: Any,
        trade_details: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> str:
        """Determine appropriate regime tag for memory"""
        # Primary regime from detection result
        primary_regime = regime_result.primary_regime.value
        
        # Validate with keyword matching
        text_context = " ".join([
            str(v) for v in trade_details.values()
            if isinstance(v, str)
        ]).lower()
        
        for regime, keywords in self.regime_keywords.items():
            if any(keyword in text_context for keyword in keywords):
                return regime
        
        return primary_regime
    
    def extract_key_features(
        self,
        market_conditions: Dict[str, Any]
    ) -> List[str]:
        """Extract key features for similarity matching"""
        features = []
        
        # Volatility feature
        vol = market_conditions.get('realized_volatility', 0)
        if vol > 0.3:
            features.append('extreme_volatility')
        elif vol > 0.2:
            features.append('high_volatility')
        elif vol < 0.1:
            features.append('low_volatility')
        
        # Trend feature
        trend = market_conditions.get('annualized_trend', 0)
        if trend > 0.2:
            features.append('strong_uptrend')
        elif trend > 0.05:
            features.append('moderate_uptrend')
        elif trend < -0.2:
            features.append('strong_downtrend')
        elif trend < -0.05:
            features.append('moderate_downtrend')
        
        # Momentum feature
        momentum = market_conditions.get('momentum_score', 0)
        if momentum > 0.5:
            features.append('strong_momentum')
        elif momentum > 0.2:
            features.append('positive_momentum')
        elif momentum < -0.5:
            features.append('strong_negative_momentum')
        elif momentum < -0.2:
            features.append('negative_momentum')
        
        return features


class PerformanceWeightedRetriever:
    """Retrieve memories weighted by historical performance"""
    
    def __init__(self, storage: RegimeMemoryStorage):
        self.storage = storage
    
    def retrieve_weighted_memories(
        self,
        current_conditions: Dict[str, Any],
        regime: str,
        min_confidence: float = 0.5
    ) -> List[Tuple[TradingMemory, float]]:
        """
        Retrieve memories with performance weighting
        
        Returns:
            List of (memory, weight) tuples sorted by weight descending
        """
        similar_memories = self.storage.get_similar_memories(
            current_conditions, regime
        )
        
        weighted_results = []
        for memory in similar_memories:
            if memory.confidence < min_confidence:
                continue
            
            # Calculate weight based on:
            # 1. Similarity (already filtered)
            # 2. Historical outcome
            # 3. Confidence
            # 4. Recency
            
            similarity = self.storage._calculate_similarity(
                current_conditions, memory.market_conditions
            )
            
            outcome_weight = max(0, memory.outcome)  # Positive outcomes weighted higher
            confidence_weight = memory.confidence
            
            # Recency decay (older memories weighted less)
            days_old = (datetime.now() - memory.timestamp).days
            recency_weight = np.exp(-days_old / 365)  # 1 year half-life
            
            # Combined weight
            total_weight = (
                similarity * 0.3 +
                outcome_weight * 0.3 +
                confidence_weight * 0.2 +
                recency_weight * 0.2
            )
            
            weighted_results.append((memory, total_weight))
        
        # Sort by weight
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results


class RegimeMemoryLearner:
    """
    Main class for regime-specific memory-based learning
    
    Usage:
        learner = RegimeMemoryLearner()
        
        # After each trade, store memory
        learner.store_trade_memory(
            regime_result=regime,
            trade_details=trade,
            outcome=pnl,
            signals=signals
        )
        
        # Before making decision, retrieve relevant memories
        relevant_memories = learner.retrieve_relevant_memories(
            current_regime=current_regime,
            market_conditions=conditions
        )
        
        # Get recommendations from memory
        recommendation = learner.get_memory_recommendation(relevant_memories)
    """
    
    def __init__(self, storage_path: str = "memory_storage"):
        self.storage = RegimeMemoryStorage(storage_path)
        self.tagger = RegimeTagger()
        self.retriever = PerformanceWeightedRetriever(self.storage)
        
        logger.info("✅ Regime memory learner initialized")
    
    def store_trade_memory(
        self,
        regime_result: Any,
        trade_details: Dict[str, Any],
        market_conditions: Dict[str, Any],
        signals: List[str],
        outcome: float,
        confidence: float
    ):
        """Store a completed trade in memory"""
        
        # Generate memory ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000):04d}"
        
        # Determine regime tag
        regime_tag = self.tagger.tag_memory(
            regime_result, trade_details, market_conditions
        )
        
        # Extract key features
        key_features = self.tagger.extract_key_features(market_conditions)
        
        # Create similarity hash
        similarity_hash = self._create_similarity_hash(regime_tag, key_features)
        
        # Determine success
        success = outcome > 0
        
        # Generate lessons learned
        lessons = self._generate_lessons_learned(success, outcome, key_features)
        
        # Create memory
        memory = TradingMemory(
            memory_id=memory_id,
            memory_type=MemoryType.TRADE_EXECUTION,
            timestamp=datetime.now(),
            regime_context=regime_tag,
            symbol=trade_details.get('symbol', 'UNKNOWN'),
            strategy_used=trade_details.get('strategy', 'unknown'),
            outcome=outcome,
            confidence=confidence,
            decision_details=trade_details,
            market_conditions=market_conditions,
            signals_present=signals,
            success=success,
            lessons_learned=lessons,
            similarity_hash=similarity_hash
        )
        
        # Store memory
        self.storage.store_memory(memory)
        
        logger.info(f"💾 Stored trade memory: {memory_id} | Outcome: {outcome:.2%} | Success: {success}")
        
        # Periodically update pattern clusters
        if len(self.storage.memories) % 50 == 0:
            self._update_pattern_clusters()
    
    def retrieve_relevant_memories(
        self,
        current_regime: str,
        market_conditions: Dict[str, Any],
        limit: int = 10
    ) -> List[TradingMemory]:
        """Retrieve most relevant historical memories"""
        
        # Get performance-weighted memories
        weighted_memories = self.retriever.retrieve_weighted_memories(
            market_conditions, current_regime
        )
        
        # Return top N memories
        return [m for m, _ in weighted_memories[:limit]]
    
    def get_memory_recommendation(
        self,
        relevant_memories: List[TradingMemory]
    ) -> Dict[str, Any]:
        """Generate recommendation based on retrieved memories"""
        
        if not relevant_memories:
            return {
                'recommendation': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'reasoning': 'No relevant historical patterns found'
            }
        
        # Analyze successful vs unsuccessful memories
        successful = [m for m in relevant_memories if m.success]
        unsuccessful = [m for m in relevant_memories if not m.success]
        
        success_rate = len(successful) / len(relevant_memories)
        avg_outcome = np.mean([m.outcome for m in relevant_memories])
        
        # Extract common patterns in successful trades
        successful_signals = []
        for m in successful:
            successful_signals.extend(m.signals_present)
        
        signal_counts = {}
        for sig in successful_signals:
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
        
        top_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate recommendation
        if success_rate > 0.6 and avg_outcome > 0.02:
            recommendation = 'FAVORABLE'
            reasoning = f"Historical patterns show {success_rate:.0%} success rate with average outcome of {avg_outcome:.2%}"
        elif success_rate < 0.4:
            recommendation = 'UNFAVORABLE'
            reasoning = f"Historical patterns show only {success_rate:.0%} success rate - exercise caution"
        else:
            recommendation = 'NEUTRAL'
            reasoning = f"Mixed historical results ({success_rate:.0%} success rate) - proceed with standard risk management"
        
        return {
            'recommendation': recommendation,
            'confidence': success_rate,
            'expected_outcome': avg_outcome,
            'reasoning': reasoning,
            'top_signals': [s[0] for s in top_signals],
            'sample_size': len(relevant_memories)
        }
    
    def _create_similarity_hash(
        self,
        regime: str,
        features: List[str]
    ) -> str:
        """Create hash for quick similarity matching"""
        key_string = f"{regime}_{'_'.join(sorted(features))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_lessons_learned(
        self,
        success: bool,
        outcome: float,
        features: List[str]
    ) -> str:
        """Generate lesson learned from trade outcome"""
        if success and outcome > 0.1:
            return "Large winner - conditions aligned favorably"
        elif success:
            return "Successful trade - pattern worth remembering"
        elif outcome < -0.05:
            return "Significant loss - review entry criteria and risk management"
        else:
            return "Small loss - acceptable outcome, review for improvement"
    
    def _update_pattern_clusters(self):
        """Periodically update pattern clusters from stored memories"""
        logger.info("Updating pattern clusters...")
        
        # Group memories by regime and features
        regime_groups = {}
        for memory in self.storage.memories:
            key = (memory.regime_context, memory.similarity_hash)
            if key not in regime_groups:
                regime_groups[key] = []
            regime_groups[key].append(memory)
        
        # Create/update clusters
        for (regime, hash_value), memories in regime_groups.items():
            if len(memories) < 3:
                continue  # Need minimum samples
            
            # Calculate cluster statistics
            outcomes = [m.outcome for m in memories]
            successes = sum(1 for m in memories if m.success)
            
            avg_outcome = np.mean(outcomes)
            success_rate = successes / len(memories)
            
            # Extract key characteristics
            all_signals = []
            for m in memories:
                all_signals.extend(m.signals_present)
            
            signal_freq = {}
            for sig in all_signals:
                signal_freq[sig] = signal_freq.get(sig, 0) + 1
            
            key_characteristics = {
                'common_signals': sorted(signal_freq.items(), key=lambda x: x[1], reverse=True)[:5],
                'avg_confidence': np.mean([m.confidence for m in memories]),
                'avg_holding_period': np.mean([
                    (m.decision_details.get('exit_date', datetime.now()) - m.decision_details.get('entry_date', datetime.now())).days
                    for m in memories
                ])
            }
            
            # Create cluster
            cluster = PatternCluster(
                cluster_id=f"cluster_{regime}_{hash_value[:8]}",
                pattern_type=hash_value,
                regime_context=regime,
                similar_memories=[m.memory_id for m in memories],
                avg_outcome=avg_outcome,
                success_rate=success_rate,
                key_characteristics=key_characteristics,
                created_at=min(m.timestamp for m in memories),
                last_updated=datetime.now()
            )
            
            self.storage.add_pattern_cluster(cluster)
        
        logger.info(f"Updated {len(regime_groups)} pattern clusters")
    
    def get_regime_statistics(self, regime: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a regime"""
        regime_memories = self.storage.get_memories_by_regime(regime)
        
        if not regime_memories:
            return {'error': 'No data available for this regime'}
        
        outcomes = [m.outcome for m in regime_memories]
        confidences = [m.confidence for m in regime_memories]
        successes = sum(1 for m in regime_memories if m.success)
        
        return {
            'total_trades': len(regime_memories),
            'success_rate': successes / len(regime_memories),
            'avg_outcome': np.mean(outcomes),
            'std_outcome': np.std(outcomes),
            'best_trade': max(outcomes),
            'worst_trade': min(outcomes),
            'avg_confidence': np.mean(confidences),
            'profit_factor': (
                sum(o for o in outcomes if o > 0) / abs(sum(o for o in outcomes if o < 0))
                if sum(o for o in outcomes if o < 0) != 0 else float('inf')
            )
        }
    
    def export_memories(self, filepath: str, regime: Optional[str] = None):
        """Export memories to JSON file"""
        if regime:
            memories = self.storage.get_memories_by_regime(regime)
        else:
            memories = self.storage.memories
        
        with open(filepath, 'w') as f:
            json.dump([m.to_dict() for m in memories], f, indent=2)
        
        logger.info(f"Exported {len(memories)} memories to {filepath}")


def create_regime_memory_learner(storage_path: str = "memory_storage") -> RegimeMemoryLearner:
    """Factory function to create regime memory learner"""
    return RegimeMemoryLearner(storage_path)
