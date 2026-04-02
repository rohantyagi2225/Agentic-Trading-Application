"""
Signal Logger - Decision & Signal Recording System

Records all trading signals and AI decisions with full context including:
- Signal details (source, type, strength, confidence)
- Regime context at time of signal
- Contributing factors
- Final decision and rationale
- Outcome tracking

Storage: JSON files with regime-based organization

Author: FinAgent Team
Version: 1.0.0
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("SignalLogger")


@dataclass
class SignalRecord:
    """Complete record of a trading signal"""
    signal_id: str
    timestamp: str
    symbol: str
    
    # Signal details
    source: str  # technical, sentiment, macro, fused
    signal_type: str  # bullish, bearish, neutral
    strength: float  # 0-1
    confidence: float  # 0-1
    
    # Context
    market_regime: str
    volatility_regime: str
    
    # Contributing signals (for fused signals)
    contributing_signals: Optional[List[Dict]]
    
    # Decision
    final_decision: str  # BUY, SELL, HOLD
    decision_confidence: float
    rationale: str
    
    # Expected outcome
    expected_direction: str
    expected_timeframe: str
    target_return: Optional[float]
    stop_loss_level: Optional[float]
    
    # Actual outcome (filled later)
    actual_return: Optional[float]
    outcome_correct: Optional[bool]
    
    # Metadata
    created_at: str
    
    def to_dict(self):
        return asdict(self)


class SignalLogger:
    """
    Persistent signal and decision logging
    
    Features:
    - Regime-based organization
    - Outcome tracking
    - Performance analytics
    - Pattern analysis
    """
    
    def __init__(self, storage_dir: str = "logs/signals"):
        """
        Initialize signal logger
        
        Args:
            storage_dir: Base directory for signal logs
        """
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "index.json"
        
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.signal_index = self._load_index()
        
        logger.info(f"✅ Signal logger initialized at {self.storage_dir}")
        logger.debug(f"   Indexed signals: {len(self.signal_index)}")
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load signal index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                return {}
        return {}
    
    def _save_index(self):
        """Save signal index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.signal_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _get_regime_log_file(self, regime: str) -> Path:
        """Get log file for a specific regime"""
        regime_safe = regime.lower().replace(' ', '_').replace('-', '_')
        return self.storage_dir / f"signals_{regime_safe}.json"
    
    def log_signal(self, signal_record: SignalRecord):
        """
        Log a new signal
        
        Args:
            signal_record: Complete signal record
        """
        # Add to index
        self.signal_index[signal_record.signal_id] = {
            'symbol': signal_record.symbol,
            'timestamp': signal_record.timestamp,
            'signal_type': signal_record.signal_type,
            'final_decision': signal_record.final_decision,
            'market_regime': signal_record.market_regime,
            'created_at': signal_record.created_at
        }
        
        # Save to regime-specific log
        log_file = self._get_regime_log_file(signal_record.market_regime)
        
        # Load existing signals for this regime
        regime_signals = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    regime_signals = json.load(f)
            except:
                regime_signals = []
        
        # Add new signal
        regime_signals.append(signal_record.to_dict())
        
        # Save back
        with open(log_file, 'w') as f:
            json.dump(regime_signals, f, indent=2)
        
        # Update index
        self._save_index()
        
        logger.info(f"📡 Logged signal {signal_record.signal_id}: {signal_record.final_decision} {signal_record.symbol}")
    
    def update_outcome(self, signal_id: str, actual_return: float):
        """
        Update signal with actual outcome
        
        Args:
            signal_id: Signal ID to update
            actual_return: Realized return
        """
        if signal_id not in self.signal_index:
            raise ValueError(f"Signal {signal_id} not found")
        
        metadata = self.signal_index[signal_id]
        log_file = self._get_regime_log_file(metadata['market_regime'])
        
        try:
            with open(log_file, 'r') as f:
                signals = json.load(f)
            
            for i, signal in enumerate(signals):
                if signal['signal_id'] == signal_id:
                    signals[i]['actual_return'] = actual_return
                    
                    # Determine if prediction was correct
                    expected_dir = signal['expected_direction']
                    if expected_dir == 'up':
                        signals[i]['outcome_correct'] = actual_return > 0
                    elif expected_dir == 'down':
                        signals[i]['outcome_correct'] = actual_return < 0
                    else:
                        signals[i]['outcome_correct'] = True  # Neutral always correct
                    
                    with open(log_file, 'w') as f:
                        json.dump(signals, f, indent=2)
                    
                    logger.info(f"✏️ Updated outcome for signal {signal_id}")
                    break
                    
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
    
    def get_signals_by_regime(self, regime: str) -> List[Dict]:
        """
        Get all signals for a specific regime
        
        Args:
            regime: Market regime to filter by
            
        Returns:
            List of signal records
        """
        log_file = self._get_regime_log_file(regime)
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def get_signals_by_symbol(self, symbol: str) -> List[Dict]:
        """
        Get all signals for a specific symbol
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of signal records
        """
        matching = []
        
        for log_file in self.storage_dir.glob("signals_*.json"):
            if log_file == self.index_file:
                continue
            
            try:
                with open(log_file, 'r') as f:
                    signals = json.load(f)
                
                for signal in signals:
                    if signal['symbol'] == symbol:
                        matching.append(signal)
            except:
                continue
        
        return matching
    
    def get_accuracy_stats(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate signal accuracy statistics
        
        Args:
            regime: Optional regime filter (None = all regimes)
            
        Returns:
            Dictionary of accuracy statistics
        """
        if regime:
            signals = self.get_signals_by_regime(regime)
        else:
            signals = []
            for log_file in self.storage_dir.glob("signals_*.json"):
                if log_file == self.index_file:
                    continue
                try:
                    with open(log_file, 'r') as f:
                        signals.extend(json.load(f))
                except:
                    continue
        
        # Filter signals with outcomes
        signals_with_outcomes = [s for s in signals if s.get('outcome_correct') is not None]
        
        if not signals_with_outcomes:
            return {'message': 'No signals with outcomes'}
        
        # Overall accuracy
        correct = sum(1 for s in signals_with_outcomes if s['outcome_correct'])
        total = len(signals_with_outcomes)
        accuracy = correct / total if total > 0 else 0
        
        # By decision type
        by_decision = {}
        for decision in ['BUY', 'SELL', 'HOLD']:
            decision_signals = [s for s in signals_with_outcomes if s['final_decision'] == decision]
            if decision_signals:
                dec_correct = sum(1 for s in decision_signals if s['outcome_correct'])
                by_decision[decision] = {
                    'count': len(decision_signals),
                    'accuracy': dec_correct / len(decision_signals)
                }
        
        # Average returns
        avg_return = sum(s.get('actual_return', 0) for s in signals_with_outcomes) / total
        
        return {
            'total_signals': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'by_decision': by_decision
        }


def create_signal_logger(**kwargs) -> SignalLogger:
    """Factory function to create signal logger"""
    return SignalLogger(**kwargs)
