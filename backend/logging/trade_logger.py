"""
Trade Logger - Comprehensive Trade Recording System

This module provides persistent logging of all trading activity including:
- Trade execution details
- Signal attribution
- Regime context
- Performance metrics
- Decision rationale

Storage: JSON files with automatic rotation and indexing

Author: FinAgent Team
Version: 1.0.0
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("TradeLogger")


@dataclass
class TradeRecord:
    """Complete record of a single trade"""
    trade_id: str
    symbol: str
    direction: int  # 1=long, -1=short
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    
    # Context
    regime_at_entry: str
    regime_at_exit: Optional[str]
    
    # Signals
    signal_source: str  # What generated the signal
    signal_strength: float
    signal_confidence: float
    
    # Decision details
    decision_rationale: str
    expected_outcome: Optional[str]
    
    # Costs
    commission: float
    slippage: float
    total_cost: float
    
    # Metadata
    created_at: str
    updated_at: str
    
    def to_dict(self):
        return asdict(self)


class TradeLogger:
    """
    Persistent trade logging system
    
    Features:
    - Automatic directory management
    - Daily log rotation
    - Index file for quick lookups
    - Query capabilities
    """
    
    def __init__(self, storage_dir: str = "logs/trades"):
        """
        Initialize trade logger
        
        Args:
            storage_dir: Base directory for trade logs
        """
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "index.json"
        
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.trade_index = self._load_index()
        
        logger.info(f"✅ Trade logger initialized at {self.storage_dir}")
        logger.debug(f"   Indexed trades: {len(self.trade_index)}")
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load trade index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                return {}
        return {}
    
    def _save_index(self):
        """Save trade index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.trade_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _get_daily_log_file(self, date: datetime) -> Path:
        """Get log file path for a specific date"""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_dir / f"trades_{date_str}.json"
    
    def log_trade(self, trade_record: TradeRecord):
        """
        Log a new trade
        
        Args:
            trade_record: Complete trade record
        """
        # Add to index
        self.trade_index[trade_record.trade_id] = {
            'symbol': trade_record.symbol,
            'entry_date': trade_record.entry_date,
            'exit_date': trade_record.exit_date,
            'pnl': trade_record.pnl,
            'regime': trade_record.regime_at_entry,
            'created_at': trade_record.created_at
        }
        
        # Save to daily log
        entry_date = datetime.fromisoformat(trade_record.entry_date)
        log_file = self._get_daily_log_file(entry_date)
        
        # Load existing trades for this day
        daily_trades = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    daily_trades = json.load(f)
            except:
                daily_trades = []
        
        # Add new trade
        daily_trades.append(trade_record.to_dict())
        
        # Save back
        with open(log_file, 'w') as f:
            json.dump(daily_trades, f, indent=2)
        
        # Update index
        self._save_index()
        
        logger.info(f"📝 Logged trade {trade_record.trade_id}: {trade_record.symbol} P&L=${trade_record.pnl:.2f}")
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """
        Update an existing trade (e.g., add exit details)
        
        Args:
            trade_id: Trade ID to update
            updates: Dictionary of fields to update
        """
        if trade_id not in self.trade_index:
            raise ValueError(f"Trade {trade_id} not found")
        
        # Find and update in daily log
        updated = False
        for log_file in self.storage_dir.glob("trades_*.json"):
            if log_file == self.index_file:
                continue
                
            try:
                with open(log_file, 'r') as f:
                    trades = json.load(f)
                
                for i, trade in enumerate(trades):
                    if trade['trade_id'] == trade_id:
                        trades[i].update(updates)
                        trades[i]['updated_at'] = datetime.now().isoformat()
                        
                        with open(log_file, 'w') as f:
                            json.dump(trades, f, indent=2)
                        
                        # Update index
                        self.trade_index[trade_id].update({
                            k: v for k, v in updates.items()
                            if k in ['exit_date', 'pnl', 'regime_at_exit']
                        })
                        self._save_index()
                        
                        updated = True
                        logger.info(f"✏️ Updated trade {trade_id}")
                        break
                
                if updated:
                    break
                    
            except Exception as e:
                logger.error(f"Error updating trade: {e}")
                continue
        
        if not updated:
            logger.warning(f"Trade {trade_id} not found in any log file")
    
    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """
        Retrieve a specific trade by ID
        
        Args:
            trade_id: Trade ID to retrieve
            
        Returns:
            Trade record or None
        """
        if trade_id not in self.trade_index:
            return None
        
        # Search daily logs
        for log_file in self.storage_dir.glob("trades_*.json"):
            if log_file == self.index_file:
                continue
            
            try:
                with open(log_file, 'r') as f:
                    trades = json.load(f)
                
                for trade in trades:
                    if trade['trade_id'] == trade_id:
                        return trade
            except:
                continue
        
        return None
    
    def get_trades_by_date(self, date: datetime) -> List[Dict]:
        """
        Get all trades for a specific date
        
        Args:
            date: Date to query
            
        Returns:
            List of trade records
        """
        log_file = self._get_daily_log_file(date)
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def get_trades_by_symbol(self, symbol: str) -> List[Dict]:
        """
        Get all trades for a specific symbol
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of trade records
        """
        matching_trades = []
        
        for trade_id, metadata in self.trade_index.items():
            if metadata.get('symbol') == symbol:
                trade = self.get_trade(trade_id)
                if trade:
                    matching_trades.append(trade)
        
        return matching_trades
    
    def get_all_trades(self) -> List[Dict]:
        """
        Get all trades across all dates
        
        Returns:
            List of all trade records
        """
        all_trades = []
        
        for log_file in self.storage_dir.glob("trades_*.json"):
            if log_file == self.index_file:
                continue
            
            try:
                with open(log_file, 'r') as f:
                    trades = json.load(f)
                    all_trades.extend(trades)
            except:
                continue
        
        return all_trades
    
    def get_summary_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary statistics for recent trading activity
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        all_trades = self.get_all_trades()
        
        # Filter by date
        recent_trades = [
            t for t in all_trades
            if datetime.fromisoformat(t['entry_date']) >= cutoff
        ]
        
        if not recent_trades:
            return {'message': 'No trades in period'}
        
        # Calculate stats
        total_pnl = sum(t['pnl'] or 0 for t in recent_trades)
        winning_trades = [t for t in recent_trades if (t['pnl'] or 0) > 0]
        losing_trades = [t for t in recent_trades if (t['pnl'] or 0) <= 0]
        
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_costs = sum(t.get('total_cost', 0) for t in recent_trades)
        
        return {
            'period_days': days,
            'total_trades': len(recent_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'total_costs': total_costs
        }
    
    def export_to_csv(self, output_path: str, days: Optional[int] = None):
        """
        Export trades to CSV format
        
        Args:
            output_path: Path to output CSV file
            days: Optional day limit (None = all time)
        """
        import csv
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [
                t for t in self.get_all_trades()
                if datetime.fromisoformat(t['entry_date']) >= cutoff
            ]
        else:
            trades = self.get_all_trades()
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        # Define CSV columns
        fieldnames = [
            'trade_id', 'symbol', 'direction', 'entry_date', 'exit_date',
            'entry_price', 'exit_price', 'quantity', 'pnl',
            'regime_at_entry', 'regime_at_exit',
            'signal_source', 'signal_strength', 'signal_confidence',
            'decision_rationale', 'commission', 'slippage', 'total_cost'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in trades:
                row = {k: trade.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        logger.info(f"📊 Exported {len(trades)} trades to {output_path}")


def create_trade_logger(**kwargs) -> TradeLogger:
    """Factory function to create trade logger"""
    return TradeLogger(**kwargs)
