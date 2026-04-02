"""
Walk-Forward Analysis Framework for Robust Strategy Validation

This module implements proper out-of-sample testing methodology to prevent
overfitting and ensure strategies generalize to unseen data.

Key Features:
- Rolling window optimization
- Out-of-sample validation
- Multiple time period testing
- Performance degradation analysis
- Stability metrics

Author: FinAgent Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("WalkForwardAnalysis")


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    train_periods: List[Tuple[datetime, datetime]]
    test_periods: List[Tuple[datetime, datetime]]
    train_metrics: List[Dict[str, float]]
    test_metrics: List[Dict[str, float]]
    
    # Aggregated statistics
    avg_train_sharpe: float
    avg_test_sharpe: float
    sharpe_degradation: float
    
    avg_train_return: float
    avg_test_return: float
    return_degradation: float
    
    consistency_score: float  # How consistent is performance across periods?
    overfitting_indicator: bool  # True if significant degradation detected
    
    def to_dict(self):
        return {
            'train_periods': [(s.isoformat(), e.isoformat()) for s, e in self.train_periods],
            'test_periods': [(s.isoformat(), e.isoformat()) for s, e in self.test_periods],
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'avg_train_sharpe': self.avg_train_sharpe,
            'avg_test_sharpe': self.avg_test_sharpe,
            'sharpe_degradation': self.sharpe_degradation,
            'avg_train_return': self.avg_train_return,
            'avg_test_return': self.avg_test_return,
            'return_degradation': self.return_degradation,
            'consistency_score': self.consistency_score,
            'overfitting_indicator': self.overfitting_indicator
        }


class WalkForwardAnalyzer:
    """
    Conducts rigorous walk-forward analysis to validate strategy robustness
    
    Methodology:
    1. Split data into multiple train/test periods (rolling windows)
    2. Optimize on training period
    3. Validate on subsequent test period
    4. Measure performance degradation
    5. Detect overfitting
    """
    
    def __init__(
        self,
        train_window_days: int = 252,      # 1 year training
        test_window_days: int = 63,         # 3 months testing
        step_days: int = 63,                # Roll forward by 3 months
        min_periods: int = 3                # Minimum number of periods required
    ):
        """
        Initialize walk-forward analyzer
        
        Args:
            train_window_days: Length of training period
            test_window_days: Length of test period
            step_days: How much to roll forward each iteration
            min_periods: Minimum periods needed for valid analysis
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_periods = min_periods
        
        logger.info(f"✅ Walk-forward analyzer initialized")
        logger.debug(f"   Train window: {train_window_days} days")
        logger.debug(f"   Test window: {test_window_days} days")
        logger.debug(f"   Step size: {step_days} days")
    
    def generate_periods(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test period splits
        
        Returns:
            List of tuples: (train_start, train_end, test_start, test_end)
        """
        periods = []
        
        current_start = start_date
        
        while True:
            # Training period
            train_end = current_start + timedelta(days=self.train_window_days)
            
            # Test period
            test_end = train_end + timedelta(days=self.test_window_days)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            periods.append((current_start, train_end, train_end, test_end))
            
            # Roll forward
            current_start += timedelta(days=self.step_days)
        
        if len(periods) < self.min_periods:
            raise ValueError(
                f"Insufficient data: need at least {self.min_periods} periods, "
                f"got {len(periods)}. Use longer history or smaller windows."
            )
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods
    
    def analyze(
        self,
        prices: pd.Series,
        strategy_func,
        optimize_func=None,
        metric_func=None
    ) -> WalkForwardResult:
        """
        Perform complete walk-forward analysis
        
        Args:
            prices: Price series for the asset
            strategy_func: Function that takes (prices, params) and returns trades
            optimize_func: Optional function to optimize parameters on train data
            metric_func: Function to calculate metrics from trades
            
        Returns:
            WalkForwardResult: Comprehensive analysis results
        """
        # Generate periods
        periods = self.generate_periods(prices.index[0], prices.index[-1])
        
        train_metrics = []
        test_metrics = []
        train_sharpes = []
        test_sharpes = []
        train_returns = []
        test_returns = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"\n{'='*60}")
            logger.info(f"PERIOD {i+1}/{len(periods)}")
            logger.info(f"Train: {train_start.date()} → {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} → {test_end.date()}")
            
            # Extract price data for each period
            train_prices = prices.loc[train_start:train_end]
            test_prices = prices.loc[test_start:test_end]
            
            # OPTIMIZE (if function provided)
            optimal_params = {}
            if optimize_func:
                logger.info("Optimizing strategy parameters on train data...")
                optimal_params = optimize_func(train_prices)
                logger.debug(f"Optimal params: {optimal_params}")
            
            # RUN STRATEGY on TRAIN data
            logger.info("Running strategy on train data...")
            train_trades = strategy_func(train_prices, optimal_params)
            
            # CALCULATE METRICS for train
            if metric_func:
                train_metric_dict = metric_func(train_trades)
            else:
                # Default: simple return calculation
                train_metric_dict = {'total_return': self._calculate_return(train_trades)}
            
            train_metrics.append(train_metric_dict)
            train_sharpe = train_metric_dict.get('sharpe_ratio', 0)
            train_return = train_metric_dict.get('total_return', 0)
            train_sharpes.append(train_sharpe)
            train_returns.append(train_return)
            
            logger.info(f"Train Sharpe: {train_sharpe:.2f}, Return: {train_return:.2%}")
            
            # RUN STRATEGY on TEST data (using SAME parameters from train)
            logger.info("Running strategy on test data (out-of-sample)...")
            test_trades = strategy_func(test_prices, optimal_params)
            
            # CALCULATE METRICS for test
            if metric_func:
                test_metric_dict = metric_func(test_trades)
            else:
                test_metric_dict = {'total_return': self._calculate_return(test_trades)}
            
            test_metrics.append(test_metric_dict)
            test_sharpe = test_metric_dict.get('sharpe_ratio', 0)
            test_return = test_metric_dict.get('total_return', 0)
            test_sharpes.append(test_sharpe)
            test_returns.append(test_return)
            
            logger.info(f"Test  Sharpe: {test_sharpe:.2f}, Return: {test_return:.2%}")
        
        # AGGREGATE RESULTS
        avg_train_sharpe = np.mean(train_sharpes)
        avg_test_sharpe = np.mean(test_sharpes)
        sharpe_degradation = avg_train_sharpe - avg_test_sharpe
        
        avg_train_return = np.mean(train_returns)
        avg_test_return = np.mean(test_returns)
        return_degradation = avg_train_return - avg_test_return
        
        # Calculate consistency score (how stable is performance?)
        consistency_score = self._calculate_consistency(test_sharpes)
        
        # Detect overfitting
        overfitting_indicator = sharpe_degradation > 0.5  # More than 0.5 degradation
        
        result = WalkForwardResult(
            train_periods=[(p[0], p[1]) for p in periods],
            test_periods=[(p[2], p[3]) for p in periods],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_degradation=sharpe_degradation,
            avg_train_return=avg_train_return,
            avg_test_return=avg_test_return,
            return_degradation=return_degradation,
            consistency_score=consistency_score,
            overfitting_indicator=overfitting_indicator
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD ANALYSIS COMPLETE")
        logger.info(f"Average Train Sharpe: {avg_train_sharpe:.2f}")
        logger.info(f"Average Test Sharpe:  {avg_test_sharpe:.2f}")
        logger.info(f"Sharpe Degradation:   {sharpe_degradation:.2f}")
        logger.info(f"Consistency Score:    {consistency_score:.2f}")
        logger.info(f"Overfitting Detected: {overfitting_indicator}")
        
        return result
    
    def _calculate_return(self, trades: List) -> float:
        """Calculate total return from trades"""
        if not trades:
            return 0.0
        
        total_pnl = sum(t.pnl for t in trades)
        total_capital = sum(abs(t.entry_price * t.quantity) for t in trades)
        
        return total_pnl / total_capital if total_capital > 0 else 0.0
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """
        Calculate consistency score (inverse of coefficient of variation)
        
        Higher score = more consistent performance
        Score of 1.0 = perfect consistency
        Score < 0.5 = highly inconsistent
        """
        if len(values) < 2:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        # Coefficient of variation
        cv = std_val / abs(mean_val)
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = 1.0 / (1.0 + cv)
        
        return consistency


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing
    
    Methods:
    1. Bootstrap sampling - Random resampling of trades
    2. Path generation - Simulate alternative price paths
    3. Parameter perturbation - Test sensitivity to parameter changes
    """
    
    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"✅ Monte Carlo simulator initialized ({n_simulations} simulations)")
    
    def bootstrap_analysis(
        self,
        original_trades: List,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Bootstrap resampling to estimate metric confidence intervals
        
        Args:
            original_trades: Original trade list
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with statistics and confidence intervals
        """
        n_trades = len(original_trades)
        
        if n_trades < 10:
            raise ValueError("Need at least 10 trades for bootstrap analysis")
        
        simulated_returns = []
        simulated_sharpes = []
        
        for i in range(self.n_simulations):
            # Sample with replacement
            sampled_indices = np.random.choice(n_trades, size=n_trades, replace=True)
            sampled_trades = [original_trades[idx] for idx in sampled_indices]
            
            # Calculate metrics
            total_return = sum(t.pnl for t in sampled_trades)
            total_capital = sum(abs(t.entry_price * t.quantity) for t in sampled_trades)
            ret = total_return / total_capital if total_capital > 0 else 0
            
            simulated_returns.append(ret)
        
        # Calculate statistics
        results = {
            'mean_return': np.mean(simulated_returns),
            'std_return': np.std(simulated_returns),
            'median_return': np.median(simulated_returns),
            'ci_lower': np.percentile(simulated_returns, (1 - confidence_level) * 100),
            'ci_upper': np.percentile(simulated_returns, confidence_level * 100),
            'prob_profit': np.mean([r > 0 for r in simulated_returns]),
            'worst_case_5pct': np.percentile(simulated_returns, 5),
            'best_case_95pct': np.percentile(simulated_returns, 95)
        }
        
        logger.info(f"Bootstrap Analysis ({self.n_simulations} simulations):")
        logger.info(f"  Mean Return: {results['mean_return']:.2%} ± {results['std_return']:.2%}")
        logger.info(f"  95% CI: [{results['ci_lower']:.2%}, {results['ci_upper']:.2%}]")
        logger.info(f"  Probability of Profit: {results['prob_profit']:.1%}")
        
        return results
    
    def path_generation(
        self,
        original_prices: pd.Series,
        n_paths: int = 100
    ) -> List[pd.Series]:
        """
        Generate alternative price paths using geometric Brownian motion
        
        Args:
            original_prices: Historical price series
            n_paths: Number of paths to generate
            
        Returns:
            List of simulated price paths
        """
        # Calculate parameters from historical data
        returns = original_prices.pct_change().dropna()
        
        mu = returns.mean()  # Expected return
        sigma = returns.std()  # Volatility
        
        n_periods = len(original_prices)
        S0 = original_prices.iloc[0]
        
        paths = []
        
        for i in range(n_paths):
            # Generate random shocks
            Z = np.random.normal(0, 1, n_periods)
            
            # Geometric Brownian motion
            path = np.zeros(n_periods)
            path[0] = S0
            
            for t in range(1, n_periods):
                path[t] = path[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z[t])
            
            paths.append(pd.Series(path, index=original_prices.index))
        
        logger.info(f"Generated {n_paths} simulated price paths")
        
        return paths


def create_walk_forward_analyzer(**kwargs) -> WalkForwardAnalyzer:
    """Factory function to create walk-forward analyzer"""
    return WalkForwardAnalyzer(**kwargs)


def create_monte_carlo_simulator(**kwargs) -> MonteCarloSimulator:
    """Factory function to create Monte Carlo simulator"""
    return MonteCarloSimulator(**kwargs)
