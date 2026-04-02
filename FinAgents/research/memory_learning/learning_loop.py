"""Feedback-Based Learning System for financial agents.

This module implements a comprehensive learning loop that processes trade
outcomes, generates lessons, calibrates confidence, and detects performance
decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from FinAgents.research.domain_agents.base_agent import (
    Action,
    ActionType,
    LearningUpdate,
    ReasoningResult,
    ResearchAgent,
)
from FinAgents.research.memory_learning.trade_memory import (
    StrategyPerformance,
    TradeMemoryStore,
    TradeRecord,
)
from FinAgents.research.memory_learning.experience_replay import (
    Experience,
    ExperienceReplayBuffer,
)
from FinAgents.research.memory_learning.reward_engine import (
    RewardBreakdown,
    RewardEngine,
)

if TYPE_CHECKING:
    pass  # Avoid circular imports


@dataclass
class PostTradeReport:
    """Report generated after post-trade analysis.

    Attributes
    ----------
    trade_id :
        Identifier of the analyzed trade.
    predicted_direction :
        Agent's predicted direction ('UP' or 'DOWN').
    actual_direction :
        Actual outcome direction.
    prediction_correct :
        Whether the prediction was correct.
    pnl :
        Profit/loss from the trade.
    reward :
        Computed reward signal.
    confidence_was :
        Agent's confidence at entry.
    confidence_should_be :
        Retrospectively optimal confidence.
    lessons :
        List of learned lessons.
    parameter_suggestions :
        Suggested parameter adjustments.
    """

    trade_id: str
    predicted_direction: str
    actual_direction: str
    prediction_correct: bool
    pnl: float
    reward: float
    confidence_was: float
    confidence_should_be: float
    lessons: List[str] = field(default_factory=list)
    parameter_suggestions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningStepResult:
    """Result of a single learning step.

    Attributes
    ----------
    parameter_updates :
        Aggregated parameter updates.
    confidence_adjustment :
        Confidence calibration adjustment.
    experiences_replayed :
        Number of experiences processed.
    avg_reward :
        Average reward in the batch.
    lessons :
        Learned lessons from the batch.
    """

    parameter_updates: Dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0
    experiences_replayed: int = 0
    avg_reward: float = 0.0
    lessons: List[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result of confidence calibration analysis.

    Attributes
    ----------
    current_calibration_error :
        Difference between stated confidence and actual accuracy.
    adjustment :
        Recommended adjustment to confidence calibration.
    overconfident :
        Whether the agent is overconfident.
    details :
        Additional calibration details.
    """

    current_calibration_error: float = 0.0
    adjustment: float = 0.0
    overconfident: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecayReport:
    """Report on performance decay detection.

    Attributes
    ----------
    decaying :
        Whether performance decay is detected.
    decay_rate :
        Estimated rate of performance decline.
    window_analyzed :
        Number of trades in the analysis window.
    recent_sharpe :
        Recent Sharpe ratio.
    historical_sharpe :
        Historical average Sharpe ratio.
    recommendation :
        Recommended action.
    """

    decaying: bool = False
    decay_rate: float = 0.0
    window_analyzed: int = 0
    recent_sharpe: float = 0.0
    historical_sharpe: float = 0.0
    recommendation: str = ""


@dataclass
class ReviewReport:
    """Comprehensive periodic review report.

    Attributes
    ----------
    trades_reviewed :
        Number of trades in the review period.
    overall_accuracy :
        Overall prediction accuracy.
    strategy_performance :
        Performance by strategy.
    regime_performance :
        Performance by regime.
    decay_detected :
        Whether performance decay was detected.
    recommendations :
        List of recommendations.
    """

    trades_reviewed: int = 0
    overall_accuracy: float = 0.0
    strategy_performance: Dict[str, StrategyPerformance] = field(default_factory=dict)
    regime_performance: Dict[str, Any] = field(default_factory=dict)
    decay_detected: bool = False
    recommendations: List[str] = field(default_factory=list)


class LearningLoop:
    """Feedback-based learning system for trading agents.

    This class orchestrates the learning process by analyzing trade outcomes,
    generating lessons, calibrating confidence, and detecting performance issues.

    Parameters
    ----------
    trade_memory :
        Trade memory store for accessing historical trades.
    replay_buffer :
        Experience replay buffer for sampling learning experiences.
    reward_engine :
        Reward engine for computing reward signals.
    config :
        Configuration dictionary with learning parameters.
    """

    def __init__(
        self,
        trade_memory: TradeMemoryStore,
        replay_buffer: ExperienceReplayBuffer,
        reward_engine: RewardEngine,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the learning loop."""
        self.trade_memory = trade_memory
        self.replay_buffer = replay_buffer
        self.reward_engine = reward_engine

        # Default configuration
        default_config = {
            "learning_rate": 0.01,
            "review_period": 50,  # Review every N trades
            "confidence_adjustment_rate": 0.05,
            "decay_detection_window": 100,
            "decay_threshold": 0.5,  # Flag decay if recent < 50% of historical
            "min_trades_for_calibration": 20,
        }
        self.config = {**default_config, **(config or {})}

        # Track trades since last review
        self._trades_since_review = 0
        self._trade_history: List[Tuple[datetime, float]] = []

    def post_trade_analysis(
        self,
        trade_record: TradeRecord,
        agent: ResearchAgent,
        predicted_direction: Optional[str] = None,
    ) -> PostTradeReport:
        """Analyze a completed trade and generate learning feedback.

        Parameters
        ----------
        trade_record :
            The completed trade record to analyze.
        agent :
            The agent that executed the trade.
        predicted_direction :
            Optional predicted direction. If None, inferred from action.

        Returns
        -------
        PostTradeReport
            Analysis report with lessons and suggestions.
        """
        # Infer predicted direction from action if not provided
        if not predicted_direction:
            predicted_direction = "UP" if trade_record.action == "BUY" else "DOWN"

        # Determine actual direction
        actual_direction = "UP" if (trade_record.pnl or 0) > 0 else "DOWN"
        prediction_correct = predicted_direction == actual_direction

        # Compute reward
        portfolio_context = {
            "predicted_direction": predicted_direction,
        }
        reward_breakdown = self.reward_engine.compute_reward(
            trade_record, portfolio_context
        )
        reward = reward_breakdown.total_reward

        # Determine what confidence should have been
        confidence_was = trade_record.confidence_at_entry
        confidence_should_be = self._compute_optimal_confidence(
            trade_record, prediction_correct
        )

        # Generate lessons
        lessons = self._generate_lessons(
            trade_record,
            prediction_correct,
            reward_breakdown,
        )

        # Generate parameter suggestions
        parameter_suggestions = self._generate_parameter_suggestions(
            trade_record,
            prediction_correct,
        )

        # Create experience and add to replay buffer
        experience = Experience.create(
            trade_record=trade_record,
            state_before=trade_record.signals_at_entry,
            action_taken={
                "action": trade_record.action,
                "symbol": trade_record.symbol,
                "quantity": trade_record.quantity,
            },
            state_after={
                "pnl": trade_record.pnl,
                "pnl_pct": trade_record.pnl_pct,
                "holding_period": str(trade_record.holding_period) if trade_record.holding_period else None,
            },
            reward=reward,
        )
        self.replay_buffer.add(experience)

        # Track for decay detection
        if trade_record.pnl is not None:
            self._trade_history.append((trade_record.entry_time, trade_record.pnl))
            self._trades_since_review += 1

        return PostTradeReport(
            trade_id=trade_record.trade_id,
            predicted_direction=predicted_direction,
            actual_direction=actual_direction,
            prediction_correct=prediction_correct,
            pnl=trade_record.pnl or 0.0,
            reward=reward,
            confidence_was=confidence_was,
            confidence_should_be=confidence_should_be,
            lessons=lessons,
            parameter_suggestions=parameter_suggestions,
        )

    def _compute_optimal_confidence(
        self,
        trade_record: TradeRecord,
        prediction_correct: bool,
    ) -> float:
        """Compute what confidence should have been in hindsight."""
        # Look at similar trades to determine optimal confidence
        similar_trades = self.trade_memory.get_similar_trades(
            {
                "regime": trade_record.market_regime,
                "volatility": trade_record.volatility_at_entry,
                "signals": trade_record.signals_at_entry,
            },
            top_k=10,
        )

        if similar_trades:
            wins = sum(1 for t in similar_trades if (t.pnl or 0) > 0)
            return wins / len(similar_trades)
        
        # Default: reduce if wrong, maintain if right
        if prediction_correct:
            return min(1.0, trade_record.confidence_at_entry + 0.1)
        else:
            return max(0.1, trade_record.confidence_at_entry - 0.2)

    def _generate_lessons(
        self,
        trade_record: TradeRecord,
        prediction_correct: bool,
        reward_breakdown: RewardBreakdown,
    ) -> List[str]:
        """Generate learned lessons from a trade outcome."""
        lessons = []

        # Direction accuracy lesson
        if not prediction_correct:
            lessons.append(
                f"Incorrect prediction for {trade_record.symbol} in "
                f"{trade_record.market_regime} regime with "
                f"{trade_record.strategy} strategy"
            )
        else:
            lessons.append(
                f"Correct prediction for {trade_record.symbol} using "
                f"{trade_record.strategy} in {trade_record.market_regime} regime"
            )

        # Regime-strategy fit lesson
        if reward_breakdown.regime_component < 0:
            lessons.append(
                f"Strategy '{trade_record.strategy}' may not be optimal for "
                f"'{trade_record.market_regime}' regime"
            )

        # Confidence calibration lesson
        if prediction_correct and trade_record.confidence_at_entry < 0.5:
            lessons.append(
                "Underconfident correct prediction - consider raising "
                "confidence in similar conditions"
            )
        elif not prediction_correct and trade_record.confidence_at_entry > 0.7:
            lessons.append(
                "Overconfident incorrect prediction - consider lowering "
                "confidence in similar conditions"
            )

        # Volatility lesson
        if trade_record.volatility_at_entry > 0.03:
            if abs(trade_record.pnl_pct or 0) > 5:
                lessons.append(
                    "High volatility trade resulted in significant outcome - "
                    "adjust position sizing in similar conditions"
                )

        return lessons

    def _generate_parameter_suggestions(
        self,
        trade_record: TradeRecord,
        prediction_correct: bool,
    ) -> Dict[str, Any]:
        """Generate parameter adjustment suggestions."""
        suggestions: Dict[str, Any] = {}

        # Position sizing suggestion
        if abs(trade_record.pnl_pct or 0) > 5:
            suggestions["position_size_multiplier"] = 0.8 if not prediction_correct else 1.0

        # Confidence threshold suggestion
        if not prediction_correct and trade_record.confidence_at_entry > 0.7:
            suggestions["confidence_threshold"] = min(
                0.9,
                trade_record.confidence_at_entry + 0.1
            )

        # Volatility adjustment
        if trade_record.volatility_at_entry > 0.03:
            suggestions["volatility_adjustment"] = "reduce_exposure"

        return suggestions

    def run_learning_step(
        self,
        agent: ResearchAgent,
        batch_size: int = 32,
        regime: Optional[str] = None,
    ) -> LearningStepResult:
        """Run a single learning step by sampling from replay buffer.

        Parameters
        ----------
        agent :
            Agent to learn.
        batch_size :
            Number of experiences to sample.
        regime :
            Optional regime to filter experiences.

        Returns
        -------
        LearningStepResult
            Result of the learning step.
        """
        # Sample experiences
        experiences = self.replay_buffer.sample(batch_size, regime)

        if not experiences:
            return LearningStepResult(
                experiences_replayed=0,
                lessons=["No experiences available for learning"],
            )

        # Aggregate parameter updates and compute metrics
        all_updates: Dict[str, List[float]] = {}
        total_reward = 0.0
        total_confidence_adj = 0.0
        all_lessons: List[str] = []

        for experience in experiences:
            # Create outcome dict for agent.learn()
            outcome = {
                "trade_id": experience.trade_record.trade_id,
                "pnl": experience.trade_record.pnl,
                "pnl_pct": experience.trade_record.pnl_pct,
                "reward": experience.reward,
                "regime": experience.regime_tag,
                "state_before": experience.state_before,
                "action_taken": experience.action_taken,
                "state_after": experience.state_after,
            }

            # Call agent's learn method
            learning_update = agent.learn(outcome)

            # Aggregate parameter changes
            for param, value in learning_update.parameter_changes.items():
                if param not in all_updates:
                    all_updates[param] = []
                if isinstance(value, (int, float)):
                    all_updates[param].append(value)

            total_confidence_adj += learning_update.confidence_adjustment
            all_lessons.extend(learning_update.lessons)
            total_reward += experience.reward

        # Average the updates
        parameter_updates: Dict[str, Any] = {}
        for param, values in all_updates.items():
            if values:
                parameter_updates[param] = float(np.mean(values)) * self.config["learning_rate"]

        avg_confidence_adj = (
            total_confidence_adj / len(experiences)
            * self.config["confidence_adjustment_rate"]
        )
        avg_reward = total_reward / len(experiences)

        # Remove duplicate lessons
        unique_lessons = list(dict.fromkeys(all_lessons))

        return LearningStepResult(
            parameter_updates=parameter_updates,
            confidence_adjustment=avg_confidence_adj,
            experiences_replayed=len(experiences),
            avg_reward=avg_reward,
            lessons=unique_lessons[:5],  # Top 5 lessons
        )

    def confidence_calibration(
        self, agent: ResearchAgent, agent_id: Optional[str] = None
    ) -> CalibrationResult:
        """Analyze and calibrate agent confidence.

        Compares stated confidence levels to actual accuracy rates.

        Parameters
        ----------
        agent :
            Agent to calibrate.
        agent_id :
            Optional agent ID for filtering trades.

        Returns
        -------
        CalibrationResult
            Calibration analysis result.
        """
        agent_id = agent_id or agent.agent_id

        # Get historical trades for this agent
        trades = self.trade_memory.query(
            agent_id=agent_id,
            limit=1000,
        )

        closed_trades = [t for t in trades if t.is_closed]

        if len(closed_trades) < self.config["min_trades_for_calibration"]:
            return CalibrationResult(
                current_calibration_error=0.0,
                adjustment=0.0,
                overconfident=False,
                details={"message": "Insufficient trades for calibration"},
            )

        # Compute calibration metrics
        confidence_buckets: Dict[str, List[Tuple[float, bool]]] = {
            "low": [],      # 0-0.33
            "medium": [],   # 0.33-0.66
            "high": [],     # 0.66-1.0
        }

        for trade in closed_trades:
            # Determine if trade was correct
            pnl = trade.pnl or 0
            if trade.action == "BUY":
                correct = pnl > 0
            else:
                correct = pnl > 0  # SELL profits from price drop

            confidence = trade.confidence_at_entry
            if confidence < 0.33:
                bucket = "low"
            elif confidence < 0.66:
                bucket = "medium"
            else:
                bucket = "high"

            confidence_buckets[bucket].append((confidence, correct))

        # Compute calibration error
        calibration_errors = []
        details: Dict[str, Any] = {"buckets": {}}

        for bucket_name, bucket_data in confidence_buckets.items():
            if bucket_data:
                avg_confidence = np.mean([c for c, _ in bucket_data])
                accuracy = np.mean([1.0 if correct else 0.0 for _, correct in bucket_data])
                error = avg_confidence - accuracy
                calibration_errors.append(error)
                details["buckets"][bucket_name] = {
                    "avg_confidence": avg_confidence,
                    "accuracy": accuracy,
                    "error": error,
                    "count": len(bucket_data),
                }

        # Overall calibration error
        overall_confidence = np.mean([t.confidence_at_entry for t in closed_trades])
        overall_accuracy = np.mean([
            1.0 if (t.pnl or 0) > 0 else 0.0 for t in closed_trades
        ])
        calibration_error = overall_confidence - overall_accuracy

        # Determine adjustment
        overconfident = calibration_error > 0.1
        adjustment = -calibration_error * self.config["confidence_adjustment_rate"]

        details["overall_confidence"] = overall_confidence
        details["overall_accuracy"] = overall_accuracy
        details["trades_analyzed"] = len(closed_trades)

        return CalibrationResult(
            current_calibration_error=calibration_error,
            adjustment=adjustment,
            overconfident=overconfident,
            details=details,
        )

    def detect_performance_decay(
        self, agent_id: str
    ) -> DecayReport:
        """Detect if agent performance is decaying.

        Parameters
        ----------
        agent_id :
            Agent identifier.

        Returns
        -------
        DecayReport
            Decay analysis report.
        """
        window = self.config["decay_detection_window"]

        # Get recent trades
        recent_trades = self.trade_memory.query(
            agent_id=agent_id,
            limit=window,
        )
        recent_closed = [t for t in recent_trades if t.is_closed]

        # Get historical trades (older than window)
        if len(self._trade_history) < window:
            return DecayReport(
                decaying=False,
                window_analyzed=len(self._trade_history),
                recommendation="Insufficient history for decay detection",
            )

        # Split into recent and historical
        half_window = len(recent_closed) // 2

        if half_window < 5:
            return DecayReport(
                decaying=False,
                window_analyzed=len(recent_closed),
                recommendation="Insufficient recent trades for comparison",
            )

        # Compute Sharpe for recent vs historical
        recent_pnls = [t.pnl for t in recent_closed[:half_window] if t.pnl is not None]
        historical_pnls = [t.pnl for t in recent_closed[half_window:] if t.pnl is not None]

        if not recent_pnls or not historical_pnls:
            return DecayReport(
                decaying=False,
                window_analyzed=len(recent_closed),
                recommendation="Insufficient PnL data for comparison",
            )

        recent_sharpe = self._compute_sharpe(recent_pnls)
        historical_sharpe = self._compute_sharpe(historical_pnls)

        # Detect decay
        decay_threshold = self.config["decay_threshold"]
        decaying = (
            recent_sharpe < historical_sharpe * decay_threshold
            and recent_sharpe < 0
        )

        # Compute decay rate
        if historical_sharpe != 0:
            decay_rate = (historical_sharpe - recent_sharpe) / abs(historical_sharpe)
        else:
            decay_rate = 0.0

        # Generate recommendation
        if decaying:
            recommendation = (
                f"Performance decay detected. Recent Sharpe ({recent_sharpe:.2f}) is "
                f"{decay_rate*100:.1f}% below historical ({historical_sharpe:.2f}). "
                "Consider strategy review, parameter retraining, or regime adjustment."
            )
        else:
            recommendation = "No significant performance decay detected."

        return DecayReport(
            decaying=decaying,
            decay_rate=decay_rate,
            window_analyzed=len(recent_closed),
            recent_sharpe=recent_sharpe,
            historical_sharpe=historical_sharpe,
            recommendation=recommendation,
        )

    def _compute_sharpe(self, pnls: List[float]) -> float:
        """Compute approximate Sharpe ratio from PnL series."""
        if len(pnls) < 2:
            return 0.0

        pnls_array = np.array(pnls)
        mean_pnl = np.mean(pnls_array)
        std_pnl = np.std(pnls_array)

        if std_pnl == 0:
            return 0.0

        # Annualize assuming daily returns
        return float(mean_pnl / std_pnl * np.sqrt(252))

    def periodic_review(
        self, agent: ResearchAgent
    ) -> ReviewReport:
        """Perform comprehensive periodic review.

        Parameters
        ----------
        agent :
            Agent to review.

        Returns
        -------
        ReviewReport
            Comprehensive review report.
        """
        agent_id = agent.agent_id

        # Get trades for review
        trades = self.trade_memory.query(
            agent_id=agent_id,
            limit=self.config["review_period"],
        )
        closed_trades = [t for t in trades if t.is_closed]

        # Compute overall accuracy
        if closed_trades:
            wins = sum(1 for t in closed_trades if (t.pnl or 0) > 0)
            overall_accuracy = wins / len(closed_trades)
        else:
            overall_accuracy = 0.0

        # Get strategy performance
        strategy_perf = self.trade_memory.get_performance_by_strategy()

        # Filter to strategies used by this agent
        agent_strategies = {t.strategy for t in closed_trades}
        strategy_performance = {
            s: p for s, p in strategy_perf.items()
            if s in agent_strategies
        }

        # Get regime performance
        regime_performance = self.trade_memory.get_performance_by_regime()

        # Detect decay
        decay_report = self.detect_performance_decay(agent_id)

        # Generate recommendations
        recommendations = []

        # Strategy recommendations
        if strategy_performance:
            best_strategy = max(
                strategy_performance.items(),
                key=lambda x: x[1].win_rate,
            )
            worst_strategy = min(
                strategy_performance.items(),
                key=lambda x: x[1].win_rate,
            )

            if worst_strategy[1].win_rate < 0.4:
                recommendations.append(
                    f"Consider deprecating or retraining '{worst_strategy[0]}' "
                    f"strategy (win rate: {worst_strategy[1].win_rate:.1%})"
                )

            recommendations.append(
                f"Best performing strategy: '{best_strategy[0]}' "
                f"(win rate: {best_strategy[1].win_rate:.1%})"
            )

        # Calibration recommendations
        calibration = self.confidence_calibration(agent)
        if calibration.overconfident:
            recommendations.append(
                f"Agent is overconfident by {calibration.current_calibration_error:.1%}. "
                "Confidence calibration recommended."
            )

        # Decay recommendations
        if decay_report.decaying:
            recommendations.append(decay_report.recommendation)

        # Reset trade counter
        self._trades_since_review = 0

        return ReviewReport(
            trades_reviewed=len(closed_trades),
            overall_accuracy=overall_accuracy,
            strategy_performance=strategy_performance,
            regime_performance={
                r: {
                    "total_trades": p.total_trades,
                    "win_rate": p.win_rate,
                    "avg_pnl": p.avg_pnl,
                }
                for r, p in regime_performance.items()
            },
            decay_detected=decay_report.decaying,
            recommendations=recommendations,
        )

    def should_review(self) -> bool:
        """Check if it's time for a periodic review.

        Returns
        -------
        bool
            True if review should be performed.
        """
        return self._trades_since_review >= self.config["review_period"]

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning system state.

        Returns
        -------
        Dict[str, Any]
            Learning system summary.
        """
        buffer_stats = self.replay_buffer.get_statistics()
        reward_stats = self.reward_engine.get_reward_statistics([])

        return {
            "trades_since_review": self._trades_since_review,
            "total_trades_tracked": len(self._trade_history),
            "buffer_statistics": buffer_stats,
            "config": self.config,
        }
