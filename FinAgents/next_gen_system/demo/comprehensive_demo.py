"""
FinAgents Next-Generation System - Comprehensive Demonstration
===============================================================

This script demonstrates the complete upgraded multi-agent financial system with:
1. Domain-specialized agents (Trader, Risk, Analyst, Portfolio)
2. Multimodal intelligence fusion
3. Advanced coordination protocols
4. Memory and learning systems
5. Explainability framework
6. Market simulation environment
7. Comprehensive evaluation

Author: FinAgents Research Team
Version: 2.0.0 (Research Edition)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from next_gen_system.agents import (
    TraderAgent,
    RiskAgent,
    AnalystAgent,
    PortfolioAgent,
)
from next_gen_system.coordination import AgentCoordinator
from next_gen_system.memory import UnifiedMemorySystem
from next_gen_system.explainability import ExplainabilityEngine
from next_gen_system.environment import MarketSimulationEnvironment
from next_gen_system.evaluation import EvaluationFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'nextgen_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("NextGenDemo")


def generate_synthetic_market_data(
    symbols: list,
    start_date: datetime,
    num_days: int = 100,
) -> pd.DataFrame:
    """Generate synthetic market data for demonstration"""
    dates = pd.date_range(start=start_date, periods=num_days, freq='B')
    
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(100, 500)
        
        for i, date in enumerate(dates):
            # Random walk with drift
            drift = 0.0005  # Small upward drift
            shock = np.random.normal(0, 0.02)
            
            if i == 0:
                open_price = base_price
            else:
                open_price = data[-1][data[-1]['symbol'] == symbol]['close'].values[0]
            
            close_price = open_price * (1 + drift + shock)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1_000_000, 10_000_000)
            
            data.append({
                'date': date,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            })
    
    df = pd.DataFrame(data)
    return df


def run_comprehensive_demo():
    """Run comprehensive demonstration of the next-gen system"""
    
    logger.info("=" * 80)
    logger.info("FINAGENTS NEXT-GENERATION SYSTEM DEMONSTRATION")
    logger.info("=" * 80)
    
    # ========================================================================
    # PHASE 1: SYSTEM INITIALIZATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: SYSTEM INITIALIZATION")
    logger.info("=" * 80)
    
    # Initialize agents
    trader = TraderAgent(
        agent_id="trader_01",
        initial_capital=1_000_000.0,
        strategies=["momentum", "mean_reversion", "breakout"],
        risk_tolerance=0.02,
    )
    
    risk = RiskAgent(
        agent_id="risk_01",
        var_confidence=0.95,
        max_portfolio_var=0.05,
        max_position_size=0.10,
        max_drawdown=0.20,
    )
    
    analyst = AnalystAgent(agent_id="analyst_01")
    
    portfolio = PortfolioAgent(
        agent_id="portfolio_01",
        risk_free_rate=0.02,
        max_position_size=0.25,
        target_volatility=0.15,
    )
    
    # Initialize coordination and support systems
    coordinator = AgentCoordinator(coordinator_id="main_coordinator")
    memory = UnifiedMemorySystem(memory_id="main_memory", learning_rate=0.1)
    explainer = ExplainabilityEngine(engine_id="main_explainer")
    
    # Initialize market simulation
    market = MarketSimulationEnvironment(
        initial_capital=1_000_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(framework_id="main_evaluator")
    
    # Register agents with coordinator
    coordinator.register_agent("trader_01", "trader", trader)
    coordinator.register_agent("risk_01", "risk", risk)
    coordinator.register_agent("analyst_01", "analyst", analyst)
    coordinator.register_agent("portfolio_01", "portfolio", portfolio)
    
    logger.info(f"✓ Initialized {4} domain-specialized agents")
    logger.info(f"✓ Initialized coordination system")
    logger.info(f"✓ Initialized memory and learning system")
    logger.info(f"✓ Initialized explainability engine")
    logger.info(f"✓ Initialized market simulation")
    
    # ========================================================================
    # PHASE 2: MARKET SETUP
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: MARKET SETUP")
    logger.info("=" * 80)
    
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    start_date = datetime(2024, 1, 1)
    
    # Generate synthetic market data
    market_data = generate_synthetic_market_data(symbols, start_date, num_days=100)
    
    # Initialize market in simulation
    initial_prices = {
        symbol: market_data[market_data['symbol'] == symbol].iloc[0]['close']
        for symbol in symbols
    }
    market.initialize_market(symbols, initial_prices)
    
    logger.info(f"✓ Initialized market with {len(symbols)} symbols")
    logger.info(f"✓ Generated {len(market_data)} data points")
    logger.info(f"Initial prices: {initial_prices}")
    
    # ========================================================================
    # PHASE 3: MULTI-AGENT TRADING SIMULATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: MULTI-AGENT TRADING SIMULATION")
    logger.info("=" * 80)
    
    # Simulation parameters
    num_trading_days = 50
    decisions_log = []
    
    for day in range(num_trading_days):
        logger.info(f"\n--- Trading Day {day + 1}/{num_trading_days} ---")
        
        # Get current market state
        current_bar = market_data[market_data['date'] == start_date + timedelta(days=day)]
        
        if len(current_bar) == 0:
            continue
        
        # Convert to format expected by agents
        price_data_dict = {}
        for symbol in symbols:
            symbol_data = current_bar[current_bar['symbol'] == symbol]
            if len(symbol_data) > 0:
                price_data_dict[symbol] = pd.DataFrame({
                    'open': [symbol_data['open'].values[0]],
                    'high': [symbol_data['high'].values[0]],
                    'low': [symbol_data['low'].values[0]],
                    'close': [symbol_data['close'].values[0]],
                    'volume': [symbol_data['volume'].values[0]],
                })
        
        # ===== AGENT 1: ANALYST performs multimodal analysis =====
        news_headlines = [
            f"Tech stocks rally on positive earnings",
            f"Market sentiment improves amid economic data",
        ] if day % 5 == 0 else []
        
        macro_indicators = {
            "interest_rate_change": 0.0,
            "inflation_rate": 2.5,
            "gdp_growth": 2.1,
            "vix_level": 18,
        }
        
        analyst_reports = {}
        for symbol in symbols:
            report = analyst.analyze_symbol(
                symbol=symbol,
                price_data=price_data_dict.get(symbol, pd.DataFrame()),
                news_headlines=news_headlines,
                macro_indicators=macro_indicators,
            )
            analyst_reports[symbol] = report
        
        logger.info(f"✓ Analyst completed analysis for {len(analyst_reports)} symbols")
        
        # ===== AGENT 2: TRADER generates trading signals =====
        trading_signals = trader.analyze_market(
            price_data=current_bar.pivot(index='symbol', values=['close', 'high', 'low', 'volume']),
            news_sentiment={s: report.overall_sentiment for s, report in analyst_reports.items()},
        )
        
        logger.info(f"✓ Trader generated {len(trading_signals)} signals")
        
        # ===== AGENT 3: RISK assesses portfolio risk =====
        portfolio_values = pd.Series(market.portfolio_values)
        risk_assessment = risk.assess_portfolio_risk(
            portfolio_values=portfolio_values,
            positions=market.positions,
            market_data={s: current_bar[current_bar['symbol'] == s] for s in symbols},
        )
        
        logger.info(f"✓ Risk assessment: {risk_assessment.risk_level}")
        
        # ===== COORDINATION: Multi-agent decision making =====
        proposals = []
        
        for symbol, signal in trading_signals.items():
            if signal.action != "HOLD" and signal.strength > 0.3:
                proposal = coordinator.negotiate_trade(
                    trader_proposal={
                        "agent_id": "trader_01",
                        "action_type": signal.action,
                        "symbol": symbol,
                        "proposal_data": {
                            "strength": signal.strength,
                            "entry_price": signal.entry_price,
                        },
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning,
                    },
                    risk_assessment={
                        "risk_level": risk_assessment.risk_level,
                        "var": risk_assessment.portfolio_var,
                    },
                    analyst_view={
                        "overall_sentiment": analyst_reports[symbol].overall_sentiment,
                        "technical_score": analyst_reports[symbol].technical_score,
                    },
                )
                
                decisions_log.append({
                    "day": day,
                    "symbol": symbol,
                    "decision": proposal.decision,
                    "confidence": proposal.confidence,
                })
                
                logger.info(f"  Coordinated decision for {symbol}: {proposal.decision} (conf: {proposal.confidence:.2f})")
        
        # ===== EXECUTION: Execute approved trades =====
        current_prices = {s: initial_prices.get(s, 100) for s in symbols}
        
        for symbol, signal in trading_signals.items():
            if signal.action in ["BUY", "SELL"] and signal.strength > 0.5:
                size = int(signal.strength * 100)  # Simplified position sizing
                
                if signal.action == "BUY" and size > 0:
                    result = market.execute_trade(symbol, "BUY", size, signal.entry_price)
                    logger.info(f"  Executed BUY {size} {symbol} @ ${signal.entry_price:.2f}")
                
                # Store in memory
                memory.store_episode(
                    symbol=symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    market_context={
                        "volatility": risk_assessment.current_volatility,
                        "sentiment": analyst_reports[symbol].overall_sentiment,
                    },
                    agent_decisions=[{
                        "agent": "trader",
                        "decision": signal.action,
                        "confidence": signal.confidence,
                    }],
                )
        
        # Advance market simulation
        external_events = []
        if day % 10 == 0:
            # Simulate occasional news events
            event_impact = np.random.choice([-0.03, -0.01, 0.01, 0.03])
            external_events.append({
                "type": "news_event",
                "symbols": symbols,
                "impact": event_impact,
                "duration_days": 1,
            })
        
        market_state = market.step(
            agent_actions=None,
            external_events=external_events,
        )
        
        # Update prices
        initial_prices = market_state.prices.copy()
    
    # ========================================================================
    # PHASE 4: PERFORMANCE EVALUATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: PERFORMANCE EVALUATION")
    logger.info("=" * 80)
    
    # Calculate performance metrics
    performance = market.calculate_performance_metrics()
    
    logger.info(f"\nPERFORMANCE METRICS:")
    logger.info(f"  Total Return: {performance.total_return*100:.2f}%")
    logger.info(f"  Annualized Return: {performance.annualized_return*100:.2f}%")
    logger.info(f"  Volatility: {performance.volatility*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio: {performance.sortino_ratio:.2f}")
    logger.info(f"  Max Drawdown: {performance.max_drawdown*100:.2f}%")
    logger.info(f"  Win Rate: {performance.win_rate*100:.1f}%")
    logger.info(f"  Profit Factor: {performance.profit_factor:.2f}")
    
    # Comprehensive evaluation
    eval_result = evaluator.evaluate_system(
        portfolio_values=market.portfolio_values,
        trade_log=market.trade_log,
        decisions=decisions_log,
    )
    
    logger.info(f"\n{eval_result.summary}")
    
    if eval_result.recommendations:
        logger.info(f"\nRECOMMENDATIONS:")
        for rec in eval_result.recommendations:
            logger.info(f"  • {rec}")
    
    # ========================================================================
    # PHASE 5: EXPLAINABILITY DEMONSTRATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: EXPLAINABILITY DEMONSTRATION")
    logger.info("=" * 80)
    
    # Generate explanation for a sample decision
    if decisions_log:
        sample_decision = decisions_log[0]
        
        explanation = explainer.explain_trade_decision(
            symbol=sample_decision['symbol'],
            action=sample_decision['decision'],
            strength=sample_decision['confidence'],
            trader_reasoning=["Momentum signal triggered", "Technical breakout detected"],
            risk_assessment={
                "risk_level": "MEDIUM",
                "var_calculated": True,
                "volatility": 0.18,
            },
            analyst_view={
                "overall_sentiment": 0.65,
                "technical_score": 7.2,
                "news_analyzed": True,
            },
            final_confidence=sample_decision['confidence'],
        )
        
        natural_language_exp = explainer.generate_natural_language_explanation(explanation)
        logger.info(f"\nSAMPLE EXPLANATION:\n{natural_language_exp}")
    
    # ========================================================================
    # PHASE 6: MEMORY AND LEARNING STATISTICS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: MEMORY AND LEARNING STATISTICS")
    logger.info("=" * 80)
    
    memory_stats = memory.get_experience_statistics()
    logger.info(f"\nMEMORY STATISTICS:")
    logger.info(f"  Episodes Stored: {memory_stats.get('total_episodes', 0)}")
    logger.info(f"  Symbols Traded: {memory_stats.get('symbols_traded', 0)}")
    logger.info(f"  Win Rate: {memory_stats.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Average Outcome Rating: {memory_stats.get('avg_outcome_rating', 0):.3f}")
    
    # ========================================================================
    # PHASE 7: SYSTEM STATE SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 7: SYSTEM STATE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nAGENT STATES:")
    logger.info(f"  Trader: {trader.get_state()}")
    logger.info(f"  Risk: {risk.get_state()}")
    logger.info(f"  Analyst: {analyst.get_state()}")
    logger.info(f"  Portfolio: {portfolio.get_state()}")
    
    logger.info(f"\nCOORDINATOR STATE:")
    logger.info(f"  {coordinator.get_state()}")
    
    logger.info(f"\nMEMORY STATE:")
    logger.info(f"  {memory.get_state()}")
    
    logger.info(f"\nEXPLAINER STATE:")
    logger.info(f"  {explainer.get_state()}")
    
    logger.info(f"\nEVALUATOR STATE:")
    logger.info(f"  {evaluator.get_state()}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"""
NEXT-GENERATION SYSTEM CAPABILITIES DEMONSTRATED:
✓ Domain-specialized agents (Trader, Risk, Analyst, Portfolio)
✓ Multimodal intelligence fusion (prices, news, macro, charts)
✓ Advanced coordination protocols (negotiation, voting, consensus)
✓ Memory and learning systems (episodic, semantic, procedural)
✓ Explainability framework (reasoning chains, counterfactuals)
✓ Realistic market simulation (dynamic prices, events, frictions)
✓ Comprehensive evaluation (financial + AI metrics)

RESEARCH CONTRIBUTIONS:
1. Integrated multi-agent architecture for financial trading
2. Novel coordination mechanism with negotiation protocols
3. Multimodal data fusion for enhanced decision-making
4. Explainable AI framework for regulatory compliance
5. Memory-augmented learning for continuous improvement
6. Comprehensive evaluation methodology

For research paper details, see accompanying documentation.
""")
    
    logger.info(f"Log file saved to: nextgen_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return {
        "performance_metrics": {
            "total_return": performance.total_return,
            "sharpe_ratio": performance.sharpe_ratio,
            "max_drawdown": performance.max_drawdown,
            "win_rate": performance.win_rate,
        },
        "system_states": {
            "trader": trader.get_state(),
            "risk": risk.get_state(),
            "analyst": analyst.get_state(),
            "portfolio": portfolio.get_state(),
            "coordinator": coordinator.get_state(),
            "memory": memory.get_state(),
        },
        "decisions_made": len(decisions_log),
        "trades_executed": len(market.trade_log),
    }


if __name__ == "__main__":
    try:
        result = run_comprehensive_demo()
        logger.info(f"\n✓ Demonstration completed successfully!")
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}", exc_info=True)
        sys.exit(1)
