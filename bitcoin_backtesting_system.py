"""
Bitcoin Prediction Backtesting & Validation System
================================================
Professional-grade backtesting system for banking risk management.
Measures model accuracy and performance across different market conditions.

Features:
- Historical performance validation
- Risk-adjusted return calculations  
- Market regime analysis (bull/bear/sideways)
- Drawdown analysis and risk metrics
- Performance comparison vs benchmarks
- Detailed improvement recommendations
- Banking compliance reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yfinance as yf

class BitcoinBacktestingSystem:
    """
    Comprehensive backtesting system for Bitcoin prediction models.
    Designed for banking risk management and regulatory compliance.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.max_position_size = 0.03  # 3% maximum position
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.min_confidence = 0.65     # 65% minimum confidence
        
        # Risk management parameters
        self.stop_loss = 0.05          # 5% stop loss
        self.take_profit = 0.10        # 10% take profit
        self.max_drawdown_limit = 0.15 # 15% maximum drawdown before halt
        
        print(f"🏦 Backtesting System Initialized")
        print(f"   Initial Capital: ${initial_capital:,.0f}")
        print(f"   Max Position Size: {self.max_position_size:.1%}")
        print(f"   Transaction Cost: {self.transaction_cost:.1%}")
        print(f"   Stop Loss: {self.stop_loss:.1%}")
        print(f"   Take Profit: {self.take_profit:.1%}")
    
    def prepare_backtest_data(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame, 
                            predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive dataset for backtesting.
        
        Args:
            price_df: Historical Bitcoin price data
            sentiment_df: Wikipedia sentiment scores
            predictions_df: Model predictions with confidence and risk scores
        """
        print("📊 Preparing backtesting dataset...")
        
        # Ensure date columns are datetime
        for df in [price_df, sentiment_df, predictions_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Merge all data
        backtest_df = price_df[['date', 'price_usd', 'volume_24h']].copy()
        
        # Add sentiment data
        backtest_df = pd.merge(backtest_df, 
                              sentiment_df[['date', 'final_compound', 'daily_sentiment']], 
                              on='date', how='left')
        
        # Add predictions
        backtest_df = pd.merge(backtest_df, 
                              predictions_df[['date', 'direction', 'confidence', 'risk_level']], 
                              on='date', how='left')
        
        # Calculate actual returns
        backtest_df['actual_return'] = backtest_df['price_usd'].pct_change()
        backtest_df['actual_direction'] = np.where(
            backtest_df['actual_return'] > 0.02, 'UP',
            np.where(backtest_df['actual_return'] < -0.02, 'DOWN', 'NEUTRAL')
        )
        
        # Market regime classification
        backtest_df = self._classify_market_regimes(backtest_df)
        
        # Remove NaN values
        backtest_df = backtest_df.dropna()
        
        print(f"✅ Backtest dataset prepared: {len(backtest_df)} trading days")
        print(f"   Date range: {backtest_df['date'].min().date()} to {backtest_df['date'].max().date()}")
        
        return backtest_df
    
    def _classify_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market conditions into Bull, Bear, or Sideways regimes.
        """
        df = df.copy()
        
        # Calculate rolling performance metrics
        df['price_sma_30'] = df['price_usd'].rolling(window=30).mean()
        df['price_sma_90'] = df['price_usd'].rolling(window=90).mean()
        df['volatility_30d'] = df['actual_return'].rolling(window=30).std() * np.sqrt(365)
        
        # Market regime classification
        conditions = [
            (df['price_usd'] > df['price_sma_30']) & (df['price_sma_30'] > df['price_sma_90']) & (df['volatility_30d'] < 1.5),
            (df['price_usd'] < df['price_sma_30']) & (df['price_sma_30'] < df['price_sma_90']) & (df['volatility_30d'] < 1.5),
        ]
        choices = ['Bull Market', 'Bear Market']
        df['market_regime'] = np.select(conditions, choices, default='Sideways Market')
        
        return df
    
    def run_classification_backtest(self, backtest_df: pd.DataFrame) -> Dict:
        """
        Run comprehensive classification accuracy backtest.
        """
        print("🎯 Running classification accuracy backtest...")
        
        # Filter for valid predictions
        valid_predictions = backtest_df.dropna(subset=['direction', 'actual_direction']).copy()
        
        # Overall accuracy metrics
        overall_accuracy = accuracy_score(valid_predictions['actual_direction'], 
                                        valid_predictions['direction'])
        
        # High confidence accuracy
        high_conf_mask = valid_predictions['confidence'] >= self.min_confidence
        high_conf_data = valid_predictions[high_conf_mask]
        high_conf_accuracy = accuracy_score(high_conf_data['actual_direction'], 
                                          high_conf_data['direction']) if len(high_conf_data) > 0 else 0
        
        # Low risk accuracy
        low_risk_mask = valid_predictions['risk_level'] <= 3
        low_risk_data = valid_predictions[low_risk_mask]
        low_risk_accuracy = accuracy_score(low_risk_data['actual_direction'], 
                                         low_risk_data['direction']) if len(low_risk_data) > 0 else 0
        
        # Conservative strategy (high confidence + low risk)
        conservative_mask = (valid_predictions['confidence'] >= self.min_confidence) & \
                           (valid_predictions['risk_level'] <= 3)
        conservative_data = valid_predictions[conservative_mask]
        conservative_accuracy = accuracy_score(conservative_data['actual_direction'], 
                                             conservative_data['direction']) if len(conservative_data) > 0 else 0
        
        # Accuracy by market regime
        regime_accuracy = {}
        for regime in ['Bull Market', 'Bear Market', 'Sideways Market']:
            regime_data = valid_predictions[valid_predictions['market_regime'] == regime]
            if len(regime_data) > 0:
                regime_accuracy[regime] = accuracy_score(regime_data['actual_direction'], 
                                                       regime_data['direction'])
            else:
                regime_accuracy[regime] = 0
        
        # Confusion matrix for detailed analysis
        cm = confusion_matrix(valid_predictions['actual_direction'], 
                            valid_predictions['direction'], 
                            labels=['DOWN', 'NEUTRAL', 'UP'])
        
        # Classification report
        class_report = classification_report(valid_predictions['actual_direction'], 
                                           valid_predictions['direction'], 
                                           output_dict=True)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'low_risk_accuracy': low_risk_accuracy,
            'conservative_accuracy': conservative_accuracy,
            'regime_accuracy': regime_accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'total_predictions': len(valid_predictions),
            'high_conf_predictions': len(high_conf_data),
            'conservative_predictions': len(conservative_data),
            'data_coverage': len(valid_predictions) / len(backtest_df)
        }
        
        print(f"📈 Classification Results:")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        print(f"   High Confidence Accuracy: {high_conf_accuracy:.1%} ({len(high_conf_data)} predictions)")
        print(f"   Conservative Strategy Accuracy: {conservative_accuracy:.1%} ({len(conservative_data)} predictions)")
        print(f"   Bull Market Accuracy: {regime_accuracy.get('Bull Market', 0):.1%}")
        print(f"   Bear Market Accuracy: {regime_accuracy.get('Bear Market', 0):.1%}")
        print(f"   Sideways Market Accuracy: {regime_accuracy.get('Sideways Market', 0):.1%}")
        
        return results
    
    def run_trading_backtest(self, backtest_df: pd.DataFrame) -> Dict:
        """
        Run comprehensive trading performance backtest with risk management.
        """
        print("💰 Running trading performance backtest...")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'btc_holdings': 0.0,
            'total_value': self.initial_capital,
            'position_size': 0.0,
            'entry_price': 0.0,
            'stop_loss_price': 0.0,
            'take_profit_price': 0.0
        }
        
        # Trading history
        trades = []
        portfolio_history = []
        
        for idx, row in backtest_df.iterrows():
            date = row['date']
            price = row['price_usd']
            direction = row['direction']
            confidence = row['confidence']
            risk_level = row['risk_level']
            
            # Calculate current portfolio value
            portfolio['total_value'] = portfolio['cash'] + (portfolio['btc_holdings'] * price)
            
            # Risk management: check stop loss and take profit
            if portfolio['btc_holdings'] > 0:
                # Long position checks
                if price <= portfolio['stop_loss_price']:
                    # Stop loss triggered
                    self._execute_trade(portfolio, 'SELL', price, date, 'STOP_LOSS', trades)
                elif price >= portfolio['take_profit_price']:
                    # Take profit triggered
                    self._execute_trade(portfolio, 'SELL', price, date, 'TAKE_PROFIT', trades)
            elif portfolio['btc_holdings'] < 0:
                # Short position checks (if implemented)
                pass
            
            # New trading signals
            if pd.notna(direction) and pd.notna(confidence):
                # Check if we should take action based on conservative criteria
                if (confidence >= self.min_confidence and 
                    risk_level <= 3 and 
                    portfolio['btc_holdings'] == 0):  # Only enter if no current position
                    
                    if direction == 'UP':
                        # Enter long position
                        position_value = portfolio['total_value'] * self.max_position_size
                        if position_value <= portfolio['cash']:
                            self._execute_trade(portfolio, 'BUY', price, date, 'SIGNAL', trades)
                            
                            # Set stop loss and take profit
                            portfolio['stop_loss_price'] = price * (1 - self.stop_loss)
                            portfolio['take_profit_price'] = price * (1 + self.take_profit)
                    
                    elif direction == 'DOWN' and portfolio['btc_holdings'] > 0:
                        # Exit long position
                        self._execute_trade(portfolio, 'SELL', price, date, 'SIGNAL', trades)
            
            # Record portfolio state
            portfolio_history.append({
                'date': date,
                'price': price,
                'cash': portfolio['cash'],
                'btc_holdings': portfolio['btc_holdings'],
                'total_value': portfolio['total_value'],
                'market_regime': row.get('market_regime', 'Unknown')
            })
            
            # Emergency stop: halt trading if drawdown exceeds limit
            max_portfolio_value = max([p['total_value'] for p in portfolio_history])
            current_drawdown = (max_portfolio_value - portfolio['total_value']) / max_portfolio_value
            if current_drawdown > self.max_drawdown_limit:
                print(f"⚠️ Emergency stop triggered: {current_drawdown:.1%} drawdown on {date.date()}")
                break
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_history)
        trades_df = pd.DataFrame(trades)
        
        performance_metrics = self._calculate_performance_metrics(portfolio_df, trades_df)
        
        return {
            'portfolio_history': portfolio_df,
            'trades': trades_df,
            'performance_metrics': performance_metrics,
            'final_portfolio': portfolio
        }
    
    def _execute_trade(self, portfolio: Dict, action: str, price: float, date: datetime, 
                      reason: str, trades: List[Dict]):
        """Execute a trade and update portfolio."""
        
        if action == 'BUY':
            # Calculate position size
            position_value = portfolio['total_value'] * self.max_position_size
            transaction_cost = position_value * self.transaction_cost
            net_position_value = position_value - transaction_cost
            
            btc_amount = net_position_value / price
            
            portfolio['cash'] -= position_value
            portfolio['btc_holdings'] += btc_amount
            portfolio['entry_price'] = price
            
            trades.append({
                'date': date,
                'action': action,
                'price': price,
                'amount': btc_amount,
                'value': position_value,
                'cost': transaction_cost,
                'reason': reason,
                'portfolio_value': portfolio['cash'] + (portfolio['btc_holdings'] * price)
            })
            
        elif action == 'SELL' and portfolio['btc_holdings'] > 0:
            # Sell all BTC holdings
            sale_value = portfolio['btc_holdings'] * price
            transaction_cost = sale_value * self.transaction_cost
            net_sale_value = sale_value - transaction_cost
            
            portfolio['cash'] += net_sale_value
            btc_amount = portfolio['btc_holdings']
            portfolio['btc_holdings'] = 0
            portfolio['entry_price'] = 0
            portfolio['stop_loss_price'] = 0
            portfolio['take_profit_price'] = 0
            
            trades.append({
                'date': date,
                'action': action,
                'price': price,
                'amount': btc_amount,
                'value': sale_value,
                'cost': transaction_cost,
                'reason': reason,
                'portfolio_value': portfolio['cash']
            })
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if portfolio_df.empty:
            return {"error": "No portfolio data available"}
        
        # Basic performance
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = (final_value / initial_value) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        volatility = portfolio_df['daily_return'].std() * np.sqrt(365)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate and average trade
        if not trades_df.empty:
            # Pair up buy/sell trades
            profitable_trades = 0
            total_completed_trades = 0
            total_profit = 0
            
            for i in range(0, len(trades_df) - 1, 2):
                if (i + 1 < len(trades_df) and 
                    trades_df.iloc[i]['action'] == 'BUY' and 
                    trades_df.iloc[i + 1]['action'] == 'SELL'):
                    
                    buy_price = trades_df.iloc[i]['price']
                    sell_price = trades_df.iloc[i + 1]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    
                    total_completed_trades += 1
                    total_profit += trade_return
                    
                    if trade_return > 0:
                        profitable_trades += 1
            
            win_rate = profitable_trades / total_completed_trades if total_completed_trades > 0 else 0
            avg_trade_return = total_profit / total_completed_trades if total_completed_trades > 0 else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            total_completed_trades = 0
        
        # Buy and hold comparison
        btc_initial_price = portfolio_df['price'].iloc[0]
        btc_final_price = portfolio_df['price'].iloc[-1]
        buy_hold_return = (btc_final_price - btc_initial_price) / btc_initial_price
        
        # Performance by market regime
        regime_performance = {}
        for regime in portfolio_df['market_regime'].unique():
            regime_data = portfolio_df[portfolio_df['market_regime'] == regime]
            if len(regime_data) > 1:
                regime_return = (regime_data['total_value'].iloc[-1] - regime_data['total_value'].iloc[0]) / regime_data['total_value'].iloc[0]
                regime_performance[regime] = regime_return
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_completed_trades,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'regime_performance': regime_performance,
            'final_portfolio_value': final_value,
            'trading_days': len(portfolio_df)
        }
    
    def run_comprehensive_analysis(self, backtest_df: pd.DataFrame) -> Dict:
        """
        Run complete backtesting analysis combining classification and trading performance.
        """
        print("\n" + "="*70)
        print("🔬 COMPREHENSIVE BACKTESTING ANALYSIS")
        print("="*70)
        
        # Run classification backtest
        classification_results = self.run_classification_backtest(backtest_df)
        
        print("\n" + "-"*50)
        
        # Run trading backtest
        trading_results = self.run_trading_backtest(backtest_df)
        
        # Combine results
        comprehensive_results = {
            'classification_performance': classification_results,
            'trading_performance': trading_results,
            'backtest_period': {
                'start_date': backtest_df['date'].min(),
                'end_date': backtest_df['date'].max(),
                'total_days': len(backtest_df)
            }
        }
        
        # Generate improvement recommendations
        improvements = self._generate_improvement_recommendations(comprehensive_results)
        comprehensive_results['improvement_recommendations'] = improvements
        
        return comprehensive_results
    
    def _generate_improvement_recommendations(self, results: Dict) -> List[str]:
        """
        Generate specific recommendations for model and strategy improvements.
        """
        recommendations = []
        
        # Classification performance analysis
        class_perf = results['classification_performance']
        overall_acc = class_perf['overall_accuracy']
        conservative_acc = class_perf['conservative_accuracy']
        
        if overall_acc < 0.6:
            recommendations.append("🔄 Overall accuracy below 60% - consider retraining with more features or different architecture")
        
        if conservative_acc < 0.7:
            recommendations.append("⚠️ Conservative strategy accuracy below 70% - increase confidence threshold or add feature engineering")
        
        # Regime-specific improvements
        regime_acc = class_perf['regime_accuracy']
        for regime, accuracy in regime_acc.items():
            if accuracy < 0.55:
                recommendations.append(f"📉 {regime} accuracy low ({accuracy:.1%}) - consider regime-specific model training")
        
        # Trading performance analysis
        if 'trading_performance' in results:
            trading_perf = results['trading_performance']['performance_metrics']
            
            if trading_perf['sharpe_ratio'] < 1.0:
                recommendations.append("📊 Sharpe ratio below 1.0 - consider risk management parameter tuning")
            
            if trading_perf['max_drawdown'] < -0.1:
                recommendations.append("🛡️ Maximum drawdown exceeds -10% - implement tighter stop losses")
            
            if trading_perf['win_rate'] < 0.5:
                recommendations.append("🎯 Win rate below 50% - review entry/exit criteria and position sizing")
            
            if trading_perf['excess_return'] < 0:
                recommendations.append("📈 Strategy underperforming buy-and-hold - consider reducing trading frequency")
        
        # Data quality recommendations
        data_coverage = class_perf.get('data_coverage', 1.0)
        if data_coverage < 0.8:
            recommendations.append("📊 Low data coverage - improve data quality and reduce missing predictions")
        
        # Model complexity recommendations
        high_conf_predictions = class_perf.get('high_conf_predictions', 0)
        total_predictions = class_perf.get('total_predictions', 1)
        high_conf_rate = high_conf_predictions / total_predictions
        
        if high_conf_rate < 0.3:
            recommendations.append("🔍 Low high-confidence prediction rate - model may be overconfident or undertrained")
        elif high_conf_rate > 0.8:
            recommendations.append("⚡ Very high confidence rate - model may be underconfident, consider lower thresholds")
        
        return recommendations
    
    def create_performance_visualizations(self, results: Dict):
        """
        Create comprehensive performance visualizations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bitcoin Prediction Model - Backtesting Results', fontsize=16, fontweight='bold')
        
        # 1. Classification Confusion Matrix
        cm = results['classification_performance']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['DOWN', 'NEUTRAL', 'UP'],
                   yticklabels=['DOWN', 'NEUTRAL', 'UP'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Accuracy by Market Regime
        regime_acc = results['classification_performance']['regime_accuracy']
        regimes = list(regime_acc.keys())
        accuracies = [regime_acc[r] for r in regimes]
        
        bars = axes[0, 1].bar(regimes, accuracies, color=['green', 'red', 'orange'])
        axes[0, 1].set_title('Accuracy by Market Regime')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        # Add percentage labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.1%}', ha='center', va='bottom')
        
        # 3. Portfolio Value Over Time
        if 'trading_performance' in results:
            portfolio_df = results['trading_performance']['portfolio_history']
            axes[0, 2].plot(portfolio_df['date'], portfolio_df['total_value'], 
                           label='Strategy', linewidth=2)
            
            # Buy and hold comparison
            initial_price = portfolio_df['price'].iloc[0]
            buy_hold_values = (portfolio_df['price'] / initial_price) * self.initial_capital
            axes[0, 2].plot(portfolio_df['date'], buy_hold_values, 
                           label='Buy & Hold', linewidth=2, alpha=0.7)
            
            axes[0, 2].set_title('Portfolio Performance')
            axes[0, 2].set_ylabel('Portfolio Value ($)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        if 'trading_performance' in results:
            metrics = results['trading_performance']['performance_metrics']
            metric_names = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Win Rate']
            metric_values = [
                metrics['total_return'],
                metrics['annualized_return'], 
                metrics['sharpe_ratio'] / 3,  # Scale for visualization
                metrics['win_rate']
            ]
            
            bars = axes[1, 0].bar(metric_names, metric_values, 
                                color=['blue', 'green', 'orange', 'purple'])
            axes[1, 0].set_title('Key Performance Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.1%}' if 'Ratio' not in bar.get_x() else f'{value*3:.2f}',
                               ha='center', va='bottom')
        
        # 5. Drawdown Analysis
        if 'trading_performance' in results:
            portfolio_df = results['trading_performance']['portfolio_history']
            rolling_max = portfolio_df['total_value'].expanding().max()
            drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
            
            axes[1, 1].fill_between(portfolio_df['date'], drawdown, 0, 
                                  color='red', alpha=0.3)
            axes[1, 1].plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
            axes[1, 1].set_title('Portfolio Drawdown')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confidence vs Accuracy Scatter
        # This would require more detailed prediction data
        axes[1, 2].text(0.5, 0.5, 'Confidence vs Accuracy\nAnalysis\n(Requires detailed\nprediction data)', 
                       ha='center', va='center', transform=axes[1, 2].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Model Calibration')
        
        plt.tight_layout()
        plt.show()
    
    def generate_executive_summary(self, results: Dict) -> str:
        """
        Generate executive summary for banking management.
        """
        class_perf = results['classification_performance']
        
        if 'trading_performance' in results:
            trading_perf = results['trading_performance']['performance_metrics']
            summary = f"""
EXECUTIVE SUMMARY - BITCOIN PREDICTION MODEL BACKTEST
====================================================

CLASSIFICATION PERFORMANCE:
• Overall Model Accuracy: {class_perf['overall_accuracy']:.1%}
• Conservative Strategy Accuracy: {class_perf['conservative_accuracy']:.1%}
• High Confidence Predictions: {class_perf['high_conf_predictions']} / {class_perf['total_predictions']}

TRADING PERFORMANCE:
• Total Return: {trading_perf['total_return']:.1%}
• Annualized Return: {trading_perf['annualized_return']:.1%}
• vs Buy & Hold: {trading_perf['excess_return']:+.1%} excess return
• Sharpe Ratio: {trading_perf['sharpe_ratio']:.2f}
• Maximum Drawdown: {trading_perf['max_drawdown']:.1%}
• Win Rate: {trading_perf['win_rate']:.1%}

RISK ASSESSMENT:
• Strategy demonstrates {'SUPERIOR' if trading_perf['excess_return'] > 0 else 'INFERIOR'} performance vs buy-and-hold
• Risk-adjusted returns {'ACCEPTABLE' if trading_perf['sharpe_ratio'] > 1 else 'REQUIRE IMPROVEMENT'}
• Drawdown levels {'WITHIN' if trading_perf['max_drawdown'] > -0.15 else 'EXCEED'} banking risk tolerance

RECOMMENDATION: {'APPROVE FOR DEPLOYMENT' if class_perf['conservative_accuracy'] > 0.65 and trading_perf['sharpe_ratio'] > 0.8 else 'REQUIRES IMPROVEMENT BEFORE DEPLOYMENT'}
"""
        else:
            summary = f"""
EXECUTIVE SUMMARY - BITCOIN PREDICTION MODEL BACKTEST
====================================================

CLASSIFICATION PERFORMANCE:
• Overall Model Accuracy: {class_perf['overall_accuracy']:.1%}
• Conservative Strategy Accuracy: {class_perf['conservative_accuracy']:.1%}
• High Confidence Predictions: {class_perf['high_conf_predictions']} / {class_perf['total_predictions']}

RECOMMENDATION: {'APPROVE FOR DEPLOYMENT' if class_perf['conservative_accuracy'] > 0.65 else 'REQUIRES IMPROVEMENT BEFORE DEPLOYMENT'}
"""
        
        return summary

# Example usage and demonstration
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BITCOIN PREDICTION BACKTESTING SYSTEM")
    print("="*70)
    
    # Initialize backtesting system
    backtest_system = BitcoinBacktestingSystem(initial_capital=100000)
    
    print(f"\n🎯 BACKTESTING OBJECTIVES:")
    print(f"   ✅ Measure classification accuracy across market conditions")
    print(f"   ✅ Evaluate risk-adjusted trading performance")
    print(f"   ✅ Compare against buy-and-hold benchmark")
    print(f"   ✅ Identify specific areas for improvement")
    print(f"   ✅ Generate banking compliance report")
    
    print(f"\n📊 KEY METRICS TO EVALUATE:")
    print(f"   • Overall classification accuracy (target: >60%)")
    print(f"   • Conservative strategy accuracy (target: >70%)")
    print(f"   • Sharpe ratio (target: >1.0)")
    print(f"   • Maximum drawdown (limit: <15%)")
    print(f"   • Win rate (target: >50%)")
    print(f"   • Excess return vs buy-and-hold")
    
    print(f"\n🏦 BANKING RISK MANAGEMENT:")
    print(f"   • Maximum position size: {backtest_system.max_position_size:.1%}")
    print(f"   • Stop loss: {backtest_system.stop_loss:.1%}")
    print(f"   • Take profit: {backtest_system.take_profit:.1%}")
    print(f"   • Emergency drawdown limit: {backtest_system.max_drawdown_limit:.1%}")
    
    print(f"\n🔄 NEXT STEPS FOR IMPLEMENTATION:")
    print(f"   1. Load historical Bitcoin price data")
    print(f"   2. Load Wikipedia sentiment analysis results")
    print(f"   3. Load model predictions with confidence scores")
    print(f"   4. Run: backtest_data = system.prepare_backtest_data(price_df, sentiment_df, predictions_df)")
    print(f"   5. Run: results = system.run_comprehensive_analysis(backtest_data)")
    print(f"   6. Review performance and implement improvements")
    
    print(f"\n" + "="*70)
    print("🚀 READY FOR COMPREHENSIVE BACKTESTING")
    print("="*70)
