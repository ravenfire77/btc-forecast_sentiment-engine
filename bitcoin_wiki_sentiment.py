"""
Bitcoin Wikipedia Sentiment Analysis Engine
==========================================
Analyzes daily Wikipedia edits on Bitcoin page for sentiment classification.
Designed for 24/7 dashboard deployment and real-time sentiment tracking.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from textblob import TextBlob
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinWikipediaSentimentEngine:
    """
    Comprehensive sentiment analysis engine for Bitcoin Wikipedia page edits.
    
    Features:
    - Complete revision history extraction via Wikipedia API
    - Multi-model sentiment analysis (VADER, TextBlob, custom crypto-aware)
    - Daily aggregation with zero-edit handling
    - Historical trend analysis
    - Real-time monitoring capabilities
    """
    
    def __init__(self, user_agent: str = "BitcoinSentimentBot/1.0"):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": user_agent}
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Crypto-specific sentiment keywords for enhanced analysis
        self.bullish_keywords = {
            'adoption', 'breakthrough', 'milestone', 'surge', 'rally', 'moon', 
            'bullish', 'institutional', 'mainstream', 'breakthrough', 'innovation',
            'partnership', 'integration', 'approval', 'legal tender', 'etf'
        }
        
        self.bearish_keywords = {
            'crash', 'dump', 'bear', 'regulation', 'ban', 'illegal', 'hack',
            'security', 'vulnerability', 'scam', 'bubble', 'ponzi', 'crime',
            'energy', 'environmental', 'volatility', 'manipulation'
        }
        
    def get_all_revisions(self, title: str = "Bitcoin") -> List[Dict]:
        """
        Retrieve complete revision history for Bitcoin Wikipedia page.
        Handles pagination to get ALL revisions from first edit to present.
        """
        all_revisions = []
        continue_param = None
        
        logger.info(f"Fetching revision history for {title}...")
        
        while True:
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'revisions',
                'titles': title,
                'rvlimit': 500,  # Maximum allowed per request
                'rvprop': 'ids|timestamp|user|comment|size',
                'rvdir': 'newer',  # Start from oldest (first edit)
                'formatversion': 2
            }
            
            if continue_param:
                params['rvcontinue'] = continue_param
                
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    logger.error(f"API Error: {data['error']}")
                    break
                    
                pages = data.get('query', {}).get('pages', [])
                if not pages:
                    logger.warning("No pages found")
                    break
                    
                revisions = pages[0].get('revisions', [])
                all_revisions.extend(revisions)
                
                # Check for continuation
                if 'continue' in data:
                    continue_param = data['continue']['rvcontinue']
                    logger.info(f"Retrieved {len(all_revisions)} revisions so far...")
                    time.sleep(0.1)  # Rate limiting
                else:
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
                
        logger.info(f"Total revisions retrieved: {len(all_revisions)}")
        return all_revisions
    
    def analyze_edit_sentiment(self, comment: str) -> Dict[str, float]:
        """
        Multi-model sentiment analysis of edit comments.
        Combines VADER, TextBlob, and crypto-aware analysis.
        """
        if not comment or comment.strip() == "":
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'textblob_polarity': 0.0,
                'crypto_sentiment': 0.0,
                'final_sentiment': 'neutral'
            }
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(comment.lower())
        
        # TextBlob sentiment analysis
        try:
            blob = TextBlob(comment)
            textblob_polarity = blob.sentiment.polarity
        except:
            textblob_polarity = 0.0
        
        # Crypto-specific sentiment analysis
        comment_lower = comment.lower()
        crypto_score = 0.0
        
        for word in self.bullish_keywords:
            if word in comment_lower:
                crypto_score += 0.2
                
        for word in self.bearish_keywords:
            if word in comment_lower:
                crypto_score -= 0.2
        
        # Normalize crypto score
        crypto_score = max(-1.0, min(1.0, crypto_score))
        
        # Combine all sentiment scores with weights
        final_compound = (
            vader_scores['compound'] * 0.5 +
            textblob_polarity * 0.3 +
            crypto_score * 0.2
        )
        
        # Classify final sentiment
        if final_compound >= 0.05:
            final_sentiment = 'positive'
        elif final_compound <= -0.05:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'crypto_sentiment': crypto_score,
            'final_sentiment': final_sentiment,
            'final_compound': final_compound
        }
    
    def aggregate_daily_sentiment(self, revisions: List[Dict]) -> pd.DataFrame:
        """
        Aggregate revisions by day and calculate daily sentiment scores.
        Handles days with zero edits per analyst requirements.
        """
        daily_data = []
        
        # Process each revision
        for revision in revisions:
            timestamp = revision.get('timestamp', '')
            if not timestamp:
                continue
                
            try:
                # Parse timestamp
                date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                comment = revision.get('comment', '')
                user = revision.get('user', 'Anonymous')
                size = revision.get('size', 0)
                
                # Analyze sentiment
                sentiment = self.analyze_edit_sentiment(comment)
                
                daily_data.append({
                    'date': date,
                    'timestamp': timestamp,
                    'user': user,
                    'comment': comment,
                    'size': size,
                    'revid': revision.get('revid'),
                    **sentiment
                })
                
            except Exception as e:
                logger.warning(f"Error processing revision: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(daily_data)
        
        if df.empty:
            logger.warning("No valid revisions found")
            return pd.DataFrame()
        
        # Group by date and aggregate
        daily_agg = df.groupby('date').agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'textblob_polarity': 'mean',
            'crypto_sentiment': 'mean',
            'final_compound': 'mean',
            'revid': 'count',  # Number of edits per day
            'size': 'last',    # Final page size for the day
            'comment': lambda x: ' | '.join(x)  # Concatenate all comments
        }).reset_index()
        
        # Rename columns for clarity
        daily_agg.rename(columns={'revid': 'edit_count'}, inplace=True)
        
        # Fill missing dates with neutral sentiment (analyst requirement)
        if not daily_agg.empty:
            date_range = pd.date_range(
                start=daily_agg['date'].min(),
                end=daily_agg['date'].max(),
                freq='D'
            )
            
            # Create complete date index
            complete_df = pd.DataFrame({'date': date_range.date})
            daily_agg = complete_df.merge(daily_agg, on='date', how='left')
            
            # Fill missing values for zero-edit days
            sentiment_columns = ['compound', 'positive', 'negative', 'neutral', 
                               'textblob_polarity', 'crypto_sentiment', 'final_compound']
            
            for col in sentiment_columns:
                daily_agg[col] = daily_agg[col].fillna(0.0)
            
            daily_agg['edit_count'] = daily_agg['edit_count'].fillna(0)
            daily_agg['comment'] = daily_agg['comment'].fillna('No edits')
            
            # Classify final sentiment for each day
            daily_agg['daily_sentiment'] = daily_agg['final_compound'].apply(
                lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
            )
        
        return daily_agg
    
    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive sentiment analysis report.
        """
        if df.empty:
            return {"error": "No data available for analysis"}
        
        total_days = len(df)
        zero_edit_days = len(df[df['edit_count'] == 0])
        
        sentiment_distribution = df['daily_sentiment'].value_counts()
        avg_daily_edits = df['edit_count'].mean()
        
        # Calculate sentiment trends
        df_sorted = df.sort_values('date')
        recent_30_days = df_sorted.tail(30) if len(df_sorted) >= 30 else df_sorted
        
        report = {
            'analysis_period': {
                'start_date': str(df['date'].min()),
                'end_date': str(df['date'].max()),
                'total_days': total_days,
                'days_with_edits': total_days - zero_edit_days,
                'zero_edit_days': zero_edit_days,
                'zero_edit_percentage': round((zero_edit_days / total_days) * 100, 2)
            },
            'overall_sentiment': {
                'positive_days': int(sentiment_distribution.get('positive', 0)),
                'negative_days': int(sentiment_distribution.get('negative', 0)),
                'neutral_days': int(sentiment_distribution.get('neutral', 0)),
                'dominant_sentiment': sentiment_distribution.index[0] if not sentiment_distribution.empty else 'neutral'
            },
            'edit_statistics': {
                'total_edits': int(df['edit_count'].sum()),
                'average_daily_edits': round(avg_daily_edits, 2),
                'max_daily_edits': int(df['edit_count'].max()),
                'most_active_day': str(df.loc[df['edit_count'].idxmax(), 'date'])
            },
            'recent_trend': {
                'last_30_days_sentiment': recent_30_days['daily_sentiment'].value_counts().to_dict(),
                'recent_average_compound': round(recent_30_days['final_compound'].mean(), 3),
                'trend_direction': 'improving' if recent_30_days['final_compound'].tail(7).mean() > 
                                 recent_30_days['final_compound'].head(7).mean() else 'declining'
            }
        }
        
        return report
    
    def run_complete_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute complete sentiment analysis pipeline.
        Returns processed data and comprehensive report.
        """
        logger.info("Starting complete Bitcoin Wikipedia sentiment analysis...")
        
        # Step 1: Get all revisions
        revisions = self.get_all_revisions()
        
        if not revisions:
            logger.error("No revisions retrieved")
            return pd.DataFrame(), {"error": "Failed to retrieve data"}
        
        # Step 2: Process and aggregate daily sentiment
        daily_sentiment_df = self.aggregate_daily_sentiment(revisions)
        
        # Step 3: Generate report
        report = self.generate_sentiment_report(daily_sentiment_df)
        
        logger.info("Analysis complete!")
        return daily_sentiment_df, report
    
    def visualize_sentiment_trends(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive sentiment visualization dashboard.
        """
        if df.empty:
            print("No data to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Wikipedia Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Daily sentiment trends
        ax1 = axes[0, 0]
        df_plot = df.sort_values('date')
        ax1.plot(df_plot['date'], df_plot['final_compound'], linewidth=2, alpha=0.8)
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Positive threshold')
        ax1.axhline(y=-0.05, color='red', linestyle='--', alpha=0.7, label='Negative threshold')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_title('Daily Sentiment Compound Score')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Sentiment distribution
        ax2 = axes[0, 1]
        sentiment_counts = df['daily_sentiment'].value_counts()
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[colors.get(x, 'blue') for x in sentiment_counts.index])
        ax2.set_title('Overall Sentiment Distribution')
        ax2.set_ylabel('Number of Days')
        
        # Add percentage labels on bars
        total = sentiment_counts.sum()
        for bar, count in zip(bars, sentiment_counts.values):
            percentage = (count / total) * 100
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 3. Edit activity vs sentiment
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['edit_count'], df['final_compound'], 
                            c=df['final_compound'], cmap='RdYlGn', alpha=0.6)
        ax3.set_title('Edit Activity vs Sentiment')
        ax3.set_xlabel('Daily Edit Count')
        ax3.set_ylabel('Sentiment Score')
        plt.colorbar(scatter, ax=ax3, label='Sentiment')
        
        # 4. Moving average trend
        ax4 = axes[1, 1]
        if len(df) >= 7:
            df_sorted = df.sort_values('date')
            df_sorted['sentiment_ma7'] = df_sorted['final_compound'].rolling(window=7, center=True).mean()
            ax4.plot(df_sorted['date'], df_sorted['sentiment_ma7'], 'b-', linewidth=2, label='7-day MA')
            if len(df) >= 30:
                df_sorted['sentiment_ma30'] = df_sorted['final_compound'].rolling(window=30, center=True).mean()
                ax4.plot(df_sorted['date'], df_sorted['sentiment_ma30'], 'r-', linewidth=2, label='30-day MA')
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_title('Sentiment Moving Averages')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Sentiment Score')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor moving averages', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sentiment Moving Averages')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Initialize the sentiment engine
    engine = BitcoinWikipediaSentimentEngine()
    
    # Run complete analysis
    sentiment_data, analysis_report = engine.run_complete_analysis()
    
    # Display results
    if not sentiment_data.empty:
        print("\n" + "="*60)
        print("BITCOIN WIKIPEDIA SENTIMENT ANALYSIS REPORT")
        print("="*60)
        print(json.dumps(analysis_report, indent=2))
        
        print(f"\nFirst 10 days of sentiment data:")
        print(sentiment_data.head(10))
        
        print(f"\nLast 10 days of sentiment data:")
        print(sentiment_data.tail(10))
        
        # Generate visualizations
        engine.visualize_sentiment_trends(sentiment_data)
        
        # Save data for ML model (next step)
        sentiment_data.to_csv('bitcoin_wikipedia_sentiment_daily.csv', index=False)
        
        with open('sentiment_analysis_report.json', 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        print(f"\nData saved to:")
        print("- bitcoin_wikipedia_sentiment_daily.csv")
        print("- sentiment_analysis_report.json")
        
    else:
        print("No data retrieved. Please check your internet connection and try again.")
