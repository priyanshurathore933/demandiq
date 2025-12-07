"""
Complete Automated Pipeline for DemandIQ
This script orchestrates the entire workflow:
1. Data collection from social media
2. Sentiment analysis
3. Model training
4. Forecasting
5. Database updates
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collectors.instagram_collector import InstagramCollector
from data_collectors.reddit_collector import RedditCollector
from models.sentiment_analyzer import SentimentAnalyzer
from models.forecasting_model import HybridForecastingModel

class DemandIQPipeline:
    """Complete pipeline for demand forecasting with social trends"""
    
    def __init__(self, db_path='database.db'):
        self.db_path = db_path
        self.instagram_collector = None
        self.reddit_collector = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.forecast_model = HybridForecastingModel()
        
        # Configuration
        self.product_keywords = [
            'earbuds', 'smart watch', 'smartwatch', 'wireless',
            'bluetooth', 'portable charger', 'phone case', 'laptop stand'
        ]
        
        self.instagram_hashtags = [
            'techgadgets', 'wirelessearbuds', 'smartwatch',
            'techdeals', 'gadgetlover', 'techreview'
        ]
        
        self.reddit_subreddits = [
            'technology', 'gadgets', 'tech', 'headphones',
            'BuyItForLife', 'ProductPorn'
        ]
    
    def initialize_collectors(self, reddit_client_id=None, reddit_secret=None):
        """Initialize social media collectors"""
        print("🔧 Initializing data collectors...")
        
        # Instagram
        self.instagram_collector = InstagramCollector()
        
        # Reddit
        if reddit_client_id and reddit_secret:
            self.reddit_collector = RedditCollector(
                client_id=reddit_client_id,
                client_secret=reddit_secret
            )
        else:
            print("⚠️ Reddit credentials not provided. Skipping Reddit collection.")
    
    def step1_collect_social_data(self):
        """Step 1: Collect data from social media"""
        print("\n" + "="*60)
        print("📱 STEP 1: COLLECTING SOCIAL MEDIA DATA")
        print("="*60)
        
        all_social_data = []
        
        # Collect Instagram data
        if self.instagram_collector:
            try:
                print("\n📸 Collecting Instagram data...")
                instagram_posts = self.instagram_collector.collect_multiple_hashtags(
                    self.instagram_hashtags,
                    max_posts_per_tag=50,
                    days_back=7
                )
                
                # Filter for product mentions
                relevant_ig = self.instagram_collector.extract_product_mentions(
                    instagram_posts,
                    self.product_keywords
                )
                
                print(f"✅ Collected {len(relevant_ig)} relevant Instagram posts")
                
                # Format for database
                for post in relevant_ig:
                    all_social_data.append({
                        'platform': 'Instagram',
                        'text': post.get('caption', ''),
                        'engagement': post.get('engagement', 0),
                        'date': post.get('date'),
                        'post_id': post.get('post_id')
                    })
                    
            except Exception as e:
                print(f"❌ Instagram collection failed: {e}")
        
        # Collect Reddit data
        if self.reddit_collector:
            try:
                print("\n🤖 Collecting Reddit data...")
                reddit_posts = self.reddit_collector.collect_multiple_subreddits(
                    self.reddit_subreddits,
                    limit_per_sub=50
                )
                
                # Filter for product mentions
                relevant_reddit = self.reddit_collector.filter_product_mentions(
                    reddit_posts,
                    self.product_keywords
                )
                
                print(f"✅ Collected {len(relevant_reddit)} relevant Reddit posts")
                
                # Format for database
                for post in relevant_reddit:
                    all_social_data.append({
                        'platform': 'Reddit',
                        'text': f"{post.get('title', '')} {post.get('selftext', '')}",
                        'engagement': post.get('score', 0) + post.get('num_comments', 0),
                        'date': post.get('created_utc', '')[:10],
                        'post_id': post.get('post_id')
                    })
                    
            except Exception as e:
                print(f"❌ Reddit collection failed: {e}")
        
        print(f"\n📊 Total social media posts collected: {len(all_social_data)}")
        return all_social_data
    
    def step2_analyze_sentiment(self, social_data):
        """Step 2: Analyze sentiment of collected data"""
        print("\n" + "="*60)
        print("🎭 STEP 2: ANALYZING SENTIMENT")
        print("="*60)
        
        if not social_data:
            print("⚠️ No social data to analyze")
            return []
        
        sentiment_results = []
        
        for item in social_data:
            try:
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.ensemble_analysis(item['text'])
                
                # Combine with original data
                result = {
                    'platform': item['platform'],
                    'text': sentiment['text'],
                    'sentiment_score': sentiment['ensemble_score'],
                    'sentiment_class': sentiment['classification'],
                    'engagement': item['engagement'],
                    'date': item['date'],
                    'post_id': item['post_id']
                }
                
                sentiment_results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error analyzing sentiment: {e}")
                continue
        
        # Generate report
        if sentiment_results:
            report = self.sentiment_analyzer.generate_sentiment_report(
                [r['text'] for r in sentiment_results],
                product_keywords=self.product_keywords
            )
            
            print(f"\n📈 Sentiment Analysis Report:")
            print(f"   Total Analyzed: {report['summary']['total_analyzed']}")
            print(f"   Average Sentiment: {report['summary']['average_sentiment']:.3f}")
            print(f"   Positive: {report['distribution']['positive_pct']}%")
            print(f"   Neutral: {report['distribution']['neutral_pct']}%")
            print(f"   Negative: {report['distribution']['negative_pct']}%")
        
        return sentiment_results
    
    def step3_save_to_database(self, sentiment_data):
        """Step 3: Save sentiment data to database"""
        print("\n" + "="*60)
        print("💾 STEP 3: SAVING TO DATABASE")
        print("="*60)
        
        if not sentiment_data:
            print("⚠️ No data to save")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        inserted = 0
        for item in sentiment_data:
            try:
                # Aggregate by product and date
                product = 'General'  # You can enhance this with keyword matching
                
                c.execute('''
                    INSERT INTO social_data 
                    (platform, product, mentions, sentiment_score, engagement, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                ''', (
                    item['platform'],
                    product,
                    1,
                    item['sentiment_score'],
                    item['engagement'],
                    item['date']
                ))
                
                inserted += 1
                
            except Exception as e:
                print(f"⚠️ Error saving to database: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {inserted} records to database")
        return inserted
    
    def step4_prepare_training_data(self):
        """Step 4: Prepare data for model training"""
        print("\n" + "="*60)
        print("🔧 STEP 4: PREPARING TRAINING DATA")
        print("="*60)
        
        conn = sqlite3.connect(self.db_path)
        
        # Load sales data
        sales_df = pd.read_sql_query('''
            SELECT date, product, quantity, price, revenue
            FROM sales_data
            ORDER BY date
        ''', conn)
        
        # Load sentiment data
        sentiment_df = pd.read_sql_query('''
            SELECT date, platform, AVG(sentiment_score) as avg_sentiment,
                   SUM(mentions) as total_mentions,
                   SUM(engagement) as total_engagement
            FROM social_data
            GROUP BY date, platform
        ''', conn)
        
        conn.close()
        
        if sales_df.empty:
            print("⚠️ No sales data available. Please upload historical sales data.")
            return None, None
        
        print(f"✅ Loaded {len(sales_df)} sales records")
        print(f"✅ Loaded {len(sentiment_df)} sentiment records")
        
        # Merge datasets
        if not sentiment_df.empty:
            merged_df = sales_df.merge(
                sentiment_df,
                on='date',
                how='left'
            )
            merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
        else:
            merged_df = sales_df.copy()
            merged_df['avg_sentiment'] = 0
        
        return merged_df, sentiment_df
    
    def step5_train_model(self, merged_df):
        """Step 5: Train forecasting model"""
        print("\n" + "="*60)
        print("🤖 STEP 5: TRAINING FORECASTING MODEL")
        print("="*60)
        
        if merged_df is None or merged_df.empty:
            print("⚠️ No data available for training")
            return False
        
        try:
            # Prepare data
            quantities = merged_df['quantity'].values
            sentiment_features = merged_df['avg_sentiment'].values if 'avg_sentiment' in merged_df.columns else None
            
            # Train LSTM
            print("\n🧠 Training LSTM model...")
            self.forecast_model.train_lstm(
                quantities,
                sentiment_features=sentiment_features,
                epochs=30,
                batch_size=16
            )
            
            # Train XGBoost
            print("\n🌲 Training XGBoost model...")
            sentiment_data = {
                'sentiment_score': merged_df['avg_sentiment'].mean() if 'avg_sentiment' in merged_df.columns else 0,
                'mentions': merged_df['total_mentions'].sum() if 'total_mentions' in merged_df.columns else 0,
                'engagement': merged_df['total_engagement'].sum() if 'total_engagement' in merged_df.columns else 0
            }
            
            self.forecast_model.train_xgboost(merged_df, sentiment_data)
            
            # Save model
            print("\n💾 Saving trained model...")
            self.forecast_model.save_model('models/')
            
            print("\n✅ Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
    
    def step6_generate_forecasts(self, merged_df, days=30):
        """Step 6: Generate forecasts"""
        print("\n" + "="*60)
        print(f"🔮 STEP 6: GENERATING {days}-DAY FORECASTS")
        print("="*60)
        
        if merged_df is None or merged_df.empty:
            print("⚠️ No data available for forecasting")
            return None
        
        try:
            # Prepare data
            quantities = merged_df['quantity'].values
            sentiment_features = merged_df['avg_sentiment'].values if 'avg_sentiment' in merged_df.columns else None
            
            # Generate forecasts
            predictions, lower_bounds, upper_bounds = self.forecast_model.forecast_with_confidence(
                quantities,
                sentiment_features=sentiment_features,
                df=merged_df,
                steps=days
            )
            
            # Create forecast dataframe
            today = datetime.now()
            forecast_dates = [today + timedelta(days=i+1) for i in range(days)]
            
            forecast_df = pd.DataFrame({
                'date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'predicted_quantity': predictions.astype(int),
                'lower_bound': lower_bounds.astype(int),
                'upper_bound': upper_bounds.astype(int),
                'confidence': 0.95
            })
            
            print(f"\n📊 Forecast Summary:")
            print(f"   Average predicted demand: {predictions.mean():.0f} units/day")
            print(f"   Peak demand: {predictions.max():.0f} units")
            print(f"   Low demand: {predictions.min():.0f} units")
            print(f"   Total forecast period: {days} days")
            
            # Save forecasts to database
            self._save_forecasts(forecast_df)
            
            return forecast_df
            
        except Exception as e:
            print(f"❌ Forecasting failed: {e}")
            return None
    
    def _save_forecasts(self, forecast_df):
        """Save forecasts to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for _, row in forecast_df.iterrows():
            c.execute('''
                INSERT INTO forecasts 
                (product, forecast_date, predicted_quantity, lower_bound, 
                 upper_bound, confidence, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'General',
                row['date'],
                int(row['predicted_quantity']),
                int(row['lower_bound']),
                int(row['upper_bound']),
                row['confidence'],
                'LSTM+XGBoost v1.0',
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(forecast_df)} forecasts to database")
    
    def run_complete_pipeline(self, reddit_client_id=None, reddit_secret=None):
        """Run the complete pipeline"""
        print("\n" + "🚀"*30)
        print("   DEMANDIQ COMPLETE PIPELINE")
        print("🚀"*30)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Initialize
            self.initialize_collectors(reddit_client_id, reddit_secret)
            
            # Step 1: Collect social data
            social_data = self.step1_collect_social_data()
            
            # Step 2: Analyze sentiment
            sentiment_data = self.step2_analyze_sentiment(social_data)
            
            # Step 3: Save to database
            self.step3_save_to_database(sentiment_data)
            
            # Step 4: Prepare training data
            merged_df, sentiment_df = self.step4_prepare_training_data()
            
            # Step 5: Train model
            if merged_df is not None:
                training_success = self.step5_train_model(merged_df)
                
                # Step 6: Generate forecasts
                if training_success:
                    forecast_df = self.step6_generate_forecasts(merged_df, days=30)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total execution time: {elapsed_time/60:.2f} minutes")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise

# Main execution
if __name__ == "__main__":
    # Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'your_client_id')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'your_client_secret')
    
    # Initialize and run pipeline
    pipeline = DemandIQPipeline()
    pipeline.run_complete_pipeline(
        reddit_client_id=REDDIT_CLIENT_ID,
        reddit_secret=REDDIT_CLIENT_SECRET
    )
    
    print("\n🎉 DemandIQ is ready to forecast!")
    print("📊 Access the dashboard at: http://localhost:5173")
    print("🔌 API available at: http://localhost:5000")