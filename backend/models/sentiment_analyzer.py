import re
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from collections import Counter

class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer combining TextBlob and VADER
    Optimized for social media text analysis
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.product_keywords = []
        
    def clean_text(self, text):
        """Clean and preprocess social media text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s!?.,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'classification': self._classify_sentiment(polarity)
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'classification': 'neutral'}
    
    def analyze_with_vader(self, text):
        """Analyze sentiment using VADER (better for social media)"""
        try:
            scores = self.vader.polarity_scores(text)
            
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'classification': self._classify_vader(scores['compound'])
            }
        except:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'classification': 'neutral'}
    
    def _classify_sentiment(self, score):
        """Classify sentiment based on polarity score"""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_vader(self, compound):
        """Classify sentiment based on VADER compound score"""
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def ensemble_analysis(self, text):
        """Combine TextBlob and VADER for robust sentiment analysis"""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return self._default_sentiment()
        
        # Get both analyses
        textblob_result = self.analyze_with_textblob(cleaned_text)
        vader_result = self.analyze_with_vader(cleaned_text)
        
        # Weighted ensemble (VADER weighted higher for social media)
        ensemble_score = (0.4 * textblob_result['polarity'] + 0.6 * vader_result['compound'])
        
        return {
            'text': cleaned_text,
            'textblob_score': textblob_result['polarity'],
            'vader_score': vader_result['compound'],
            'ensemble_score': ensemble_score,
            'classification': self._classify_sentiment(ensemble_score),
            'confidence': abs(ensemble_score),
            'vader_breakdown': {
                'positive': vader_result['positive'],
                'negative': vader_result['negative'],
                'neutral': vader_result['neutral']
            }
        }
    
    def _default_sentiment(self):
        """Return default neutral sentiment"""
        return {
            'text': '',
            'textblob_score': 0,
            'vader_score': 0,
            'ensemble_score': 0,
            'classification': 'neutral',
            'confidence': 0,
            'vader_breakdown': {'positive': 0, 'negative': 0, 'neutral': 1}
        }
    
    def batch_analyze(self, texts):
        """Analyze multiple texts efficiently"""
        results = []
        
        for text in texts:
            result = self.ensemble_analysis(text)
            results.append(result)
        
        return results
    
    def extract_hashtags(self, text):
        """Extract hashtags from text"""
        if not isinstance(text, str):
            return []
        
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags
    
    def extract_mentions(self, text):
        """Extract user mentions from text"""
        if not isinstance(text, str):
            return []
        
        mentions = re.findall(r'@(\w+)', text)
        return mentions
    
    def get_trending_hashtags(self, texts, top_n=10):
        """Get most frequent hashtags from a list of texts"""
        all_hashtags = []
        
        for text in texts:
            hashtags = self.extract_hashtags(text)
            all_hashtags.extend(hashtags)
        
        counter = Counter(all_hashtags)
        return counter.most_common(top_n)
    
    def analyze_product_sentiment(self, texts, product_keywords):
        """Analyze sentiment specifically for product mentions"""
        self.product_keywords = [kw.lower() for kw in product_keywords]
        
        relevant_texts = []
        sentiments = []
        
        for text in texts:
            cleaned = self.clean_text(text)
            
            # Check if text mentions product
            if any(keyword in cleaned for keyword in self.product_keywords):
                relevant_texts.append(text)
                sentiment = self.ensemble_analysis(text)
                sentiments.append(sentiment)
        
        if not sentiments:
            return {
                'total_mentions': 0,
                'average_sentiment': 0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'positive_percentage': 0,
                'neutral_percentage': 0,
                'negative_percentage': 0
            }
        
        # Calculate statistics
        classifications = [s['classification'] for s in sentiments]
        avg_sentiment = np.mean([s['ensemble_score'] for s in sentiments])
        
        positive_count = classifications.count('positive')
        neutral_count = classifications.count('neutral')
        negative_count = classifications.count('negative')
        total = len(classifications)
        
        return {
            'total_mentions': total,
            'average_sentiment': round(avg_sentiment, 3),
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'positive_percentage': round((positive_count / total) * 100, 1),
            'neutral_percentage': round((neutral_count / total) * 100, 1),
            'negative_percentage': round((negative_count / total) * 100, 1),
            'sentiment_trend': 'increasing' if avg_sentiment > 0.1 else 'decreasing' if avg_sentiment < -0.1 else 'stable'
        }
    
    def get_sentiment_time_series(self, df, text_column='text', date_column='date'):
        """Generate time series of sentiment scores"""
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Analyze sentiment
        sentiments = self.batch_analyze(df['cleaned_text'].tolist())
        df['sentiment_score'] = [s['ensemble_score'] for s in sentiments]
        df['sentiment_class'] = [s['classification'] for s in sentiments]
        
        # Group by date
        df[date_column] = pd.to_datetime(df[date_column])
        time_series = df.groupby(df[date_column].dt.date).agg({
            'sentiment_score': 'mean',
            'sentiment_class': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        return time_series
    
    def calculate_sentiment_volatility(self, sentiment_scores):
        """Calculate sentiment volatility (useful for SAFA metric)"""
        if len(sentiment_scores) < 2:
            return 0
        
        # Calculate standard deviation
        volatility = np.std(sentiment_scores)
        
        # Calculate rate of change
        changes = np.diff(sentiment_scores)
        avg_change = np.mean(np.abs(changes))
        
        return {
            'volatility': round(volatility, 3),
            'average_change': round(avg_change, 3),
            'stability': 'high' if volatility < 0.2 else 'medium' if volatility < 0.5 else 'low'
        }
    
    def generate_sentiment_report(self, texts, product_keywords=None):
        """Generate comprehensive sentiment analysis report"""
        # Analyze all texts
        results = self.batch_analyze(texts)
        
        # Overall statistics
        scores = [r['ensemble_score'] for r in results]
        classifications = [r['classification'] for r in results]
        
        positive_count = classifications.count('positive')
        neutral_count = classifications.count('neutral')
        negative_count = classifications.count('negative')
        total = len(classifications)
        
        # Trending hashtags
        trending_hashtags = self.get_trending_hashtags(texts, top_n=10)
        
        report = {
            'summary': {
                'total_analyzed': total,
                'average_sentiment': round(np.mean(scores), 3),
                'median_sentiment': round(np.median(scores), 3),
                'sentiment_std': round(np.std(scores), 3),
            },
            'distribution': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count,
                'positive_pct': round((positive_count / total) * 100, 1),
                'neutral_pct': round((neutral_count / total) * 100, 1),
                'negative_pct': round((negative_count / total) * 100, 1)
            },
            'trending_hashtags': [{'tag': tag, 'count': count} for tag, count in trending_hashtags],
            'volatility': self.calculate_sentiment_volatility(scores)
        }
        
        # Product-specific analysis if keywords provided
        if product_keywords:
            report['product_sentiment'] = self.analyze_product_sentiment(texts, product_keywords)
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Sample social media posts
    sample_texts = [
        "Just bought these wireless earbuds and they're amazing! 🎧 #tech #gadgets",
        "The smart watch is okay but battery life could be better",
        "Terrible experience with customer service 😡 #disappointed",
        "@TechBrand your products are always top quality! Love it ❤️",
        "Not sure if I should buy this... anyone have reviews?"
    ]
    
    # Generate report
    report = analyzer.generate_sentiment_report(
        sample_texts,
        product_keywords=['earbuds', 'smart watch', 'wireless']
    )
    
    print("📊 Sentiment Analysis Report")
    print("=" * 50)
    print(f"Total Analyzed: {report['summary']['total_analyzed']}")
    print(f"Average Sentiment: {report['summary']['average_sentiment']}")
    print(f"\nDistribution:")
    print(f"  Positive: {report['distribution']['positive_pct']}%")
    print(f"  Neutral: {report['distribution']['neutral_pct']}%")
    print(f"  Negative: {report['distribution']['negative_pct']}%")
    print(f"\nSentiment Volatility: {report['volatility']['stability']}")
    
    if 'product_sentiment' in report:
        print(f"\nProduct Mentions: {report['product_sentiment']['total_mentions']}")
        print(f"Product Sentiment: {report['product_sentiment']['average_sentiment']}")