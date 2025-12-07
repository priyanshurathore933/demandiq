import praw
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import time
import os

class RedditCollector:
    """
    Collect Reddit data using PRAW (Python Reddit API Wrapper)
    Features: posts, comments, upvotes, discussions from relevant subreddits
    """
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize Reddit API connection
        
        To get credentials:
        1. Go to https://www.reddit.com/prefs/apps
        2. Click "create another app"
        3. Select "script" type
        4. Copy client_id and client_secret
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or 'DemandIQ-Forecasting-Bot/1.0'
        
        self.reddit = None
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Test connection
            self.reddit.user.me()
            print("✅ Connected to Reddit API")
            
        except Exception as e:
            print(f"⚠️ Reddit API connection failed: {e}")
            print("Using read-only mode (limited functionality)")
            
            # Fallback to read-only
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                check_for_async=False
            )
    
    def collect_subreddit_posts(self, subreddit_name, limit=100, time_filter='week', sort_by='hot'):
        """
        Collect posts from a subreddit
        
        Args:
            subreddit_name: Name of subreddit (e.g., 'technology')
            limit: Maximum number of posts to collect
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            sort_by: 'hot', 'new', 'top', 'rising'
        """
        print(f"🔍 Collecting posts from r/{subreddit_name}...")
        
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sort type
            if sort_by == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort_by == 'new':
                posts = subreddit.new(limit=limit)
            elif sort_by == 'top':
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_by == 'rising':
                posts = subreddit.rising(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            for post in posts:
                try:
                    post_data = {
                        'post_id': post.id,
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'selftext': post.selftext,
                        'author': str(post.author) if post.author else '[deleted]',
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'url': post.url,
                        'is_self': post.is_self,
                        'link_flair_text': post.link_flair_text,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    posts_data.append(post_data)
                    
                except Exception as e:
                    print(f"⚠️ Error processing post: {e}")
                    continue
            
            print(f"✅ Collected {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            print(f"❌ Error collecting from r/{subreddit_name}: {e}")
        
        return posts_data
    
    def collect_post_comments(self, post_id, limit=50):
        """Collect comments from a specific post"""
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                try:
                    comment_data = {
                        'comment_id': comment.id,
                        'post_id': post_id,
                        'body': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'is_submitter': comment.is_submitter,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    comments_data.append(comment_data)
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"⚠️ Error collecting comments for post {post_id}: {e}")
        
        return comments_data
    
    def search_keyword(self, keyword, subreddits=None, limit=100, time_filter='week'):
        """
        Search Reddit for specific keywords
        
        Args:
            keyword: Search term
            subreddits: List of subreddits to search (None = all)
            limit: Max results
            time_filter: Time range for search
        """
        print(f"🔍 Searching Reddit for '{keyword}'...")
        
        results = []
        
        try:
            if subreddits:
                search_query = f"{keyword}"
                for sub in subreddits:
                    subreddit = self.reddit.subreddit(sub)
                    posts = subreddit.search(search_query, time_filter=time_filter, limit=limit)
                    
                    for post in posts:
                        try:
                            result = {
                                'post_id': post.id,
                                'subreddit': sub,
                                'title': post.title,
                                'selftext': post.selftext,
                                'author': str(post.author) if post.author else '[deleted]',
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'url': post.url,
                                'keyword': keyword,
                                'collected_at': datetime.now().isoformat()
                            }
                            results.append(result)
                        except:
                            continue
            else:
                # Search all Reddit
                posts = self.reddit.subreddit('all').search(keyword, time_filter=time_filter, limit=limit)
                
                for post in posts:
                    try:
                        result = {
                            'post_id': post.id,
                            'subreddit': post.subreddit.display_name,
                            'title': post.title,
                            'selftext': post.selftext,
                            'author': str(post.author) if post.author else '[deleted]',
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'url': post.url,
                            'keyword': keyword,
                            'collected_at': datetime.now().isoformat()
                        }
                        results.append(result)
                    except:
                        continue
            
            print(f"✅ Found {len(results)} posts matching '{keyword}'")
            
        except Exception as e:
            print(f"❌ Error searching for '{keyword}': {e}")
        
        return results
    
    def collect_multiple_subreddits(self, subreddits, limit_per_sub=50):
        """Collect posts from multiple subreddits"""
        all_posts = []
        
        for subreddit in subreddits:
            posts = self.collect_subreddit_posts(subreddit, limit=limit_per_sub)
            all_posts.extend(posts)
            time.sleep(2)  # Rate limiting
        
        return all_posts
    
    def filter_product_mentions(self, posts, product_keywords):
        """Filter posts mentioning specific products"""
        relevant_posts = []
        
        for post in posts:
            title = post.get('title', '').lower()
            selftext = post.get('selftext', '').lower()
            combined_text = f"{title} {selftext}"
            
            if any(keyword.lower() in combined_text for keyword in product_keywords):
                post['matched_keywords'] = [kw for kw in product_keywords if kw.lower() in combined_text]
                relevant_posts.append(post)
        
        return relevant_posts
    
    def aggregate_metrics(self, posts):
        """Aggregate engagement metrics"""
        if not posts:
            return {}
        
        df = pd.DataFrame(posts)
        
        metrics = {
            'total_posts': len(posts),
            'total_score': df['score'].sum(),
            'avg_score': df['score'].mean(),
            'total_comments': df['num_comments'].sum(),
            'avg_comments': df['num_comments'].mean(),
            'avg_upvote_ratio': df['upvote_ratio'].mean() if 'upvote_ratio' in df.columns else None,
            'top_posts': df.nlargest(5, 'score')[['title', 'score', 'num_comments']].to_dict('records')
        }
        
        return metrics
    
    def save_to_database(self, posts, db_path='database.db'):
        """Save posts to database"""
        if not posts:
            print("⚠️ No posts to save")
            return
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS reddit_posts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      post_id TEXT UNIQUE,
                      subreddit TEXT,
                      title TEXT,
                      selftext TEXT,
                      author TEXT,
                      score INTEGER,
                      upvote_ratio REAL,
                      num_comments INTEGER,
                      created_utc TEXT,
                      url TEXT,
                      is_self INTEGER,
                      link_flair_text TEXT,
                      collected_at TEXT)''')
        
        inserted = 0
        for post in posts:
            try:
                c.execute('''INSERT OR IGNORE INTO reddit_posts 
                            (post_id, subreddit, title, selftext, author, score, 
                             upvote_ratio, num_comments, created_utc, url, is_self, 
                             link_flair_text, collected_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (post.get('post_id'),
                          post.get('subreddit'),
                          post.get('title'),
                          post.get('selftext', ''),
                          post.get('author'),
                          post.get('score', 0),
                          post.get('upvote_ratio', 0),
                          post.get('num_comments', 0),
                          post.get('created_utc'),
                          post.get('url', ''),
                          1 if post.get('is_self', False) else 0,
                          post.get('link_flair_text', ''),
                          post.get('collected_at')))
                inserted += 1
            except Exception as e:
                print(f"⚠️ Error inserting post: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {inserted} posts to database")
        return inserted
    
    def export_to_csv(self, posts, filename='reddit_data.csv'):
        """Export to CSV"""
        if not posts:
            print("⚠️ No posts to export")
            return
        
        df = pd.DataFrame(posts)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(posts)} posts to {filename}")

# Example usage
def run_reddit_collection():
    """Main function to run Reddit data collection"""
    
    # Initialize collector with your credentials
    collector = RedditCollector(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='DemandIQ/1.0'
    )
    
    # Define relevant subreddits
    subreddits = [
        'technology',
        'gadgets',
        'tech',
        'AndroidQuestions',
        'headphones',
        'BuyItForLife',
        'ProductPorn',
        'shutupandtakemymoney'
    ]
    
    # Define product keywords
    product_keywords = [
        'earbuds',
        'smart watch',
        'smartwatch',
        'wireless',
        'bluetooth',
        'portable charger',
        'phone case',
        'laptop stand'
    ]
    
    print("🚀 Starting Reddit data collection...")
    
    # Method 1: Collect from specific subreddits
    all_posts = collector.collect_multiple_subreddits(subreddits, limit_per_sub=100)
    
    # Method 2: Search for specific keywords
    for keyword in product_keywords[:3]:  # Limit to avoid rate limits
        search_results = collector.search_keyword(keyword, subreddits=subreddits, limit=50)
        all_posts.extend(search_results)
        time.sleep(2)
    
    # Filter for product mentions
    relevant_posts = collector.filter_product_mentions(all_posts, product_keywords)
    
    # Get metrics
    metrics = collector.aggregate_metrics(relevant_posts)
    print(f"\n📊 Collection Metrics:")
    print(f"Total Posts: {metrics.get('total_posts', 0)}")
    print(f"Total Score: {metrics.get('total_score', 0)}")
    print(f"Avg Comments: {metrics.get('avg_comments', 0):.2f}")
    
    # Save to database
    collector.save_to_database(relevant_posts)
    
    # Export to CSV
    collector.export_to_csv(relevant_posts, 'reddit_product_mentions.csv')
    
    return relevant_posts, metrics

if __name__ == "__main__":
    posts, metrics = run_reddit_collection()
    print("\n✅ Reddit data collection completed!")