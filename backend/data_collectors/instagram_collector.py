import instaloader
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import time
import json
import os

class InstagramCollector:
    """
    Collect Instagram data using Instaloader API
    Features: posts, hashtags, engagement metrics, captions
    """
    
    def __init__(self, username=None, password=None):
        self.loader = instaloader.Instaloader()
        self.username = username
        self.password = password
        self.logged_in = False
        
        # Configure instaloader
        self.loader.download_pictures = False
        self.loader.download_videos = False
        self.loader.download_video_thumbnails = False
        self.loader.save_metadata = False
        self.loader.compress_json = False
        
    def login(self):
        """Login to Instagram (optional but recommended for higher rate limits)"""
        if self.username and self.password:
            try:
                self.loader.login(self.username, self.password)
                self.logged_in = True
                print("✅ Logged in to Instagram")
                return True
            except Exception as e:
                print(f"⚠️ Login failed: {e}")
                print("Continuing without login (limited rate)")
                return False
        else:
            print("ℹ️ No credentials provided. Using public access (limited rate)")
            return False
    
    def collect_hashtag_posts(self, hashtag, max_posts=100, days_back=7):
        """Collect posts for a specific hashtag"""
        print(f"🔍 Collecting posts for #{hashtag}...")
        
        posts_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            posts = instaloader.Hashtag.from_name(self.loader.context, hashtag).get_posts()
            
            count = 0
            for post in posts:
                # Check date limit
                if post.date < cutoff_date:
                    break
                
                if count >= max_posts:
                    break
                
                try:
                    # Extract post data
                    post_data = {
                        'post_id': post.shortcode,
                        'hashtag': hashtag,
                        'caption': post.caption if post.caption else '',
                        'likes': post.likes,
                        'comments': post.comments,
                        'engagement': post.likes + post.comments,
                        'date': post.date.strftime('%Y-%m-%d'),
                        'is_video': post.is_video,
                        'owner': post.owner_username,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    posts_data.append(post_data)
                    count += 1
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"⚠️ Error processing post: {e}")
                    continue
            
            print(f"✅ Collected {len(posts_data)} posts for #{hashtag}")
            
        except Exception as e:
            print(f"❌ Error collecting hashtag {hashtag}: {e}")
        
        return posts_data
    
    def collect_profile_posts(self, username, max_posts=50):
        """Collect recent posts from a profile"""
        print(f"🔍 Collecting posts from @{username}...")
        
        posts_data = []
        
        try:
            profile = instaloader.Profile.from_username(self.loader.context, username)
            posts = profile.get_posts()
            
            count = 0
            for post in posts:
                if count >= max_posts:
                    break
                
                try:
                    post_data = {
                        'post_id': post.shortcode,
                        'username': username,
                        'caption': post.caption if post.caption else '',
                        'likes': post.likes,
                        'comments': post.comments,
                        'engagement': post.likes + post.comments,
                        'date': post.date.strftime('%Y-%m-%d'),
                        'is_video': post.is_video,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    posts_data.append(post_data)
                    count += 1
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"⚠️ Error processing post: {e}")
                    continue
            
            print(f"✅ Collected {len(posts_data)} posts from @{username}")
            
        except Exception as e:
            print(f"❌ Error collecting profile {username}: {e}")
        
        return posts_data
    
    def collect_multiple_hashtags(self, hashtags, max_posts_per_tag=100, days_back=7):
        """Collect posts for multiple hashtags"""
        all_posts = []
        
        for hashtag in hashtags:
            posts = self.collect_hashtag_posts(hashtag, max_posts_per_tag, days_back)
            all_posts.extend(posts)
            
            # Rate limiting between hashtags
            time.sleep(5)
        
        return all_posts
    
    def extract_product_mentions(self, posts, product_keywords):
        """Filter posts that mention specific products"""
        relevant_posts = []
        
        for post in posts:
            caption = post.get('caption', '').lower()
            
            # Check if any product keyword is mentioned
            if any(keyword.lower() in caption for keyword in product_keywords):
                post['matched_keywords'] = [kw for kw in product_keywords if kw.lower() in caption]
                relevant_posts.append(post)
        
        return relevant_posts
    
    def aggregate_metrics(self, posts):
        """Aggregate engagement metrics from posts"""
        if not posts:
            return {}
        
        df = pd.DataFrame(posts)
        
        metrics = {
            'total_posts': len(posts),
            'total_likes': df['likes'].sum(),
            'total_comments': df['comments'].sum(),
            'total_engagement': df['engagement'].sum(),
            'avg_likes': df['likes'].mean(),
            'avg_comments': df['comments'].mean(),
            'avg_engagement': df['engagement'].mean(),
            'top_posts': df.nlargest(5, 'engagement')[['post_id', 'caption', 'engagement']].to_dict('records')
        }
        
        return metrics
    
    def save_to_database(self, posts, db_path='database.db'):
        """Save collected posts to database"""
        if not posts:
            print("⚠️ No posts to save")
            return
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS instagram_posts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      post_id TEXT UNIQUE,
                      hashtag TEXT,
                      username TEXT,
                      caption TEXT,
                      likes INTEGER,
                      comments INTEGER,
                      engagement INTEGER,
                      date TEXT,
                      is_video INTEGER,
                      owner TEXT,
                      collected_at TEXT)''')
        
        # Insert posts
        inserted = 0
        for post in posts:
            try:
                c.execute('''INSERT OR IGNORE INTO instagram_posts 
                            (post_id, hashtag, username, caption, likes, comments, 
                             engagement, date, is_video, owner, collected_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (post.get('post_id'),
                          post.get('hashtag', ''),
                          post.get('username', ''),
                          post.get('caption', ''),
                          post.get('likes', 0),
                          post.get('comments', 0),
                          post.get('engagement', 0),
                          post.get('date'),
                          1 if post.get('is_video', False) else 0,
                          post.get('owner', ''),
                          post.get('collected_at')))
                inserted += 1
            except Exception as e:
                print(f"⚠️ Error inserting post: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {inserted} posts to database")
        return inserted
    
    def export_to_csv(self, posts, filename='instagram_data.csv'):
        """Export collected data to CSV"""
        if not posts:
            print("⚠️ No posts to export")
            return
        
        df = pd.DataFrame(posts)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(posts)} posts to {filename}")
    
    def get_trending_hashtags(self, posts, top_n=10):
        """Extract most mentioned hashtags from captions"""
        import re
        from collections import Counter
        
        all_hashtags = []
        
        for post in posts:
            caption = post.get('caption', '')
            hashtags = re.findall(r'#(\w+)', caption)
            all_hashtags.extend(hashtags)
        
        counter = Counter(all_hashtags)
        return counter.most_common(top_n)

# Example usage and configuration
def run_instagram_collection():
    """Main function to run Instagram data collection"""
    
    # Initialize collector
    collector = InstagramCollector()
    
    # Optional: Login for higher rate limits
    # collector = InstagramCollector(username='your_username', password='your_password')
    # collector.login()
    
    # Define product-related hashtags to monitor
    hashtags = [
        'techgadgets',
        'wirelessearbuds',
        'smartwatch',
        'techdeals',
        'gadgetlover',
        'techreview',
        'newtech',
        'electronicsgadgets'
    ]
    
    # Define product keywords for filtering
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
    
    # Collect data
    print("🚀 Starting Instagram data collection...")
    all_posts = collector.collect_multiple_hashtags(
        hashtags,
        max_posts_per_tag=50,
        days_back=7
    )
    
    # Filter for product mentions
    relevant_posts = collector.extract_product_mentions(all_posts, product_keywords)
    
    # Get metrics
    metrics = collector.aggregate_metrics(relevant_posts)
    print(f"\n📊 Collection Metrics:")
    print(f"Total Posts: {metrics.get('total_posts', 0)}")
    print(f"Total Engagement: {metrics.get('total_engagement', 0)}")
    print(f"Avg Engagement: {metrics.get('avg_engagement', 0):.2f}")
    
    # Get trending hashtags
    trending = collector.get_trending_hashtags(relevant_posts, top_n=10)
    print(f"\n🔥 Trending Hashtags:")
    for tag, count in trending:
        print(f"  #{tag}: {count} mentions")
    
    # Save to database
    collector.save_to_database(relevant_posts)
    
    # Export to CSV
    collector.export_to_csv(relevant_posts, 'instagram_product_mentions.csv')
    
    return relevant_posts, metrics

if __name__ == "__main__":
    posts, metrics = run_instagram_collection()
    print("\n✅ Instagram data collection completed!")