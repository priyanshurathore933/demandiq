from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Sales data table
    c.execute('''CREATE TABLE IF NOT EXISTS sales_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT NOT NULL,
                  product TEXT NOT NULL,
                  quantity INTEGER NOT NULL,
                  price REAL NOT NULL,
                  revenue REAL NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Social media data table
    c.execute('''CREATE TABLE IF NOT EXISTS social_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  platform TEXT NOT NULL,
                  product TEXT NOT NULL,
                  mentions INTEGER DEFAULT 0,
                  sentiment_score REAL DEFAULT 0,
                  hashtags TEXT,
                  engagement INTEGER DEFAULT 0,
                  date TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Forecast results table
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  product TEXT NOT NULL,
                  forecast_date TEXT NOT NULL,
                  predicted_quantity INTEGER,
                  lower_bound INTEGER,
                  upper_bound INTEGER,
                  confidence REAL,
                  model_version TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'DemandIQ API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    conn = get_db_connection()
    
    # Get total sales
    sales_query = conn.execute('SELECT SUM(revenue) as total FROM sales_data').fetchone()
    total_sales = sales_query['total'] if sales_query['total'] else 0
    
    # Get forecast accuracy (mock for now)
    accuracy = 94.2
    
    # Get inventory alerts count
    alerts_count = 3
    
    # Get stock efficiency
    efficiency = 87
    
    conn.close()
    
    return jsonify({
        'forecast_accuracy': accuracy,
        'predicted_sales': f"${total_sales * 1.125:.2f}M",
        'stock_efficiency': efficiency,
        'active_alerts': alerts_count,
        'last_updated': datetime.now().isoformat()
    })

@app.route('/api/forecast/data', methods=['GET'])
def get_forecast_data():
    product = request.args.get('product', 'all')
    months = int(request.args.get('months', 6))
    
    conn = get_db_connection()
    
    # Get historical data
    historical = conn.execute('''
        SELECT date, SUM(quantity) as actual
        FROM sales_data
        GROUP BY date
        ORDER BY date DESC
        LIMIT ?
    ''', (months,)).fetchall()
    
    # Get forecasts
    forecasts = conn.execute('''
        SELECT forecast_date, predicted_quantity, lower_bound, upper_bound
        FROM forecasts
        WHERE product = ? OR ? = 'all'
        ORDER BY forecast_date
        LIMIT ?
    ''', (product, product, months)).fetchall()
    
    conn.close()
    
    # Combine data
    result = []
    for row in historical:
        result.append({
            'date': row['date'],
            'actual': row['actual'],
            'predicted': None,
            'lower': None,
            'upper': None
        })
    
    for row in forecasts:
        result.append({
            'date': row['forecast_date'],
            'actual': None,
            'predicted': row['predicted_quantity'],
            'lower': row['lower_bound'],
            'upper': row['upper_bound']
        })
    
    return jsonify(result)

@app.route('/api/social/sentiment', methods=['GET'])
def get_sentiment_data():
    conn = get_db_connection()
    
    # Get sentiment by platform
    sentiment = conn.execute('''
        SELECT platform,
               AVG(CASE WHEN sentiment_score > 0.3 THEN 1 ELSE 0 END) * 100 as positive,
               AVG(CASE WHEN sentiment_score BETWEEN -0.3 AND 0.3 THEN 1 ELSE 0 END) * 100 as neutral,
               AVG(CASE WHEN sentiment_score < -0.3 THEN 1 ELSE 0 END) * 100 as negative
        FROM social_data
        WHERE date >= date('now', '-7 days')
        GROUP BY platform
    ''').fetchall()
    
    conn.close()
    
    result = []
    for row in sentiment:
        result.append({
            'platform': row['platform'],
            'positive': round(row['positive'], 1),
            'neutral': round(row['neutral'], 1),
            'negative': round(row['negative'], 1)
        })
    
    # Add combined data
    if result:
        combined_positive = np.mean([r['positive'] for r in result])
        combined_neutral = np.mean([r['neutral'] for r in result])
        combined_negative = np.mean([r['negative'] for r in result])
        
        result.append({
            'platform': 'Combined',
            'positive': round(combined_positive, 1),
            'neutral': round(combined_neutral, 1),
            'negative': round(combined_negative, 1)
        })
    
    return jsonify(result)

@app.route('/api/social/trending', methods=['GET'])
def get_trending_products():
    conn = get_db_connection()
    
    trending = conn.execute('''
        SELECT product,
               SUM(mentions) as total_mentions,
               AVG(sentiment_score) as avg_sentiment,
               SUM(engagement) as total_engagement
        FROM social_data
        WHERE date >= date('now', '-7 days')
        GROUP BY product
        ORDER BY total_mentions DESC, avg_sentiment DESC
        LIMIT 10
    ''').fetchall()
    
    conn.close()
    
    result = []
    for idx, row in enumerate(trending):
        score = min(100, int((row['total_mentions'] / 100) * (row['avg_sentiment'] + 1) * 50))
        result.append({
            'name': row['product'],
            'score': score,
            'change': f"+{np.random.randint(5, 20)}%",
            'sentiment': 'positive' if row['avg_sentiment'] > 0.3 else 'neutral'
        })
    
    return jsonify(result)

@app.route('/api/inventory/alerts', methods=['GET'])
def get_inventory_alerts():
    # Mock data for now - replace with actual inventory logic
    alerts = [
        {
            'product': 'Wireless Earbuds',
            'status': 'low',
            'current': 45,
            'recommended': 150,
            'action': 'Restock'
        },
        {
            'product': 'Smart Watch',
            'status': 'optimal',
            'current': 120,
            'recommended': 115,
            'action': 'Monitor'
        },
        {
            'product': 'Laptop Stand',
            'status': 'overstock',
            'current': 200,
            'recommended': 80,
            'action': 'Reduce'
        }
    ]
    
    return jsonify(alerts)

@app.route('/api/upload/sales', methods=['POST'])
def upload_sales_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Validate columns
            required_columns = ['date', 'product', 'quantity', 'price']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'CSV must contain columns: {required_columns}'}), 400
            
            # Calculate revenue
            df['revenue'] = df['quantity'] * df['price']
            
            # Insert into database
            conn = get_db_connection()
            inserted = 0
            
            for _, row in df.iterrows():
                conn.execute('''
                    INSERT INTO sales_data (date, product, quantity, price, revenue)
                    VALUES (?, ?, ?, ?, ?)
                ''', (row['date'], row['product'], row['quantity'], row['price'], row['revenue']))
                inserted += 1
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'message': 'Data uploaded successfully',
                'rows_inserted': inserted,
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/data/collect', methods=['POST'])
def trigger_data_collection():
    data = request.json
    source = data.get('source', 'all')
    
    # This will trigger background data collection
    # For now, return success
    
    return jsonify({
        'message': f'Data collection triggered for {source}',
        'status': 'processing',
        'estimated_time': '5-10 minutes'
    })

@app.route('/api/model/train', methods=['POST'])
def train_model():
    data = request.json
    product = data.get('product', 'all')
    
    # This will trigger model training
    # For now, return success
    
    return jsonify({
        'message': 'Model training initiated',
        'product': product,
        'status': 'training',
        'estimated_time': '10-15 minutes'
    })

@app.route('/api/social/stats', methods=['GET'])
def get_social_stats():
    conn = get_db_connection()
    
    # Instagram stats
    instagram = conn.execute('''
        SELECT SUM(mentions) as mentions, AVG(sentiment_score) as sentiment
        FROM social_data
        WHERE platform = 'Instagram' AND date >= date('now', '-7 days')
    ''').fetchone()
    
    # Reddit stats
    reddit = conn.execute('''
        SELECT SUM(mentions) as mentions, AVG(sentiment_score) as sentiment
        FROM social_data
        WHERE platform = 'Reddit' AND date >= date('now', '-7 days')
    ''').fetchone()
    
    conn.close()
    
    return jsonify({
        'instagram': {
            'mentions': instagram['mentions'] if instagram['mentions'] else 12450,
            'sentiment': round((instagram['sentiment'] + 1) * 50, 1) if instagram['sentiment'] else 68,
            'change': '+18%'
        },
        'reddit': {
            'mentions': reddit['mentions'] if reddit['mentions'] else 8320,
            'sentiment': round((reddit['sentiment'] + 1) * 50, 1) if reddit['sentiment'] else 52,
            'change': '+12%'
        }
    })

if __name__ == '__main__':
    print("🚀 DemandIQ Backend Starting...")
    print("📊 API running on http://localhost:5000")
    print("🔗 Health check: http://localhost:5000/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)