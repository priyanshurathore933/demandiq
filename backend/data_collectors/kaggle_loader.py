"""
Kaggle Dataset Loader for DemandIQ
Recommended datasets:
1. Retail Sales Dataset
2. E-commerce Sales Data
3. Product Sales Time Series
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

class KaggleDatasetLoader:
    """Load and process Kaggle datasets for demand forecasting"""
    
    def __init__(self, db_path='database.db'):
        self.db_path = db_path
    
    def load_csv(self, filepath, date_column='date', product_column='product', 
                 quantity_column='quantity', price_column='price'):
        """
        Load CSV file with flexible column mapping
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            product_column: Name of product column
            quantity_column: Name of quantity/sales column
            price_column: Name of price column
        """
        print(f"📂 Loading dataset from: {filepath}")
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Display available columns
            print(f"Available columns: {list(df.columns)}")
            
            # Standardize column names
            column_mapping = {
                date_column: 'date',
                product_column: 'product',
                quantity_column: 'quantity',
                price_column: 'price'
            }
            
            # Keep only required columns
            required_cols = [date_column, product_column, quantity_column, price_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
                print("Please specify correct column names or preprocess the dataset.")
                return None
            
            df = df[required_cols].copy()
            df.rename(columns=column_mapping, inplace=True)
            
            # Process data
            df = self._clean_data(df)
            
            print(f"✅ Dataset processed successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def _clean_data(self, df):
        """Clean and validate dataset"""
        print("🧹 Cleaning dataset...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        before = len(df)
        df = df.dropna(subset=['date'])
        after = len(df)
        if before > after:
            print(f"⚠️ Removed {before - after} rows with invalid dates")
        
        # Convert numeric columns
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Remove negative values
        df = df[(df['quantity'] >= 0) & (df['price'] >= 0)]
        
        # Calculate revenue
        df['revenue'] = df['quantity'] * df['price']
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Format date as string
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        print(f"✅ Cleaned dataset: {len(df)} rows")
        return df
    
    def save_to_database(self, df):
        """Save dataset to SQLite database"""
        if df is None or df.empty:
            print("⚠️ No data to save")
            return 0
        
        print(f"💾 Saving {len(df)} records to database...")
        
        conn = sqlite3.connect(self.db_path)
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT INTO sales_data (date, product, quantity, price, revenue)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    row['date'],
                    row['product'],
                    int(row['quantity']),
                    float(row['price']),
                    float(row['revenue'])
                ))
                inserted += 1
            except Exception as e:
                # Skip duplicates or errors
                continue
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {inserted} records to database")
        return inserted
    
    def generate_sample_data(self, products=None, days=365, save_to_file=True):
        """
        Generate realistic sample sales data for testing
        
        Args:
            products: List of product names
            days: Number of days of data
            save_to_file: Save to CSV file
        """
        print(f"🎲 Generating sample data for {days} days...")
        
        if products is None:
            products = [
                'Wireless Earbuds',
                'Smart Watch',
                'Portable Charger',
                'Phone Case',
                'Laptop Stand'
            ]
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        
        for product in products:
            # Base parameters for each product
            base_quantity = np.random.randint(50, 200)
            base_price = np.random.uniform(15, 100)
            
            for date in dates:
                # Add seasonal patterns
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Add weekly patterns (weekend spike)
                weekend_factor = 1.2 if date.dayofweek >= 5 else 1.0
                
                # Add random noise
                noise = np.random.normal(1, 0.15)
                
                # Calculate quantity
                quantity = int(base_quantity * seasonal_factor * weekend_factor * noise)
                quantity = max(0, quantity)  # No negative sales
                
                # Price variation
                price = base_price * np.random.uniform(0.95, 1.05)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product': product,
                    'quantity': quantity,
                    'price': round(price, 2),
                    'revenue': round(quantity * price, 2)
                })
        
        df = pd.DataFrame(data)
        
        print(f"✅ Generated {len(df)} records")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Products: {', '.join(products)}")
        
        if save_to_file:
            filename = 'sample_sales_data.csv'
            df.to_csv(filename, index=False)
            print(f"💾 Saved to {filename}")
        
        return df
    
    def analyze_dataset(self, df):
        """Generate summary statistics of the dataset"""
        if df is None or df.empty:
            print("⚠️ No data to analyze")
            return
        
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS")
        print("="*60)
        
        print(f"\n📅 Date Range:")
        print(f"   From: {df['date'].min()}")
        print(f"   To: {df['date'].max()}")
        print(f"   Days: {df['date'].nunique()}")
        
        print(f"\n📦 Products:")
        print(f"   Total products: {df['product'].nunique()}")
        print(f"   Product list: {', '.join(df['product'].unique()[:5])}")
        
        print(f"\n💰 Sales Statistics:")
        print(f"   Total quantity sold: {df['quantity'].sum():,.0f}")
        print(f"   Total revenue: ${df['revenue'].sum():,.2f}")
        print(f"   Average daily sales: {df['quantity'].mean():.0f} units")
        print(f"   Average price: ${df['price'].mean():.2f}")
        
        print(f"\n📈 By Product:")
        product_summary = df.groupby('product').agg({
            'quantity': 'sum',
            'revenue': 'sum'
        }).sort_values('revenue', ascending=False)
        
        for product, row in product_summary.head().iterrows():
            print(f"   {product}: {row['quantity']:,.0f} units, ${row['revenue']:,.2f}")
        
        print("\n" + "="*60)
    
    def load_popular_kaggle_datasets(self):
        """
        Instructions for downloading popular Kaggle datasets
        """
        print("\n📚 POPULAR KAGGLE DATASETS FOR DEMAND FORECASTING:")
        print("="*60)
        
        datasets = [
            {
                'name': 'Online Retail Dataset',
                'url': 'https://www.kaggle.com/datasets/vijayuv/onlineretail',
                'description': 'Transactional data from UK online retailer',
                'columns': ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']
            },
            {
                'name': 'Superstore Sales Dataset',
                'url': 'https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting',
                'description': 'Sales data with product categories',
                'columns': ['Order Date', 'Product Name', 'Quantity', 'Sales', 'Profit']
            },
            {
                'name': 'E-commerce Dataset',
                'url': 'https://www.kaggle.com/datasets/carrie1/ecommerce-data',
                'description': 'E-commerce transaction data',
                'columns': ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']
            }
        ]
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   URL: {dataset['url']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Columns: {', '.join(dataset['columns'])}")
        
        print("\n" + "="*60)
        print("\nTo download:")
        print("1. Create Kaggle account: https://www.kaggle.com")
        print("2. Go to dataset URL and click 'Download'")
        print("3. Extract CSV file")
        print("4. Use this loader to import into DemandIQ")

# Example usage
def main():
    """Main function demonstrating usage"""
    
    loader = KaggleDatasetLoader()
    
    # Option 1: Generate sample data for testing
    print("\n🎲 Generating sample data...")
    df = loader.generate_sample_data(
        products=['Wireless Earbuds', 'Smart Watch', 'Portable Charger'],
        days=365,
        save_to_file=True
    )
    
    # Analyze dataset
    loader.analyze_dataset(df)
    
    # Save to database
    loader.save_to_database(df)
    
    # Option 2: Load from Kaggle dataset (commented out - customize for your dataset)
    # df = loader.load_csv(
    #     filepath='path/to/your/kaggle_dataset.csv',
    #     date_column='Order Date',
    #     product_column='Product Name',
    #     quantity_column='Quantity',
    #     price_column='Unit Price'
    # )
    
    # Option 3: Show popular datasets
    # loader.load_popular_kaggle_datasets()
    
    print("\n✅ Dataset loading complete!")
    print("📊 Data is ready for forecasting")

if __name__ == "__main__":
    main()