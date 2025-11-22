"""
Download and Prepare Large-Scale Real Dataset
Uses Kaggle Credit Card Transactions Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def download_kaggle_dataset():
    """
    Download dataset using Kaggle API
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get API credentials from kaggle.com/settings
    3. Place kaggle.json in ~/.kaggle/
    """
    print("\n" + "="*80)
    print("  📥 DOWNLOADING LARGE-SCALE DATASET FROM KAGGLE")
    print("="*80 + "\n")
    
    try:
        import kaggle
        print("✓ Kaggle API installed")
        
        # Download dataset
        print("\n📦 Downloading Credit Card Transactions Dataset...")
        print("   (This may take 2-5 minutes depending on your internet)")
        
        kaggle.api.dataset_download_files(
            'priyamchoksi/credit-card-transactions-dataset',
            path='../data/',
            unzip=True
        )
        
        print("✓ Dataset downloaded successfully!")
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed")
        print("\nTo install:")
        print("  pip install kaggle")
        print("\nThen get your API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\YourName\\.kaggle\\ (Windows)")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset")
        return False

def prepare_dataset(filename):
    """Prepare and standardize the dataset"""
    print("\n" + "="*80)
    print("  🔧 PREPARING DATASET")
    print("="*80 + "\n")
    
    # Try to find the downloaded file
    possible_files = [
        '../data/credit_card_transactions.csv',
        '../data/transactions.csv',
        '../data/Credit Card Transactions.csv',
    ]
    
    filepath = None
    for file in possible_files:
        if os.path.exists(file):
            filepath = file
            break
    
    if not filepath:
        # List files in data directory
        print("📂 Files in data directory:")
        try:
            files = os.listdir('../data/')
            for f in files:
                if f.endswith('.csv'):
                    print(f"  - {f}")
                    filepath = f'../data/{f}'
                    break
        except:
            pass
    
    if not filepath:
        print("❌ Could not find dataset file")
        print("\n💡 Please manually place the CSV file in the data/ folder")
        print("   and name it 'transactions.csv'")
        return None
    
    print(f"✓ Found dataset: {filepath}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df):,} transactions")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Standardize column names to match our model
    column_mapping = {
        'Date': 'date',
        'Transaction Date': 'date',
        'trans_date': 'date',
        'Amount': 'amount',
        'Transaction Amount': 'amount',
        'amt': 'amount',
        'Category': 'category',
        'Merchant': 'merchant',
        'merchant_name': 'merchant',
        'Description': 'description',
        'Payment Method': 'payment_method',
        'payment_type': 'payment_method',
    }
    
    # Rename columns
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Ensure required columns exist
    required_columns = ['date', 'amount', 'category']
    
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️  Warning: Missing column '{col}'")
    
    # Add missing columns with defaults
    if 'merchant' not in df.columns:
        df['merchant'] = 'Unknown Merchant'
    
    if 'description' not in df.columns:
        df['description'] = df['category'] + ' Purchase'
    
    if 'payment_method' not in df.columns:
        df['payment_method'] = 'Credit Card'
    
    if 'is_recurring' not in df.columns:
        df['is_recurring'] = 0
    
    # Convert date to datetime
    print("\n🗓️  Converting dates...")
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        print("⚠️  Date conversion issues, trying alternative formats...")
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
                break
            except:
                continue
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Convert amount to numeric
    print("💰 Converting amounts...")
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    
    # Filter to positive amounts only (expenses)
    df = df[df['amount'] > 0]
    
    # Sort by date
    df = df.sort_values('date')
    
    # Keep only needed columns
    final_columns = ['date', 'amount', 'category', 'merchant', 'description', 'payment_method', 'is_recurring']
    df = df[[col for col in final_columns if col in df.columns]]
    
    # Save prepared dataset
    output_file = '../data/transactions.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset prepared and saved!")
    print(f"   File: {output_file}")
    print(f"   Transactions: {len(df):,}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total Amount: ₹{df['amount'].sum():,.2f}")
    print(f"   Categories: {df['category'].nunique()}")
    
    # Show sample
    print("\n📋 Sample transactions:")
    print(df.head(10).to_string(index=False))
    
    # Show category breakdown
    print("\n📊 Category Distribution:")
    category_counts = df['category'].value_counts().head(10)
    for cat, count in category_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {cat:30s} {count:>8,} ({pct:>5.1f}%)")
    
    return df

def show_manual_instructions():
    """Show instructions for manual download"""
    print("\n" + "="*80)
    print("  📥 MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80 + "\n")
    
    print("If automatic download doesn't work, follow these steps:")
    print("\n1️⃣  Go to Kaggle:")
    print("   https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset")
    print("\n2️⃣  Click 'Download' button")
    print("   (You may need to create a free Kaggle account)")
    print("\n3️⃣  Extract the ZIP file")
    print("\n4️⃣  Copy the CSV file to:")
    print("   opam_project/data/transactions.csv")
    print("\n5️⃣  Run this script again:")
    print("   python prepare_large_dataset.py")
    print("\n" + "="*80)

def main():
    print("\n" + "="*80)
    print("  🚀 OPAM - Large Dataset Preparation")
    print("="*80)
    
    print("\nThis script will:")
    print("  1. Download 100,000+ real credit card transactions from Kaggle")
    print("  2. Prepare and standardize the data")
    print("  3. Save it for your ML model")
    print("\nThis will make your project MUCH more impressive! 🎉")
    
    input("\nPress Enter to continue...")
    
    # Try automatic download
    success = download_kaggle_dataset()
    
    if not success:
        show_manual_instructions()
        print("\n⏸️  Waiting for manual download...")
        input("   Press Enter after you've placed the CSV in data/ folder...")
    
    # Prepare the dataset
    df = prepare_dataset('transactions.csv')
    
    if df is not None:
        print("\n" + "="*80)
        print("  ✅ SUCCESS! LARGE DATASET READY!")
        print("="*80)
        print("\nNow you can run your ML model with real data:")
        print("  cd ../src")
        print("  python run_all_systems.py")
        print("\nYour model will now train on 100,000+ REAL transactions! 🚀")
    else:
        print("\n❌ Setup incomplete. Please follow manual instructions above.")

if __name__ == "__main__":
    main()
