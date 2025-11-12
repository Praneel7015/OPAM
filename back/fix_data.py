"""
Data Column Fixer
Checks your CSV and fixes column names to work with the model
"""

import pandas as pd
import sys

def check_and_fix_columns():
    """Check CSV columns and fix them"""
    
    print("\n" + "="*80)
    print("  🔍 CHECKING YOUR DATA FILE")
    print("="*80 + "\n")
    
    # Load the CSV
    csv_path = '../data/transactions.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ File loaded: {len(df):,} rows")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return None
    
    # Show current columns
    print(f"\n📋 CURRENT COLUMNS IN YOUR FILE:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Show first few rows
    print(f"\n👀 FIRST 3 ROWS:")
    print(df.head(3).to_string())
    
    # Define possible column mappings
    column_mappings = {
        'date': ['date', 'Date', 'transaction_date', 'trans_date', 'Transaction Date', 'DATE', 'TransDate'],
        'amount': ['amount', 'Amount', 'amt', 'transaction_amount', 'Transaction Amount', 'AMT', 'TransAmount'],
        'category': ['category', 'Category', 'type', 'Type', 'merchant_category', 'Category Name', 'CATEGORY'],
        'merchant': ['merchant', 'Merchant', 'merchant_name', 'Merchant Name', 'store', 'Store'],
        'description': ['description', 'Description', 'memo', 'Memo', 'details', 'Details'],
    }
    
    print(f"\n🔧 FIXING COLUMN NAMES...")
    
    # Create mapping
    rename_map = {}
    for target_col, possible_names in column_mappings.items():
        for col in df.columns:
            if col in possible_names:
                if col != target_col:
                    rename_map[col] = target_col
                break
    
    # Rename columns
    if rename_map:
        print(f"\n✓ Renaming columns:")
        for old, new in rename_map.items():
            print(f"   {old} → {new}")
        df = df.rename(columns=rename_map)
    
    # Add missing required columns
    required = {
        'merchant': 'Unknown',
        'description': 'Purchase',
        'payment_method': 'Credit Card',
        'is_recurring': 0
    }
    
    for col, default_value in required.items():
        if col not in df.columns:
            print(f"   Adding missing column: {col}")
            df[col] = default_value
    
    # Verify we have critical columns
    critical_columns = ['date', 'amount', 'category']
    missing = [col for col in critical_columns if col not in df.columns]
    
    if missing:
        print(f"\n❌ ERROR: Still missing critical columns: {missing}")
        print(f"\n💡 MANUAL MAPPING NEEDED:")
        print(f"   Your columns: {list(df.columns)}")
        print(f"   Which column contains:")
        
        manual_map = {}
        for critical in missing:
            print(f"\n   {critical}?")
            for i, col in enumerate(df.columns, 1):
                print(f"      {i}. {col}")
            choice = input(f"   Enter number (or 0 to skip): ").strip()
            try:
                choice = int(choice)
                if 1 <= choice <= len(df.columns):
                    manual_map[df.columns[choice-1]] = critical
            except:
                pass
        
        if manual_map:
            print(f"\n✓ Applying manual mapping:")
            for old, new in manual_map.items():
                print(f"   {old} → {new}")
            df = df.rename(columns=manual_map)
    
    # Final check
    print(f"\n✅ FINAL COLUMNS:")
    for col in df.columns:
        print(f"   ✓ {col}")
    
    # Keep only needed columns
    keep_columns = ['date', 'amount', 'category', 'merchant', 'description', 'payment_method', 'is_recurring']
    keep_columns = [col for col in keep_columns if col in df.columns]
    df = df[keep_columns]
    
    # Convert data types
    print(f"\n🔄 Converting data types...")
    
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"   ✓ date → datetime")
    except:
        print(f"   ⚠️  Could not convert date")
    
    try:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        print(f"   ✓ amount → numeric")
    except:
        print(f"   ⚠️  Could not convert amount")
    
    # Remove invalid rows
    initial_count = len(df)
    df = df.dropna(subset=['date', 'amount'])
    df = df[df['amount'] > 0]
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"   ℹ️  Removed {removed:,} invalid rows")
    
    # Save fixed file
    output_path = '../data/transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 SAVED FIXED FILE:")
    print(f"   Location: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Show summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total amount: ₹{df['amount'].sum():,.2f}")
    print(f"   Categories: {df['category'].nunique()}")
    
    print(f"\n✅ DONE! Your file is now ready!")
    print(f"\nNow run:")
    print(f"   cd ../ml_models")
    print(f"   python expense_predictor.py")
    
    return df

if __name__ == "__main__":
    check_and_fix_columns()
