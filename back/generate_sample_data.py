import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

CATEGORIES = {
    "Food & Dining": {"min": 50, "max": 800, "avg": 250},
    "Transportation": {"min": 30, "max": 500, "avg": 150},
    "Shopping": {"min": 200, "max": 3000, "avg": 800},
    "Entertainment": {"min": 100, "max": 1500, "avg": 400},
    "Groceries": {"min": 500, "max": 4000, "avg": 2000},
    "Healthcare": {"min": 200, "max": 2000, "avg": 600},
    "Utilities": {"min": 500, "max": 3000, "avg": 1500},
    "Education": {"min": 1000, "max": 10000, "avg": 3000},
    "Personal Care": {"min": 200, "max": 1500, "avg": 500},
    "Other": {"min": 100, "max": 2000, "avg": 500}
}

MERCHANTS = {
    "Food & Dining": ["Swiggy", "Zomato", "McDonald's", "Starbucks", "Domino's"],
    "Transportation": ["Uber", "Ola", "Rapido", "IRCTC", "Petrol Pump"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa"],
    "Entertainment": ["BookMyShow", "Netflix", "Amazon Prime", "Spotify"],
    "Groceries": ["BigBasket", "Blinkit", "Zepto", "DMart"],
    "Healthcare": ["Apollo", "PharmEasy", "1mg", "Hospital"],
    "Utilities": ["Electricity", "Water", "Internet", "Mobile"],
    "Education": ["Udemy", "Coursera", "Books", "Tuition"],
    "Personal Care": ["Salon", "Gym", "Spa", "Grooming"],
    "Other": ["Gift", "Donation", "Repair", "Misc"]
}

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash"]

def generate_transactions(num_months=12, transactions_per_month=30):
    """Generate realistic transaction data"""
    
    transactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_months * 30)
    
    current_date = start_date
    
    while current_date <= end_date:
        month_transactions = random.randint(
            transactions_per_month - 10, 
            transactions_per_month + 10
        )
        
        for _ in range(month_transactions):
            category = random.choice(list(CATEGORIES.keys()))
            cat_info = CATEGORIES[category]
            
            if random.random() < 0.1:
                amount = random.uniform(cat_info["max"] * 0.5, cat_info["max"] * 1.5)
            else:
                amount = random.gauss(cat_info["avg"], cat_info["avg"] * 0.3)
                amount = max(cat_info["min"], min(cat_info["max"], amount))
            
            merchant = random.choice(MERCHANTS[category])
            days_in_month = 30
            random_day = random.randint(0, days_in_month - 1)
            transaction_date = current_date + timedelta(days=random_day)
            
            if transaction_date > end_date:
                continue
            
            payment_method = random.choice(PAYMENT_METHODS)
            is_recurring = 1 if random.random() < 0.05 else 0
            
            transaction = {
                "amount": round(amount, 2),
                "category": category,
                "description": f"Purchase at {merchant}",
                "merchant": merchant,
                "date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
                "payment_method": payment_method,
                "is_recurring": is_recurring
            }
            
            transactions.append(transaction)
        
        current_date += timedelta(days=30)
    
    transactions.sort(key=lambda x: x["date"])
    
    return transactions

def save_transactions_to_csv(transactions, filename="transactions.csv"):
    """Save to CSV"""
    df = pd.DataFrame(transactions)
    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(transactions)} transactions to {filename}")
    return df

if __name__ == "__main__":
    print("🎲 Generating sample data...")
    print("="*60)
    
    transactions = generate_transactions(num_months=12, transactions_per_month=30)
    df = save_transactions_to_csv(transactions, "../data/transactions.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby('month').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    
    print("\n📊 Monthly Summary:")
    print(monthly)
    
    category = df.groupby('category').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    
    print("\n📈 Category Summary:")
    print(category)
    
    print("\n"+"="*60)
    print("✅ Data generation complete!")
    print(f"Total: {len(transactions)} transactions")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")
