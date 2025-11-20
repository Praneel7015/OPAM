"""
OPAM - Simplified Anomaly Detection (FIXED)
Works perfectly with large datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

class SimpleAnomalyDetector:
    """Simplified anomaly detector - no index issues"""
    
    def __init__(self):
        self.results = {}
        
    def print_header(self, text, char="="):
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def detect_all(self, df):
        """Run all detection methods"""
        
        self.print_header("🔍 ANOMALY DETECTION - SIMPLIFIED & FAST")
        
        print(f"📊 Dataset: {len(df):,} transactions")
        print(f"💰 Total Amount: ₹{df['amount'].sum():,.2f}\n")
        
        # Reset index to avoid issues
        df = df.reset_index(drop=True)
        
        # Method 1: Statistical
        print("1️⃣  Statistical Z-Score Detection...")
        stat_anomalies = self.detect_statistical(df)
        
        # Method 2: Isolation Forest (on sample for speed)
        print("\n2️⃣  Isolation Forest Detection...")
        iso_anomalies = self.detect_isolation_forest(df)
        
        # Method 3: Amount-based (simple & effective)
        print("\n3️⃣  High-Value Transaction Detection...")
        high_value_anomalies = self.detect_high_value(df)
        
        # Summary
        self.print_header("📊 DETECTION SUMMARY")
        
        total_anomalies = set()
        
        print(f"Statistical (Z-Score):     {len(stat_anomalies):>6,} anomalies "
              f"({len(stat_anomalies)/len(df)*100:>5.2f}%)")
        total_anomalies.update(stat_anomalies.index)
        
        print(f"Isolation Forest:          {len(iso_anomalies):>6,} anomalies "
              f"({len(iso_anomalies)/len(df)*100:>5.2f}%)")
        total_anomalies.update(iso_anomalies.index)
        
        print(f"High-Value Transactions:   {len(high_value_anomalies):>6,} anomalies "
              f"({len(high_value_anomalies)/len(df)*100:>5.2f}%)")
        total_anomalies.update(high_value_anomalies.index)
        
        print(f"\n{'─' * 80}")
        print(f"Total Unique Anomalies:    {len(total_anomalies):>6,} "
              f"({len(total_anomalies)/len(df)*100:>5.2f}%)")
        
        # Get all anomalies
        all_anomaly_df = df.loc[list(total_anomalies)].copy()
        all_anomaly_df = all_anomaly_df.sort_values('amount', ascending=False)
        
        # Show top 10
        self.print_header("🔝 TOP 10 ANOMALOUS TRANSACTIONS")
        
        print(f"{'Amount':>12} | {'Category':20s} | {'Date':12s} | {'Merchant'}")
        print("─" * 80)
        
        for idx, row in all_anomaly_df.head(10).iterrows():
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            merchant = str(row.get('merchant', 'Unknown'))[:30]
            print(f"₹{row['amount']:>10,.2f} | {row['category']:20s} | "
                  f"{date_str} | {merchant}")
        
        # Insights
        self.generate_insights(df, all_anomaly_df)
        
        # Save results
        self.save_results(stat_anomalies, iso_anomalies, 
                         high_value_anomalies, all_anomaly_df)
        
        return {
            'statistical': stat_anomalies,
            'isolation_forest': iso_anomalies,
            'high_value': high_value_anomalies,
            'all_anomalies': all_anomaly_df
        }
    
    def detect_statistical(self, df):
        """Z-score based detection"""
        z_scores = np.abs(stats.zscore(df['amount']))
        anomaly_mask = z_scores > 3
        anomalies = df[anomaly_mask].copy()
        
        print(f"   ✓ Found {len(anomalies):,} anomalies")
        if len(anomalies) > 0:
            print(f"   📊 Range: ₹{anomalies['amount'].min():.2f} - "
                  f"₹{anomalies['amount'].max():.2f}")
            print(f"   💰 Total: ₹{anomalies['amount'].sum():,.2f}")
        
        return anomalies
    
    def detect_isolation_forest(self, df, sample_size=100000):
        """Isolation Forest on sample (for speed)"""
        
        # Sample for speed
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
            print(f"   ℹ️  Using sample of {sample_size:,} transactions")
        else:
            sample_df = df
        
        # Prepare features
        features = pd.DataFrame({
            'amount': sample_df['amount'],
            'hour': pd.to_datetime(sample_df['date']).dt.hour,
            'day_of_week': pd.to_datetime(sample_df['date']).dt.dayofweek,
        })
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Train
        iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
        predictions = iso.fit_predict(X_scaled)
        
        # Get anomalies
        anomaly_indices = sample_df.index[predictions == -1]
        anomalies = df.loc[anomaly_indices].copy()
        
        print(f"   ✓ Found {len(anomalies):,} anomalies")
        if len(anomalies) > 0:
            print(f"   📊 Range: ₹{anomalies['amount'].min():.2f} - "
                  f"₹{anomalies['amount'].max():.2f}")
            print(f"   💰 Total: ₹{anomalies['amount'].sum():,.2f}")
        
        return anomalies
    
    def detect_high_value(self, df):
        """Simple high-value detection"""
        
        # Top 1% by amount
        threshold = df['amount'].quantile(0.99)
        anomalies = df[df['amount'] > threshold].copy()
        
        print(f"   ✓ Found {len(anomalies):,} anomalies (top 1%)")
        print(f"   📊 Threshold: ₹{threshold:.2f}")
        if len(anomalies) > 0:
            print(f"   💰 Total: ₹{anomalies['amount'].sum():,.2f}")
        
        return anomalies
    
    def generate_insights(self, df, anomaly_df):
        """Generate actionable insights"""
        
        self.print_header("💡 INSIGHTS & RECOMMENDATIONS")
        
        if len(anomaly_df) == 0:
            print("✓ No significant anomalies detected - spending looks normal!")
            return
        
        # Statistics
        normal_df = df[~df.index.isin(anomaly_df.index)]
        
        print("📊 Comparison:")
        print(f"   Normal Transactions:    {len(normal_df):>10,}")
        print(f"   Anomalous Transactions: {len(anomaly_df):>10,}")
        print(f"   Anomaly Rate:           {len(anomaly_df)/len(df)*100:>10.2f}%")
        
        print(f"\n💰 Amount Analysis:")
        print(f"   Normal Avg:      ₹{normal_df['amount'].mean():>10,.2f}")
        print(f"   Anomalous Avg:   ₹{anomaly_df['amount'].mean():>10,.2f}")
        print(f"   Difference:      {(anomaly_df['amount'].mean()/normal_df['amount'].mean() - 1)*100:>10.1f}%")
        
        # Category breakdown
        print(f"\n📋 Top Anomalous Categories:")
        top_cats = anomaly_df['category'].value_counts().head(5)
        for cat, count in top_cats.items():
            pct = count / len(anomaly_df) * 100
            total = anomaly_df[anomaly_df['category'] == cat]['amount'].sum()
            print(f"   {cat:20s} {count:>5,} ({pct:>4.1f}%)  ₹{total:>12,.2f}")
        
        # Recommendations
        print(f"\n✅ Recommendations:")
        avg_anomaly = anomaly_df['amount'].mean()
        print(f"   • Set alert threshold at ₹{avg_anomaly * 0.8:,.0f}")
        print(f"   • Review {len(anomaly_df):,} flagged transactions")
        print(f"   • Monitor {top_cats.index[0]} category closely")
        print(f"   • Enable real-time anomaly notifications")
    
    def save_results(self, stat, iso, high_val, all_anom):
        """Save results to CSV"""
        
        self.print_header("💾 SAVING RESULTS")
        
        import os
        os.makedirs('../results', exist_ok=True)
        
        # Save each type
        files_saved = 0
        
        if len(stat) > 0:
            stat.to_csv('../results/anomalies_statistical.csv', index=False)
            print(f"✓ Statistical anomalies: ../results/anomalies_statistical.csv")
            files_saved += 1
        
        if len(iso) > 0:
            iso.to_csv('../results/anomalies_isolation_forest.csv', index=False)
            print(f"✓ Isolation Forest: ../results/anomalies_isolation_forest.csv")
            files_saved += 1
        
        if len(high_val) > 0:
            high_val.to_csv('../results/anomalies_high_value.csv', index=False)
            print(f"✓ High-Value: ../results/anomalies_high_value.csv")
            files_saved += 1
        
        if len(all_anom) > 0:
            all_anom.to_csv('../results/all_anomalies.csv', index=False)
            print(f"✓ All anomalies: ../results/all_anomalies.csv")
            files_saved += 1
        
        print(f"\n✓ Saved {files_saved} result files")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🔍 OPAM: Simplified Anomaly Detection")
    print("  ⚡ Fast & Reliable")
    print("="*80)
    
    # Load data
    print("\n📁 Loading data...")
    df = pd.read_csv('../data/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Detect
    detector = SimpleAnomalyDetector()
    results = detector.detect_all(df)
    
    # Complete
    print("\n" + "="*80)
    print("  ✅ DETECTION COMPLETE!")
    print("="*80)
    print("\n📂 Results saved in: ../results/")
    print("\nNext steps:")
    print("  1. Review results CSV files")
    print("  2. Run: python3 visualize_anomalies.py")
    print("  3. Check charts in ../charts/")
    print()

if __name__ == "__main__":
    main()
