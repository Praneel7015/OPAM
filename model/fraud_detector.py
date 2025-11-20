"""
OPAM - Fraud Detection System
Advanced fraud scoring and pattern detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    """Advanced fraud detection with scoring system"""
    
    def __init__(self):
        self.fraud_patterns = []
        self.risk_scores = {}
        
    def print_header(self, text, char="="):
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def detect_fraud(self, df):
        """Run comprehensive fraud detection"""
        
        self.print_header("🚨 FRAUD DETECTION SYSTEM")
        
        print(f"📊 Analyzing {len(df):,} transactions for fraud patterns...")
        print(f"💰 Total Amount: ₹{df['amount'].sum():,.2f}\n")
        
        # Reset index
        df = df.reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate fraud scores
        print("🔍 Calculating fraud risk scores...")
        df = self.calculate_fraud_scores(df)
        
        # Detect fraud patterns
        print("\n🕵️  Detecting fraud patterns...")
        patterns = self.detect_fraud_patterns(df)
        
        # Classify transactions
        print("\n📊 Classifying transaction risks...")
        risk_summary = self.classify_risk_levels(df)
        
        # Generate alerts
        print("\n🚨 Generating fraud alerts...")
        alerts = self.generate_fraud_alerts(df)
        
        # Show results
        self.show_fraud_summary(df, patterns, alerts)
        
        # Save results
        self.save_fraud_results(df, patterns, alerts)
        
        return df, patterns, alerts
    
    def calculate_fraud_scores(self, df):
        """Calculate fraud risk score (0-100) for each transaction"""
        
        df['fraud_score'] = 0.0
        
        # 1. Amount-based risk (0-30 points)
        amount_percentile = df['amount'].rank(pct=True)
        df['fraud_score'] += amount_percentile * 30
        
        # 2. Time-based risk (0-20 points)
        df['hour'] = df['date'].dt.hour
        # High risk hours: 2-5 AM
        df['fraud_score'] += df['hour'].apply(
            lambda h: 20 if 2 <= h <= 5 else 
                     10 if 0 <= h <= 1 or 23 <= h <= 23 else 0
        )
        
        # 3. Merchant fraud indicator (0-25 points)
        if 'merchant' in df.columns:
            df['fraud_score'] += df['merchant'].str.contains(
                'fraud', case=False, na=False
            ).astype(int) * 25
        
        # 4. Rapid transaction risk (0-15 points)
        df = df.sort_values('date')
        df['time_since_last'] = df.groupby('category')['date'].diff().dt.total_seconds() / 60
        df['fraud_score'] += (df['time_since_last'] < 5).fillna(False).astype(int) * 15
        
        # 5. Weekend high-value risk (0-10 points)
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        df['fraud_score'] += (
            (df['is_weekend']) & 
            (df['amount'] > df['amount'].quantile(0.90))
        ).astype(int) * 10
        
        # Normalize to 0-100
        if df['fraud_score'].max() > 0:
            df['fraud_score'] = (df['fraud_score'] / df['fraud_score'].max()) * 100
        
        print(f"   ✓ Calculated fraud scores")
        print(f"   📊 Score range: {df['fraud_score'].min():.1f} - {df['fraud_score'].max():.1f}")
        print(f"   📊 Average score: {df['fraud_score'].mean():.1f}")
        
        return df
    
    def detect_fraud_patterns(self, df):
        """Detect specific fraud patterns"""
        
        patterns = {
            'high_frequency': [],
            'velocity_fraud': [],
            'late_night': [],
            'high_value': [],
            'merchant_fraud': []
        }
        
        # 1. High-frequency pattern (multiple transactions quickly)
        df_sorted = df.sort_values('date')
        df_sorted['time_diff'] = df_sorted['date'].diff().dt.total_seconds() / 60
        high_freq = df_sorted[df_sorted['time_diff'] < 3]  # < 3 minutes
        if len(high_freq) > 0:
            patterns['high_frequency'] = high_freq.index.tolist()
            print(f"   • High-frequency: {len(high_freq):,} transactions (< 3 min apart)")
        
        # 2. Velocity fraud (spending spree)
        df['rolling_count_1h'] = df.groupby(df['date'].dt.floor('H')).cumcount()
        velocity = df[df['rolling_count_1h'] > 5]  # > 5 transactions in 1 hour
        if len(velocity) > 0:
            patterns['velocity_fraud'] = velocity.index.tolist()
            print(f"   • Velocity fraud: {len(velocity):,} transactions (>5 per hour)")
        
        # 3. Late-night transactions
        late_night = df[(df['hour'] >= 2) & (df['hour'] <= 5)]
        if len(late_night) > 0:
            patterns['late_night'] = late_night.index.tolist()
            print(f"   • Late-night: {len(late_night):,} transactions (2-5 AM)")
        
        # 4. High-value unusual
        threshold = df['amount'].quantile(0.98)
        high_value = df[df['amount'] > threshold]
        if len(high_value) > 0:
            patterns['high_value'] = high_value.index.tolist()
            print(f"   • High-value: {len(high_value):,} transactions (top 2%)")
        
        # 5. Merchant fraud indicator
        if 'merchant' in df.columns:
            merchant_fraud = df[df['merchant'].str.contains('fraud', case=False, na=False)]
            if len(merchant_fraud) > 0:
                patterns['merchant_fraud'] = merchant_fraud.index.tolist()
                print(f"   • Merchant fraud: {len(merchant_fraud):,} suspicious merchants")
        
        return patterns
    
    def classify_risk_levels(self, df):
        """Classify transactions into risk levels"""
        
        df['risk_level'] = 'Low'
        df.loc[df['fraud_score'] >= 30, 'risk_level'] = 'Medium'
        df.loc[df['fraud_score'] >= 60, 'risk_level'] = 'High'
        df.loc[df['fraud_score'] >= 80, 'risk_level'] = 'Critical'
        
        summary = df['risk_level'].value_counts()
        
        for level in ['Low', 'Medium', 'High', 'Critical']:
            count = summary.get(level, 0)
            pct = (count / len(df)) * 100
            print(f"   {level:8s} Risk: {count:>10,} ({pct:>5.2f}%)")
        
        return summary
    
    def generate_fraud_alerts(self, df):
        """Generate fraud alerts for high-risk transactions"""
        
        # Critical and High risk transactions
        alerts = df[df['risk_level'].isin(['Critical', 'High'])].copy()
        alerts = alerts.sort_values('fraud_score', ascending=False)
        
        print(f"   🚨 Generated {len(alerts):,} fraud alerts")
        
        if len(alerts) > 0:
            print(f"   💰 Total flagged amount: ₹{alerts['amount'].sum():,.2f}")
            print(f"   📊 Average fraud score: {alerts['fraud_score'].mean():.1f}")
        
        return alerts
    
    def show_fraud_summary(self, df, patterns, alerts):
        """Display comprehensive fraud summary"""
        
        self.print_header("📊 FRAUD DETECTION SUMMARY")
        
        # Overall statistics
        total_fraud_score = df['fraud_score'].sum()
        avg_fraud_score = df['fraud_score'].mean()
        high_risk_count = len(df[df['fraud_score'] >= 60])
        
        print("📈 Fraud Statistics:")
        print(f"   Total Transactions:     {len(df):>10,}")
        print(f"   High Risk (Score ≥60):  {high_risk_count:>10,} ({high_risk_count/len(df)*100:>5.2f}%)")
        print(f"   Average Fraud Score:    {avg_fraud_score:>10.2f}")
        print(f"   Fraud Alerts Generated: {len(alerts):>10,}")
        
        # Pattern summary
        print("\n🕵️  Fraud Patterns Detected:")
        total_patterns = sum(len(indices) for indices in patterns.values())
        print(f"   Total Pattern Matches:  {total_patterns:>10,}")
        
        for pattern_name, indices in patterns.items():
            if len(indices) > 0:
                pattern_display = pattern_name.replace('_', ' ').title()
                print(f"   - {pattern_display:20s} {len(indices):>8,}")
        
        # Top 10 most suspicious transactions
        self.print_header("🚨 TOP 10 MOST SUSPICIOUS TRANSACTIONS")
        
        top_suspicious = df.nlargest(10, 'fraud_score')
        
        print(f"{'Score':>6} | {'Risk':8s} | {'Amount':>12} | {'Category':20s} | {'Date'}")
        print("─" * 85)
        
        for _, row in top_suspicious.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M')
            print(f"{row['fraud_score']:>6.1f} | {row['risk_level']:8s} | "
                  f"₹{row['amount']:>10,.2f} | {row['category']:20s} | {date_str}")
        
        # Recommendations
        self.print_header("💡 FRAUD PREVENTION RECOMMENDATIONS")
        
        critical_count = len(df[df['risk_level'] == 'Critical'])
        high_count = len(df[df['risk_level'] == 'High'])
        
        print("Based on fraud analysis:\n")
        
        if critical_count > 0:
            print(f"1️⃣  CRITICAL: Immediately review {critical_count:,} critical-risk transactions")
            print(f"   • Average fraud score: {df[df['risk_level']=='Critical']['fraud_score'].mean():.1f}")
            print(f"   • Total amount: ₹{df[df['risk_level']=='Critical']['amount'].sum():,.2f}")
        
        if high_count > 0:
            print(f"\n2️⃣  HIGH PRIORITY: Review {high_count:,} high-risk transactions")
            print(f"   • Set up alerts for scores above 60")
            print(f"   • Enable 2FA for high-value transactions")
        
        # Pattern-specific recommendations
        if len(patterns['late_night']) > 0:
            print(f"\n3️⃣  TIME-BASED: Block late-night transactions (2-5 AM)")
            print(f"   • {len(patterns['late_night']):,} suspicious late-night transactions detected")
        
        if len(patterns['velocity_fraud']) > 0:
            print(f"\n4️⃣  VELOCITY: Implement transaction velocity limits")
            print(f"   • {len(patterns['velocity_fraud']):,} rapid spending patterns detected")
        
        if 'merchant_fraud' in patterns and len(patterns['merchant_fraud']) > 0:
            print(f"\n5️⃣  MERCHANT: Block suspicious merchants")
            print(f"   • {len(patterns['merchant_fraud']):,} transactions with fraud indicators")
        
        print("\n6️⃣  GENERAL:")
        print("   • Enable real-time fraud scoring")
        print("   • Set up SMS/email alerts for high-risk transactions")
        print("   • Review and update fraud rules monthly")
        print("   • Implement machine learning fraud model")
    
    def save_fraud_results(self, df, patterns, alerts):
        """Save fraud detection results"""
        
        self.print_header("💾 SAVING FRAUD RESULTS")
        
        import os
        os.makedirs('../results', exist_ok=True)
        
        # Save fraud-scored transactions
        fraud_df = df[['date', 'category', 'amount', 'fraud_score', 'risk_level']].copy()
        if 'merchant' in df.columns:
            fraud_df['merchant'] = df['merchant']
        
        fraud_df.to_csv('../results/fraud_scores.csv', index=False)
        print("✓ Fraud scores: ../results/fraud_scores.csv")
        
        # Save alerts
        if len(alerts) > 0:
            alerts_export = alerts[['date', 'category', 'amount', 'fraud_score', 'risk_level']].copy()
            if 'merchant' in alerts.columns:
                alerts_export['merchant'] = alerts['merchant']
            alerts_export.to_csv('../results/fraud_alerts.csv', index=False)
            print(f"✓ Fraud alerts ({len(alerts):,}): ../results/fraud_alerts.csv")
        
        # Save pattern details
        pattern_summary = []
        for pattern_name, indices in patterns.items():
            if len(indices) > 0:
                pattern_df = df.loc[indices, ['date', 'category', 'amount', 'fraud_score']]
                pattern_df['pattern_type'] = pattern_name
                pattern_summary.append(pattern_df)
        
        if pattern_summary:
            all_patterns = pd.concat(pattern_summary, ignore_index=True)
            all_patterns.to_csv('../results/fraud_patterns.csv', index=False)
            print(f"✓ Fraud patterns: ../results/fraud_patterns.csv")
        
        # Save risk summary
        risk_summary = df['risk_level'].value_counts().to_frame('count')
        risk_summary['percentage'] = (risk_summary['count'] / len(df)) * 100
        risk_summary.to_csv('../results/fraud_risk_summary.csv')
        print("✓ Risk summary: ../results/fraud_risk_summary.csv")
        
        print(f"\n✓ Saved 4 fraud detection files")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🚨 OPAM: Fraud Detection System")
    print("  🔒 Advanced Fraud Scoring & Pattern Detection")
    print("="*80)
    
    # Load data
    print("\n📁 Loading transaction data...")
    df = pd.read_csv('../data/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Detect fraud
    detector = FraudDetector()
    df_with_scores, patterns, alerts = detector.detect_fraud(df)
    
    # Complete
    print("\n" + "="*80)
    print("  ✅ FRAUD DETECTION COMPLETE!")
    print("="*80)
    print("\n📂 Results saved in: ../results/")
    print("\n📊 Files created:")
    print("   • fraud_scores.csv - All transactions with fraud scores")
    print("   • fraud_alerts.csv - High-risk transactions requiring review")
    print("   • fraud_patterns.csv - Detected fraud patterns")
    print("   • fraud_risk_summary.csv - Risk level distribution")
    print("\nNext steps:")
    print("  1. Review fraud alerts CSV")
    print("  2. Run: python3 visualize_fraud.py")
    print("  3. Integrate with main system")
    print("  4. Set up real-time alerts")
    print()

if __name__ == "__main__":
    main()
