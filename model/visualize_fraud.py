"""
OPAM - Fraud Visualization Module
Creates charts for fraud detection results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudVisualizer:
    """Visualize fraud detection results"""
    
    def __init__(self, output_dir='../charts'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_fraud_charts(self, df):
        """Generate all fraud visualizations"""
        
        print("\n" + "="*80)
        print("  🎨 GENERATING FRAUD VISUALIZATIONS")
        print("="*80 + "\n")
        
        # 1. Fraud Score Distribution
        print("1️⃣  Creating Fraud Score Distribution...")
        self.plot_fraud_score_distribution(df)
        
        # 2. Risk Level Analysis
        print("2️⃣  Creating Risk Level Analysis...")
        self.plot_risk_levels(df)
        
        # 3. Fraud Patterns Timeline
        print("3️⃣  Creating Fraud Timeline...")
        self.plot_fraud_timeline(df)
        
        print("\n✅ All fraud visualizations created!")
        print(f"📂 Charts saved in: {self.output_dir}/\n")
    
    def plot_fraud_score_distribution(self, df):
        """Plot fraud score distribution"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Score histogram
        ax1.hist(df['fraud_score'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax1.axvline(df['fraud_score'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["fraud_score"].mean():.1f}')
        ax1.axvline(df['fraud_score'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {df["fraud_score"].median():.1f}')
        ax1.set_xlabel('Fraud Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Fraud Score Distribution', fontsize=12, fontweight='bold', pad=20)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Score by category
        category_scores = df.groupby('category')['fraud_score'].mean().sort_values(ascending=False).head(10)
        colors = sns.color_palette('Reds_r', len(category_scores))
        
        category_scores.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_xlabel('Average Fraud Score', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Category', fontsize=11, fontweight='bold')
        ax2.set_title('Average Fraud Score by Category', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(category_scores):
            ax2.text(v, i, f' {v:.1f}', va='center', fontsize=9, fontweight='bold')
        
        # 3. Score vs Amount scatter
        sample_df = df.sample(min(10000, len(df)), random_state=42)
        scatter = ax3.scatter(sample_df['amount'], sample_df['fraud_score'], 
                            c=sample_df['fraud_score'], cmap='Reds', alpha=0.6, s=20)
        ax3.set_xlabel('Transaction Amount (₹)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Fraud Score', fontsize=11, fontweight='bold')
        ax3.set_title('Fraud Score vs Transaction Amount', fontsize=12, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Fraud Score')
        
        # 4. Cumulative fraud risk
        sorted_scores = np.sort(df['fraud_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
        
        ax4.plot(sorted_scores, cumulative, linewidth=2, color='#FF6B6B')
        ax4.fill_between(sorted_scores, cumulative, alpha=0.3, color='#FF6B6B')
        ax4.axvline(60, color='orange', linestyle='--', linewidth=2, label='High Risk Threshold')
        ax4.axvline(80, color='red', linestyle='--', linewidth=2, label='Critical Risk Threshold')
        ax4.set_xlabel('Fraud Score', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative Fraud Risk Distribution', fontsize=12, fontweight='bold', pad=20)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/14_fraud_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_risk_levels(self, df):
        """Plot risk level analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Risk level pie chart
        risk_counts = df['risk_level'].value_counts()
        colors_risk = ['#4ECDC4', '#FFE66D', '#FF6B6B', '#8B0000']
        
        wedges, texts, autotexts = ax1.pie(risk_counts, labels=risk_counts.index, 
                                            autopct='%1.1f%%', startangle=90, 
                                            colors=colors_risk, explode=[0.05]*len(risk_counts))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Transaction Distribution by Risk Level', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # 2. Amount by risk level
        risk_amounts = df.groupby('risk_level')['amount'].sum().reindex(
            ['Low', 'Medium', 'High', 'Critical'], fill_value=0
        )
        
        bars = ax2.bar(range(len(risk_amounts)), risk_amounts, color=colors_risk, alpha=0.8)
        ax2.set_xticks(range(len(risk_amounts)))
        ax2.set_xticklabels(risk_amounts.index, fontsize=10)
        ax2.set_ylabel('Total Amount (₹)', fontsize=11, fontweight='bold')
        ax2.set_title('Total Transaction Amount by Risk Level', 
                     fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, amount in zip(bars, risk_amounts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height, 
                    f'₹{amount:,.0f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # 3. Risk level by category
        risk_by_category = pd.crosstab(df['category'], df['risk_level'])
        top_categories = df['category'].value_counts().head(8).index
        risk_by_category_top = risk_by_category.loc[top_categories]
        
        risk_by_category_top.plot(kind='barh', stacked=True, ax=ax3, 
                                  color=colors_risk[:len(risk_by_category_top.columns)])
        ax3.set_xlabel('Number of Transactions', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Category', fontsize=11, fontweight='bold')
        ax3.set_title('Risk Levels by Category (Top 8)', 
                     fontsize=12, fontweight='bold', pad=20)
        ax3.legend(title='Risk Level', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Statistics table
        stats_data = []
        for level in ['Low', 'Medium', 'High', 'Critical']:
            level_df = df[df['risk_level'] == level]
            if len(level_df) > 0:
                stats_data.append([
                    level,
                    f'{len(level_df):,}',
                    f'₹{level_df["amount"].mean():.2f}',
                    f'₹{level_df["amount"].sum():,.0f}',
                    f'{level_df["fraud_score"].mean():.1f}'
                ])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=stats_data,
                         colLabels=['Risk Level', 'Count', 'Avg Amount', 'Total Amount', 'Avg Score'],
                         cellLoc='center', loc='center',
                         colColours=['#E8E8E8']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title('Risk Level Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/15_fraud_risk_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fraud_timeline(self, df):
        """Plot fraud patterns over time"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. High-risk transactions over time
        df['month'] = df['date'].dt.to_period('M')
        high_risk_by_month = df[df['fraud_score'] >= 60].groupby('month').size()
        
        ax1.plot(high_risk_by_month.index.astype(str), high_risk_by_month.values, 
                marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        ax1.fill_between(range(len(high_risk_by_month)), high_risk_by_month.values, 
                        alpha=0.3, color='#FF6B6B')
        ax1.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax1.set_ylabel('High-Risk Transactions', fontsize=11, fontweight='bold')
        ax1.set_title('High-Risk Transactions Over Time', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Average fraud score by month
        avg_score_by_month = df.groupby('month')['fraud_score'].mean()
        
        ax2.plot(avg_score_by_month.index.astype(str), avg_score_by_month.values,
                marker='s', linewidth=2, markersize=8, color='#FFA500')
        ax2.axhline(df['fraud_score'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Overall Avg: {df["fraud_score"].mean():.1f}')
        ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Fraud Score', fontsize=11, fontweight='bold')
        ax2.set_title('Average Fraud Score Trend', fontsize=12, fontweight='bold', pad=20)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Fraud score by hour of day
        hourly_fraud = df.groupby(df['date'].dt.hour)['fraud_score'].mean()
        
        bars = ax3.bar(hourly_fraud.index, hourly_fraud.values, color='#4ECDC4', alpha=0.8)
        
        # Highlight risky hours
        for hour in [2, 3, 4, 5]:
            if hour in hourly_fraud.index:
                bars[hour].set_color('#FF6B6B')
        
        ax3.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Fraud Score', fontsize=11, fontweight='bold')
        ax3.set_title('Fraud Risk by Hour of Day', fontsize=12, fontweight='bold', pad=20)
        ax3.set_xticks(range(24))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add risk zone annotation
        ax3.axvspan(2, 5, alpha=0.2, color='red', label='High Risk Period')
        ax3.legend(fontsize=10)
        
        # 4. Day of week analysis
        df['day_name'] = df['date'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_fraud = df.groupby('day_name')['fraud_score'].mean().reindex(days_order)
        
        colors_day = ['#4ECDC4']*5 + ['#FF6B6B']*2  # Weekdays vs weekends
        ax4.bar(range(len(daily_fraud)), daily_fraud.values, color=colors_day, alpha=0.8)
        ax4.set_xticks(range(len(daily_fraud)))
        ax4.set_xticklabels(days_order, rotation=45, ha='right')
        ax4.set_ylabel('Average Fraud Score', fontsize=11, fontweight='bold')
        ax4.set_title('Fraud Risk by Day of Week', fontsize=12, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(daily_fraud.values):
            ax4.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/16_fraud_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main visualization generation"""
    print("\n" + "="*80)
    print("  🎨 FRAUD VISUALIZATION MODULE")
    print("="*80)
    
    # Load fraud scores
    print("\n📁 Loading fraud detection results...")
    
    import os
    if not os.path.exists('../results/fraud_scores.csv'):
        print("\n❌ ERROR: Fraud scores not found!")
        print("   Please run fraud_detector.py first:")
        print("   python3 fraud_detector.py")
        return
    
    df = pd.read_csv('../results/fraud_scores.csv')
    print(f"✓ Loaded {len(df):,} transactions with fraud scores")
    
    # Create visualizer
    visualizer = FraudVisualizer(output_dir='../charts')
    
    # Generate charts
    visualizer.create_all_fraud_charts(df)
    
    print("\n" + "="*80)
    print("  ✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📊 Generated fraud visualization charts:")
    print("   • Chart 14: Fraud Score Distribution")
    print("   • Chart 15: Risk Level Analysis")
    print("   • Chart 16: Fraud Timeline & Patterns")
    print(f"\n📂 Find them in: ../charts/\n")

if __name__ == "__main__":
    main()
