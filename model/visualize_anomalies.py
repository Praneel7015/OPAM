"""
OPAM - Anomaly Visualization Module
Creates charts for anomaly detection results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnomalyVisualizer:
    """Visualize anomaly detection results"""
    
    def __init__(self, output_dir='../charts'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_anomaly_charts(self, df, all_anomalies, consensus):
        """Generate all anomaly visualizations"""
        
        print("\n" + "="*80)
        print("  🎨 GENERATING ANOMALY VISUALIZATIONS")
        print("="*80 + "\n")
        
        # 1. Anomaly Distribution
        print("1️⃣  Creating Anomaly Distribution Chart...")
        self.plot_anomaly_distribution(df, all_anomalies)
        
        # 2. Amount vs Normal
        print("2️⃣  Creating Amount Comparison...")
        self.plot_amount_comparison(df, all_anomalies)
        
        # 3. Time-based anomalies
        print("3️⃣  Creating Time-Based Anomaly Heatmap...")
        self.plot_time_anomalies(df, all_anomalies)
        
        # 4. Category anomalies
        print("4️⃣  Creating Category Anomaly Chart...")
        self.plot_category_anomalies(df, all_anomalies)
        
        # 5. Consensus anomalies
        if len(consensus) > 0:
            print("5️⃣  Creating Consensus Anomaly Chart...")
            self.plot_consensus_anomalies(consensus)
        
        print("\n✅ All anomaly visualizations created!")
        print(f"📂 Charts saved in: {self.output_dir}/\n")
    
    def plot_anomaly_distribution(self, df, all_anomalies):
        """Plot distribution of anomalies by method"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count by method
        methods = []
        counts = []
        for method, anomalies in all_anomalies.items():
            if isinstance(anomalies, pd.DataFrame):
                methods.append(method.replace(' (', '\n(').replace(' ', '\n', 1))
                counts.append(len(anomalies))
        
        # Bar chart
        colors = sns.color_palette('Set2', len(methods))
        bars = ax1.barh(methods, counts, color=colors)
        ax1.set_xlabel('Number of Anomalies', fontsize=12, fontweight='bold')
        ax1.set_title('Anomalies Detected by Method', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax1.text(count, bar.get_y() + bar.get_height()/2, 
                    f' {count:,}', va='center', fontsize=10, fontweight='bold')
        
        # Percentage of total
        percentages = [(c / len(df)) * 100 for c in counts]
        ax2.barh(methods, percentages, color=colors)
        ax2.set_xlabel('Percentage of Total (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Anomaly Rate by Method', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (method, pct) in enumerate(zip(methods, percentages)):
            ax2.text(pct, i, f' {pct:.2f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_anomaly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_amount_comparison(self, df, all_anomalies):
        """Compare anomalous vs normal transaction amounts"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get all anomaly indices
        all_anomaly_indices = set()
        for anomalies in all_anomalies.values():
            if isinstance(anomalies, pd.DataFrame) and len(anomalies) > 0:
                all_anomaly_indices.update(anomalies.index)
        
        if len(all_anomaly_indices) == 0:
            return
        
        normal_df = df[~df.index.isin(all_anomaly_indices)]
        anomaly_df = df[df.index.isin(all_anomaly_indices)]
        
        # 1. Box plot comparison
        data_to_plot = [normal_df['amount'], anomaly_df['amount']]
        box = ax1.boxplot(data_to_plot, labels=['Normal', 'Anomalous'],
                         patch_artist=True, showfliers=False)
        box['boxes'][0].set_facecolor('#4ECDC4')
        box['boxes'][1].set_facecolor('#FF6B6B')
        ax1.set_ylabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax1.set_title('Amount Distribution: Normal vs Anomalous', 
                     fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Histogram comparison
        ax2.hist(normal_df['amount'], bins=50, alpha=0.6, label='Normal', color='#4ECDC4')
        ax2.hist(anomaly_df['amount'], bins=50, alpha=0.6, label='Anomalous', color='#FF6B6B')
        ax2.set_xlabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Amount Distribution Histogram', fontsize=12, fontweight='bold', pad=20)
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, anomaly_df['amount'].quantile(0.95))
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistics comparison
        stats_data = {
            'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Max'],
            'Normal': [
                f'{len(normal_df):,}',
                f'₹{normal_df["amount"].mean():.2f}',
                f'₹{normal_df["amount"].median():.2f}',
                f'₹{normal_df["amount"].std():.2f}',
                f'₹{normal_df["amount"].max():.2f}'
            ],
            'Anomalous': [
                f'{len(anomaly_df):,}',
                f'₹{anomaly_df["amount"].mean():.2f}',
                f'₹{anomaly_df["amount"].median():.2f}',
                f'₹{anomaly_df["amount"].std():.2f}',
                f'₹{anomaly_df["amount"].max():.2f}'
            ]
        }
        
        ax3.axis('tight')
        ax3.axis('off')
        table = ax3.table(cellText=[[stats_data['Metric'][i], 
                                     stats_data['Normal'][i], 
                                     stats_data['Anomalous'][i]] 
                                    for i in range(len(stats_data['Metric']))],
                         colLabels=['Metric', 'Normal', 'Anomalous'],
                         cellLoc='center', loc='center',
                         colColours=['#E8E8E8']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax3.set_title('Statistical Comparison', fontsize=12, fontweight='bold', pad=20)
        
        # 4. Cumulative distribution
        normal_sorted = np.sort(normal_df['amount'].sample(min(10000, len(normal_df))))
        anomaly_sorted = np.sort(anomaly_df['amount'])
        
        ax4.plot(normal_sorted, np.linspace(0, 100, len(normal_sorted)), 
                label='Normal', linewidth=2, color='#4ECDC4')
        ax4.plot(anomaly_sorted, np.linspace(0, 100, len(anomaly_sorted)), 
                label='Anomalous', linewidth=2, color='#FF6B6B')
        ax4.set_xlabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative Distribution', fontsize=12, fontweight='bold', pad=20)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_amount_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_anomalies(self, df, all_anomalies):
        """Plot time-based anomaly patterns"""
        
        time_anomalies = all_anomalies.get('Time-Based', pd.DataFrame())
        if len(time_anomalies) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        
        time_anomalies['date'] = pd.to_datetime(time_anomalies['date'])
        time_anomalies['hour'] = time_anomalies['date'].dt.hour
        time_anomalies['day_of_week'] = time_anomalies['date'].dt.day_name()
        
        # 1. Hourly distribution
        hours = range(24)
        normal_hourly = df.groupby('hour').size()
        anomaly_hourly = time_anomalies.groupby('hour').size()
        
        x = np.arange(24)
        width = 0.35
        
        ax1.bar(x - width/2, [normal_hourly.get(h, 0) for h in hours], 
               width, label='Normal', alpha=0.7, color='#4ECDC4')
        ax1.bar(x + width/2, [anomaly_hourly.get(h, 0) for h in hours], 
               width, label='Anomalous', alpha=0.7, color='#FF6B6B')
        
        ax1.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
        ax1.set_title('Transaction Distribution by Hour', fontsize=12, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(hours)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Highlight suspicious hours
        suspicious_hours = [2, 3, 4, 5]
        for hour in suspicious_hours:
            ax1.axvspan(hour-0.5, hour+0.5, alpha=0.2, color='red')
        
        # 2. Day of week distribution
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        normal_daily = df.groupby('day_of_week').size()
        anomaly_daily = time_anomalies.groupby('day_of_week').size()
        
        x = np.arange(len(days_order))
        
        ax2.bar(x - width/2, [normal_daily.get(d, 0) for d in days_order], 
               width, label='Normal', alpha=0.7, color='#4ECDC4')
        ax2.bar(x + width/2, [anomaly_daily.get(d, 0) for d in days_order], 
               width, label='Anomalous', alpha=0.7, color='#FF6B6B')
        
        ax2.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
        ax2.set_title('Transaction Distribution by Day', fontsize=12, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(days_order, rotation=45, ha='right')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/11_time_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_anomalies(self, df, all_anomalies):
        """Plot category-specific anomalies"""
        
        # Get all anomaly indices
        all_anomaly_indices = set()
        for anomalies in all_anomalies.values():
            if isinstance(anomalies, pd.DataFrame) and len(anomalies) > 0:
                all_anomaly_indices.update(anomalies.index)
        
        if len(all_anomaly_indices) == 0:
            return
        
        anomaly_df = df[df.index.isin(all_anomaly_indices)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Anomalies by category
        category_counts = anomaly_df['category'].value_counts()
        colors = sns.color_palette('Set3', len(category_counts))
        
        category_counts.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('Number of Anomalies', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Category', fontsize=11, fontweight='bold')
        ax1.set_title('Anomalies by Category', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(category_counts):
            ax1.text(v, i, f' {v:,}', va='center', fontsize=9, fontweight='bold')
        
        # 2. Anomaly rate by category
        total_by_category = df['category'].value_counts()
        anomaly_rate = (category_counts / total_by_category * 100).sort_values(ascending=False)
        
        anomaly_rate.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_xlabel('Anomaly Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Category', fontsize=11, fontweight='bold')
        ax2.set_title('Anomaly Rate by Category', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, v in enumerate(anomaly_rate):
            ax2.text(v, i, f' {v:.2f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/12_category_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_consensus_anomalies(self, consensus):
        """Plot consensus (high-confidence) anomalies"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top 10 consensus anomalies
        top10 = consensus.nlargest(10, 'amount')
        colors = sns.color_palette('Reds_r', len(top10))
        
        y_pos = range(len(top10))
        amounts = top10['amount'].values
        
        bars = ax1.barh(y_pos, amounts, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"#{i+1}" for i in range(len(top10))], fontsize=10)
        ax1.set_xlabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax1.set_title('Top 10 High-Confidence Anomalies', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, amount in zip(bars, amounts):
            ax1.text(amount, bar.get_y() + bar.get_height()/2, 
                    f' ₹{amount:,.0f}', va='center', fontsize=9, fontweight='bold')
        
        # 2. Category distribution
        category_dist = consensus['category'].value_counts()
        colors_pie = sns.color_palette('Set2', len(category_dist))
        
        ax2.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%',
               startangle=90, colors=colors_pie)
        ax2.set_title('Consensus Anomalies by Category', fontsize=12, fontweight='bold', pad=20)
        
        # 3. Timeline
        consensus['date'] = pd.to_datetime(consensus['date'])
        timeline = consensus.groupby(consensus['date'].dt.to_period('M')).size()
        
        ax3.plot(timeline.index.astype(str), timeline.values, 
                marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        ax3.fill_between(range(len(timeline)), timeline.values, alpha=0.3, color='#FF6B6B')
        ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Anomalies', fontsize=11, fontweight='bold')
        ax3.set_title('Consensus Anomalies Over Time', fontsize=12, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Amount distribution
        ax4.hist(consensus['amount'], bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax4.axvline(consensus['amount'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ₹{consensus["amount"].mean():,.0f}')
        ax4.axvline(consensus['amount'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: ₹{consensus["amount"].median():,.0f}')
        ax4.set_xlabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Consensus Anomaly Amount Distribution', fontsize=12, fontweight='bold', pad=20)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/13_consensus_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main visualization generation"""
    print("\n" + "="*80)
    print("  🎨 ANOMALY VISUALIZATION MODULE")
    print("="*80)
    
    # Load data
    print("\n📁 Loading data...")
    df = pd.read_csv('../data/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Load anomaly results
    print("\n📁 Loading anomaly detection results...")
    
    import os
    results_dir = '../results'
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"\n❌ ERROR: Results directory not found!")
        print(f"   Expected: {results_dir}")
        print(f"\n💡 SOLUTION:")
        print(f"   1. Run anomaly detector first:")
        print(f"      python3 anomaly_detector_simple.py")
        print(f"   2. Then run visualization:")
        print(f"      python3 visualize_anomalies.py")
        return
    
    # Load anomaly files
    all_anomalies = {}
    file_mapping = {
        'anomalies_statistical.csv': 'Statistical',
        'anomalies_isolation_forest.csv': 'Isolation Forest',
        'anomalies_high_value.csv': 'High Value',
        'all_anomalies.csv': 'All Anomalies'
    }
    
    for filename, method_name in file_mapping.items():
        filepath = f'{results_dir}/{filename}'
        if os.path.exists(filepath):
            all_anomalies[method_name] = pd.read_csv(filepath)
            print(f"✓ Loaded {method_name}: {len(all_anomalies[method_name]):,} anomalies")
    
    if not all_anomalies:
        print("\n⚠️  No anomaly files found!")
        print("   Please run: python3 anomaly_detector_simple.py")
        return
    
    # Use all_anomalies as consensus if it exists
    if 'All Anomalies' in all_anomalies:
        consensus = all_anomalies['All Anomalies'].copy()
        print(f"✓ Using all anomalies as consensus: {len(consensus):,}")
    else:
        consensus = pd.DataFrame()
        print("ℹ️  No consensus file found")
    
    # Create visualizer
    visualizer = AnomalyVisualizer(output_dir='../charts')
    
    # Generate charts
    visualizer.create_all_anomaly_charts(df, all_anomalies, consensus)
    
    print("\n" + "="*80)
    print("  ✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📊 Generated anomaly visualization charts!")
    print(f"📂 Find them in: ../charts/\n")

if __name__ == "__main__":
    main()
