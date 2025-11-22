"""
OPAM - Clustering Visualization Module
Creates charts for user clustering results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ClusterVisualizer:
    """Visualize clustering results"""
    
    def __init__(self, output_dir='../charts'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_cluster_charts(self, user_profiles, cluster_summary):
        """Generate all clustering visualizations"""
        
        print("\n" + "="*80)
        print("  🎨 GENERATING CLUSTERING VISUALIZATIONS")
        print("="*80 + "\n")
        
        # 1. Cluster Distribution
        print("1️⃣  Creating Cluster Distribution Chart...")
        self.plot_cluster_distribution(user_profiles, cluster_summary)
        
        # 2. Cluster Characteristics
        print("2️⃣  Creating Cluster Characteristics Chart...")
        self.plot_cluster_characteristics(cluster_summary)
        
        print("\n✅ All clustering visualizations created!")
        print(f"📂 Charts saved in: {self.output_dir}/\n")
    
    def plot_cluster_distribution(self, user_profiles, cluster_summary):
        """Plot cluster distribution and 2D visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cluster size pie chart
        cluster_sizes = cluster_summary.sort_values('cluster_id')
        colors = sns.color_palette('Set3', len(cluster_sizes))
        
        # Prepare labels with names
        labels = [f"Cluster {row['cluster_id']}\n{row['cluster_name'].replace('💎 ', '').replace('🏆 ', '').replace('⚡ ', '').replace('🌈 ', '').replace('🏠 ', '')}" 
                 for _, row in cluster_sizes.iterrows()]
        
        wedges, texts, autotexts = ax1.pie(cluster_sizes['size'], labels=labels,
                                            autopct='%1.1f%%', startangle=90,
                                            colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        for text in texts:
            text.set_fontsize(8)
        ax1.set_title('User Distribution by Cluster', fontsize=12, fontweight='bold', pad=20)
        
        # 2. PCA 2D visualization
        feature_cols = ['avg_amount', 'total_spent', 'transaction_count', 
                       'txn_per_day', 'category_diversity', 'weekend_ratio']
        X = user_profiles[feature_cols].fillna(0)
        
        # Perform PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Plot clusters
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=user_profiles['cluster'], 
                            cmap='Set3', alpha=0.6, s=50)
        
        # Add cluster centers
        for cluster_id in user_profiles['cluster'].unique():
            cluster_center = X_pca[user_profiles['cluster'] == cluster_id].mean(axis=0)
            ax2.scatter(cluster_center[0], cluster_center[1], 
                       marker='X', s=300, c='red', edgecolors='black', linewidths=2,
                       label=f'Cluster {cluster_id}' if cluster_id == 0 else '')
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                      fontsize=11, fontweight='bold')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                      fontsize=11, fontweight='bold')
        ax2.set_title('User Clusters (PCA Projection)', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cluster spending comparison
        cluster_order = cluster_sizes.sort_values('total_spent', ascending=False)['cluster_id'].values
        total_spending = [cluster_sizes[cluster_sizes['cluster_id'] == cid]['total_spent'].values[0] 
                         for cid in cluster_order]
        cluster_labels = [f"C{cid}" for cid in cluster_order]
        
        bars = ax3.bar(cluster_labels, total_spending, color=colors, alpha=0.8)
        ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Total Spending (₹)', fontsize=11, fontweight='bold')
        ax3.set_title('Total Spending by Cluster', fontsize=12, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, amount in zip(bars, total_spending):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'₹{amount/1e6:.1f}M' if amount >= 1e6 else f'₹{amount/1e3:.0f}K',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Average transaction amount by cluster
        avg_amounts = [cluster_sizes[cluster_sizes['cluster_id'] == cid]['avg_amount'].values[0] 
                      for cid in range(len(cluster_sizes))]
        cluster_names_short = [row['cluster_name'].split('(')[0].replace('💎 ', '').replace('🏆 ', '').replace('⚡ ', '').replace('🌈 ', '').replace('🏠 ', '').strip() 
                              for _, row in cluster_sizes.sort_values('cluster_id').iterrows()]
        
        bars = ax4.barh(cluster_names_short, avg_amounts, color=colors, alpha=0.8)
        ax4.set_xlabel('Average Transaction Amount (₹)', fontsize=11, fontweight='bold')
        ax4.set_title('Average Transaction by Cluster', fontsize=12, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, amount in zip(bars, avg_amounts):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f' ₹{amount:.2f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/17_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_characteristics(self, cluster_summary):
        """Plot detailed cluster characteristics"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_summary = cluster_summary.sort_values('cluster_id')
        colors = sns.color_palette('Set3', len(cluster_summary))
        
        cluster_labels = [f"C{row['cluster_id']}" for _, row in cluster_summary.iterrows()]
        
        # 1. Transaction frequency
        txn_per_day = cluster_summary['txn_per_day'].values
        bars = ax1.bar(cluster_labels, txn_per_day, color=colors, alpha=0.8)
        ax1.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Transactions per Day', fontsize=11, fontweight='bold')
        ax1.set_title('Transaction Frequency by Cluster', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, txn_per_day):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Category diversity
        diversity = cluster_summary['category_diversity'].values
        bars = ax2.bar(cluster_labels, diversity, color=colors, alpha=0.8)
        ax2.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Categories', fontsize=11, fontweight='bold')
        ax2.set_title('Category Diversity by Cluster', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, diversity):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Weekend activity ratio
        weekend_ratios = cluster_summary['weekend_ratio'].values * 100
        bars = ax3.bar(cluster_labels, weekend_ratios, color=colors, alpha=0.8)
        ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Weekend Activity (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Weekend vs Weekday Activity', fontsize=12, fontweight='bold', pad=20)
        ax3.axhline(y=28.6, color='red', linestyle='--', linewidth=2, 
                   label='Expected (2/7 days)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, weekend_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Cluster characteristics heatmap
        # Normalize features for heatmap
        features_for_heatmap = cluster_summary[['avg_amount', 'avg_transactions', 
                                                'txn_per_day', 'category_diversity', 
                                                'weekend_ratio']].copy()
        
        # Normalize each column to 0-1 scale
        for col in features_for_heatmap.columns:
            features_for_heatmap[col] = (
                (features_for_heatmap[col] - features_for_heatmap[col].min()) / 
                (features_for_heatmap[col].max() - features_for_heatmap[col].min())
            )
        
        features_for_heatmap.index = cluster_labels
        features_for_heatmap.columns = ['Avg\nAmount', 'Avg\nTransactions', 
                                       'Txn per\nDay', 'Category\nDiversity', 
                                       'Weekend\nRatio']
        
        sns.heatmap(features_for_heatmap.T, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Normalized Value'}, ax=ax4, linewidths=1)
        ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax4.set_title('Cluster Characteristics Heatmap (Normalized)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/18_cluster_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main visualization generation"""
    print("\n" + "="*80)
    print("  🎨 CLUSTERING VISUALIZATION MODULE")
    print("="*80)
    
    # Load clustering results
    print("\n📁 Loading clustering results...")
    
    import os
    if not os.path.exists('../results/user_clusters.csv'):
        print("\n❌ ERROR: Clustering results not found!")
        print("   Please run user_clusterer.py first:")
        print("   python3 user_clusterer.py")
        return
    
    user_profiles = pd.read_csv('../results/user_clusters.csv')
    cluster_summary = pd.read_csv('../results/cluster_summary.csv')
    
    print(f"✓ Loaded {len(user_profiles):,} user profiles")
    print(f"✓ Loaded {len(cluster_summary)} cluster summaries")
    
    # Create visualizer
    visualizer = ClusterVisualizer(output_dir='../charts')
    
    # Generate charts
    visualizer.create_all_cluster_charts(user_profiles, cluster_summary)
    
    print("\n" + "="*80)
    print("  ✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📊 Generated clustering visualization charts:")
    print("   • Chart 17: Cluster Distribution & PCA")
    print("   • Chart 18: Cluster Characteristics")
    print(f"\n📂 Find them in: ../charts/\n")

if __name__ == "__main__":
    main()
