"""
OPAM - User Clustering Module
Groups users by spending behavior patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class UserClusterer:
    """Cluster users by spending behavior"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        
    def print_header(self, text, char="="):
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def cluster_users(self, df):
        """Perform user clustering analysis"""
        
        self.print_header("👥 USER BEHAVIOR CLUSTERING")
        
        print(f"📊 Analyzing spending patterns...")
        print(f"   Total Transactions: {len(df):,}")
        print(f"   Clustering Algorithm: K-Means")
        print(f"   Number of Clusters: {self.n_clusters}\n")
        
        # Create user profiles
        print("1️⃣  Creating user spending profiles...")
        user_profiles = self.create_user_profiles(df)
        
        # Perform clustering
        print("\n2️⃣  Running K-Means clustering...")
        user_profiles = self.perform_clustering(user_profiles)
        
        # Analyze clusters
        print("\n3️⃣  Analyzing cluster characteristics...")
        cluster_analysis = self.analyze_clusters(user_profiles, df)
        
        # Name clusters
        print("\n4️⃣  Profiling user segments...")
        cluster_names = self.name_clusters(cluster_analysis)
        
        # Show results
        self.show_cluster_summary(user_profiles, cluster_analysis, cluster_names)
        
        # Save results
        self.save_clustering_results(user_profiles, cluster_analysis, cluster_names)
        
        return user_profiles, cluster_analysis, cluster_names
    
    def create_user_profiles(self, df):
        """Create aggregate user spending profiles"""
        
        # Aggregate by user (assuming each row represents unique transactions)
        # For this dataset, we'll create synthetic user IDs based on patterns
        df['date'] = pd.to_datetime(df['date'])
        
        # Create user profiles based on date ranges (simulated users)
        # Group transactions into user-like segments
        df['user_id'] = (df.index // 5000).astype(int)  # ~5000 transactions per "user"
        
        # Calculate user-level features
        user_features = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'sum', 'count'],
            'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        user_features.columns = ['user_id', 'avg_amount', 'std_amount', 
                                 'total_spent', 'transaction_count', 'top_category']
        
        # Add time-based features
        time_features = df.groupby('user_id').agg({
            'date': ['min', 'max']
        }).reset_index()
        time_features.columns = ['user_id', 'first_transaction', 'last_transaction']
        
        user_features = user_features.merge(time_features, on='user_id')
        
        # Calculate activity duration
        user_features['days_active'] = (
            user_features['last_transaction'] - user_features['first_transaction']
        ).dt.days + 1
        
        # Transactions per day
        user_features['txn_per_day'] = (
            user_features['transaction_count'] / user_features['days_active']
        )
        
        # Add category diversity
        category_diversity = df.groupby('user_id')['category'].nunique().reset_index()
        category_diversity.columns = ['user_id', 'category_diversity']
        user_features = user_features.merge(category_diversity, on='user_id')
        
        # Weekend vs weekday ratio
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        weekend_ratio = df.groupby('user_id')['is_weekend'].mean().reset_index()
        weekend_ratio.columns = ['user_id', 'weekend_ratio']
        user_features = user_features.merge(weekend_ratio, on='user_id')
        
        print(f"   ✓ Created profiles for {len(user_features):,} users")
        print(f"   📊 Features per user: {len(user_features.columns) - 4}")  # Exclude non-feature columns
        
        return user_features
    
    def perform_clustering(self, user_profiles):
        """Perform K-Means clustering"""
        
        # Select features for clustering
        feature_cols = ['avg_amount', 'std_amount', 'total_spent', 
                       'transaction_count', 'txn_per_day', 
                       'category_diversity', 'weekend_ratio']
        
        X = user_profiles[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        user_profiles['cluster'] = self.model.fit_predict(X_scaled)
        
        # Calculate cluster quality
        inertia = self.model.inertia_
        
        print(f"   ✓ Clustering complete")
        print(f"   📊 Inertia (lower is better): {inertia:,.2f}")
        
        # Show cluster distribution
        cluster_counts = user_profiles['cluster'].value_counts().sort_index()
        print(f"\n   Cluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            pct = (count / len(user_profiles)) * 100
            print(f"   Cluster {cluster_id}: {count:>6,} users ({pct:>5.1f}%)")
        
        return user_profiles
    
    def analyze_clusters(self, user_profiles, df):
        """Analyze characteristics of each cluster"""
        
        cluster_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_users = user_profiles[user_profiles['cluster'] == cluster_id]
            
            analysis = {
                'size': len(cluster_users),
                'avg_amount': cluster_users['avg_amount'].mean(),
                'total_spent': cluster_users['total_spent'].sum(),
                'avg_transactions': cluster_users['transaction_count'].mean(),
                'txn_per_day': cluster_users['txn_per_day'].mean(),
                'category_diversity': cluster_users['category_diversity'].mean(),
                'weekend_ratio': cluster_users['weekend_ratio'].mean(),
                'top_category': cluster_users['top_category'].mode()[0] if len(cluster_users) > 0 else 'N/A'
            }
            
            cluster_analysis[cluster_id] = analysis
        
        print(f"   ✓ Analyzed {len(cluster_analysis)} clusters")
        
        return cluster_analysis
    
    def name_clusters(self, cluster_analysis):
        """Assign descriptive names to clusters"""
        
        cluster_names = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            # Determine cluster persona based on characteristics
            avg_amount = analysis['avg_amount']
            txn_per_day = analysis['txn_per_day']
            category_diversity = analysis['category_diversity']
            
            if avg_amount > 80 and txn_per_day > 5:
                name = "💎 High Spender (Active)"
                description = "Frequent, high-value transactions"
            elif avg_amount > 80 and txn_per_day <= 5:
                name = "🏆 Premium User (Selective)"
                description = "High-value but infrequent purchases"
            elif avg_amount <= 80 and txn_per_day > 5:
                name = "⚡ Active Saver"
                description = "Frequent, moderate spending"
            elif category_diversity > 10:
                name = "🌈 Diverse Shopper"
                description = "Wide variety of categories"
            else:
                name = "🏠 Casual User"
                description = "Low to moderate activity"
            
            cluster_names[cluster_id] = {
                'name': name,
                'description': description
            }
        
        print(f"   ✓ Named {len(cluster_names)} user segments")
        
        return cluster_names
    
    def show_cluster_summary(self, user_profiles, cluster_analysis, cluster_names):
        """Display comprehensive cluster summary"""
        
        self.print_header("📊 CLUSTER ANALYSIS SUMMARY")
        
        print(f"Total Users Analyzed: {len(user_profiles):,}\n")
        
        for cluster_id in sorted(cluster_analysis.keys()):
            analysis = cluster_analysis[cluster_id]
            name_info = cluster_names[cluster_id]
            
            print(f"{'═' * 80}")
            print(f"CLUSTER {cluster_id}: {name_info['name']}")
            print(f"{'═' * 80}")
            print(f"Description: {name_info['description']}")
            print(f"\n📊 Statistics:")
            print(f"   Users in Cluster:        {analysis['size']:>10,} "
                  f"({analysis['size']/len(user_profiles)*100:>5.1f}%)")
            print(f"   Average Transaction:     ₹{analysis['avg_amount']:>9,.2f}")
            print(f"   Total Spending:          ₹{analysis['total_spent']:>9,.2f}")
            print(f"   Avg Transactions:        {analysis['avg_transactions']:>10,.1f}")
            print(f"   Transactions/Day:        {analysis['txn_per_day']:>10,.2f}")
            print(f"   Category Diversity:      {analysis['category_diversity']:>10,.1f}")
            print(f"   Weekend Activity:        {analysis['weekend_ratio']*100:>10,.1f}%")
            print(f"   Top Category:            {analysis['top_category']}")
            print()
        
        # Overall insights
        self.print_header("💡 SEGMENTATION INSIGHTS")
        
        # Find most valuable cluster
        most_valuable = max(cluster_analysis.items(), 
                          key=lambda x: x[1]['total_spent'])
        print(f"🏆 Most Valuable Segment:")
        print(f"   {cluster_names[most_valuable[0]]['name']}")
        print(f"   Total Spending: ₹{most_valuable[1]['total_spent']:,.2f}")
        
        # Find most active cluster
        most_active = max(cluster_analysis.items(), 
                         key=lambda x: x[1]['txn_per_day'])
        print(f"\n⚡ Most Active Segment:")
        print(f"   {cluster_names[most_active[0]]['name']}")
        print(f"   Transactions/Day: {most_active[1]['txn_per_day']:.2f}")
        
        # Find largest cluster
        largest = max(cluster_analysis.items(), key=lambda x: x[1]['size'])
        print(f"\n👥 Largest Segment:")
        print(f"   {cluster_names[largest[0]]['name']}")
        print(f"   Users: {largest[1]['size']:,} "
              f"({largest[1]['size']/len(user_profiles)*100:.1f}%)")
        
        # Recommendations
        self.print_header("🎯 PERSONALIZATION RECOMMENDATIONS")
        
        print("Based on cluster analysis:\n")
        
        for cluster_id in sorted(cluster_analysis.keys()):
            name_info = cluster_names[cluster_id]
            analysis = cluster_analysis[cluster_id]
            
            print(f"{name_info['name']}")
            
            if "High Spender" in name_info['name']:
                print("   • Offer premium rewards program")
                print("   • Send exclusive deals and early access")
                print("   • Provide VIP customer support")
            elif "Premium User" in name_info['name']:
                print("   • Target with luxury product recommendations")
                print("   • Offer concierge services")
                print("   • Send curated high-value offers")
            elif "Active Saver" in name_info['name']:
                print("   • Provide cashback rewards")
                print("   • Send daily deals and discounts")
                print("   • Gamify frequent transactions")
            elif "Diverse Shopper" in name_info['name']:
                print("   • Cross-category recommendations")
                print("   • Bundle deals across categories")
                print("   • Discovery-focused features")
            else:
                print("   • Re-engagement campaigns")
                print("   • New user incentives")
                print("   • Educational content about features")
            print()
    
    def save_clustering_results(self, user_profiles, cluster_analysis, cluster_names):
        """Save clustering results"""
        
        self.print_header("💾 SAVING CLUSTERING RESULTS")
        
        import os
        os.makedirs('../results', exist_ok=True)
        
        # Save user cluster assignments (with all features needed for visualization)
        user_cluster_export = user_profiles[['user_id', 'cluster', 'avg_amount', 
                                            'total_spent', 'transaction_count',
                                            'txn_per_day', 'category_diversity', 
                                            'weekend_ratio']].copy()
        user_cluster_export['cluster_name'] = user_cluster_export['cluster'].map(
            lambda x: cluster_names[x]['name']
        )
        user_cluster_export.to_csv('../results/user_clusters.csv', index=False)
        print(f"✓ User clusters: ../results/user_clusters.csv")
        
        # Save cluster summary
        cluster_summary = []
        for cluster_id, analysis in cluster_analysis.items():
            summary_row = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_names[cluster_id]['name'],
                'description': cluster_names[cluster_id]['description'],
                'size': analysis['size'],
                'avg_amount': analysis['avg_amount'],
                'total_spent': analysis['total_spent'],
                'avg_transactions': analysis['avg_transactions'],
                'txn_per_day': analysis['txn_per_day'],
                'category_diversity': analysis['category_diversity'],
                'weekend_ratio': analysis['weekend_ratio'],
                'top_category': analysis['top_category']
            }
            cluster_summary.append(summary_row)
        
        cluster_summary_df = pd.DataFrame(cluster_summary)
        cluster_summary_df.to_csv('../results/cluster_summary.csv', index=False)
        print(f"✓ Cluster summary: ../results/cluster_summary.csv")
        
        print(f"\n✓ Saved 2 clustering files")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  👥 OPAM: User Behavior Clustering")
    print("  🎯 K-Means Segmentation Analysis")
    print("="*80)
    
    # Load data
    print("\n📁 Loading transaction data...")
    df = pd.read_csv('../data/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Cluster users
    clusterer = UserClusterer(n_clusters=5)
    user_profiles, cluster_analysis, cluster_names = clusterer.cluster_users(df)
    
    # Complete
    print("\n" + "="*80)
    print("  ✅ CLUSTERING COMPLETE!")
    print("="*80)
    print("\n📂 Results saved in: ../results/")
    print("\n📊 Files created:")
    print("   • user_clusters.csv - User assignments to clusters")
    print("   • cluster_summary.csv - Cluster characteristics")
    print("\nNext steps:")
    print("  1. Review cluster assignments")
    print("  2. Run: python3 visualize_clusters.py")
    print("  3. Use segments for personalization")
    print("  4. Integrate with recommendation system")
    print()

if __name__ == "__main__":
    main()
