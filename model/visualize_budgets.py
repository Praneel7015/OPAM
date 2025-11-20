"""
OPAM - Budget Visualization Module
Creates charts for budget recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BudgetVisualizer:
    """Visualize budget recommendations"""
    
    def __init__(self, output_dir='../charts'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_budget_charts(self, budget_recs, budget_summary):
        """Generate all budget visualizations"""
        
        print("\n" + "="*80)
        print("  🎨 GENERATING BUDGET VISUALIZATIONS")
        print("="*80 + "\n")
        
        # 1. Budget Recommendations
        print("1️⃣  Creating Budget Recommendations Chart...")
        self.plot_budget_recommendations(budget_recs)
        
        # 2. Savings Analysis
        print("2️⃣  Creating Savings Analysis Chart...")
        self.plot_savings_analysis(budget_recs, budget_summary)
        
        print("\n✅ All budget visualizations created!")
        print(f"📂 Charts saved in: {self.output_dir}/\n")
    
    def plot_budget_recommendations(self, budget_recs):
        """Plot budget recommendations by category"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get top categories
        top_10 = budget_recs.nlargest(10, 'current_monthly')
        
        # 1. Current vs Recommended comparison
        x = np.arange(len(top_10))
        width = 0.35
        
        bars1 = ax1.barh(x - width/2, top_10['current_monthly'], width, 
                        label='Current', alpha=0.8, color='#FF6B6B')
        bars2 = ax1.barh(x + width/2, top_10['recommended_budget'], width,
                        label='Recommended', alpha=0.8, color='#4ECDC4')
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_10['category'], fontsize=9)
        ax1.set_xlabel('Monthly Spending (₹)', fontsize=11, fontweight='bold')
        ax1.set_title('Current vs Recommended Budget (Top 10)', 
                     fontsize=12, fontweight='bold', pad=20)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Potential savings by category
        savings_cats = budget_recs[budget_recs['potential_savings'] > 0].nlargest(10, 'potential_savings')
        
        colors = sns.color_palette('Greens_r', len(savings_cats))
        bars = ax2.barh(range(len(savings_cats)), savings_cats['potential_savings'], 
                       color=colors, alpha=0.8)
        ax2.set_yticks(range(len(savings_cats)))
        ax2.set_yticklabels(savings_cats['category'], fontsize=9)
        ax2.set_xlabel('Potential Savings (₹)', fontsize=11, fontweight='bold')
        ax2.set_title('Savings Opportunity by Category', 
                     fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, savings_cats['potential_savings'])):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' ₹{val:.0f}', va='center', fontsize=8, fontweight='bold')
        
        # 3. Budget allocation pie chart
        budget_allocation = budget_recs.nlargest(8, 'recommended_budget')
        others = budget_recs.nsmallest(len(budget_recs) - 8, 'recommended_budget')['recommended_budget'].sum()
        
        if others > 0:
            labels = list(budget_allocation['category']) + ['Others']
            sizes = list(budget_allocation['recommended_budget']) + [others]
        else:
            labels = list(budget_allocation['category'])
            sizes = list(budget_allocation['recommended_budget'])
        
        colors = sns.color_palette('Set3', len(sizes))
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        for text in texts:
            text.set_fontsize(8)
        ax3.set_title('Recommended Budget Allocation', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # 4. Priority-based budget distribution
        priority_budget = budget_recs.groupby('priority')['recommended_budget'].sum()
        priority_order = ['Essential', 'Necessary', 'Standard', 'Discretionary']
        priority_budget = priority_budget.reindex([p for p in priority_order if p in priority_budget.index])
        
        colors_priority = {'Essential': '#4ECDC4', 'Necessary': '#FFE66D', 
                          'Standard': '#FF6B6B', 'Discretionary': '#95E1D3'}
        bar_colors = [colors_priority.get(p, '#999999') for p in priority_budget.index]
        
        bars = ax4.bar(range(len(priority_budget)), priority_budget.values, 
                      color=bar_colors, alpha=0.8)
        ax4.set_xticks(range(len(priority_budget)))
        ax4.set_xticklabels(priority_budget.index, fontsize=10)
        ax4.set_ylabel('Monthly Budget (₹)', fontsize=11, fontweight='bold')
        ax4.set_title('Budget by Priority Level', fontsize=12, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, priority_budget.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'₹{val:,.0f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/19_budget_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_savings_analysis(self, budget_recs, budget_summary):
        """Plot detailed savings analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract summary values
        current_monthly = budget_summary[budget_summary['metric'] == 'Current Monthly']['value'].values[0]
        recommended_monthly = budget_summary[budget_summary['metric'] == 'Recommended Monthly']['value'].values[0]
        monthly_savings = budget_summary[budget_summary['metric'] == 'Monthly Savings']['value'].values[0]
        annual_savings = budget_summary[budget_summary['metric'] == 'Annual Savings']['value'].values[0]
        
        # 1. Overall savings comparison
        categories = ['Current\nSpending', 'Recommended\nBudget', 'Monthly\nSavings']
        values = [current_monthly, recommended_monthly, monthly_savings]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Amount (₹)', fontsize=11, fontweight='bold')
        ax1.set_title('Budget Overview', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'₹{val:,.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        # 2. Cumulative savings projection
        months = np.arange(1, 13)
        cumulative_savings = months * monthly_savings
        
        ax2.plot(months, cumulative_savings, marker='o', linewidth=3, 
                markersize=8, color='#4ECDC4')
        ax2.fill_between(months, cumulative_savings, alpha=0.3, color='#4ECDC4')
        ax2.set_xlabel('Months', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Savings (₹)', fontsize=11, fontweight='bold')
        ax2.set_title('12-Month Savings Projection', fontsize=12, fontweight='bold', pad=20)
        ax2.set_xticks(months)
        ax2.grid(True, alpha=0.3)
        
        # Add milestone markers
        for month in [3, 6, 9, 12]:
            savings_at_month = month * monthly_savings
            ax2.axhline(y=savings_at_month, color='red', linestyle='--', alpha=0.3)
            ax2.text(0.5, savings_at_month, f'{month}M: ₹{savings_at_month:,.0f}',
                    fontsize=8, va='bottom')
        
        # 3. Spending reduction opportunities
        top_reductions = budget_recs[budget_recs['potential_savings'] > 0].nlargest(8, 'potential_savings')
        
        categories_short = [cat[:15] + '...' if len(cat) > 15 else cat 
                          for cat in top_reductions['category']]
        reduction_pct = (top_reductions['potential_savings'] / top_reductions['current_monthly'] * 100)
        
        colors = sns.color_palette('RdYlGn_r', len(top_reductions))
        bars = ax3.barh(range(len(top_reductions)), reduction_pct, color=colors, alpha=0.8)
        ax3.set_yticks(range(len(top_reductions)))
        ax3.set_yticklabels(categories_short, fontsize=9)
        ax3.set_xlabel('Reduction Opportunity (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Spending Reduction Opportunities', fontsize=12, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, pct, val) in enumerate(zip(bars, reduction_pct, top_reductions['potential_savings'])):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f' {pct:.1f}% (₹{val:.0f})', va='center', fontsize=8, fontweight='bold')
        
        # 4. Budget impact summary
        # Create a visual summary table
        summary_data = [
            ['Current Monthly', f'₹{current_monthly:,.2f}'],
            ['Recommended', f'₹{recommended_monthly:,.2f}'],
            ['Monthly Savings', f'₹{monthly_savings:,.2f}'],
            ['Annual Savings', f'₹{annual_savings:,.2f}'],
            ['Savings Rate', f'{(monthly_savings/current_monthly*100):.1f}%']
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colColours=['#E8E8E8', '#E8E8E8'])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        # Style the cells
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold', color='white')
                elif i == 3 or i == 4:  # Highlight savings rows
                    cell.set_facecolor('#E8F8F5')
                    cell.set_text_props(weight='bold')
        
        ax4.set_title('Budget Impact Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/20_savings_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main visualization generation"""
    print("\n" + "="*80)
    print("  🎨 BUDGET VISUALIZATION MODULE")
    print("="*80)
    
    # Load budget recommendations
    print("\n📁 Loading budget recommendations...")
    
    import os
    if not os.path.exists('../results/budget_recommendations.csv'):
        print("\n❌ ERROR: Budget recommendations not found!")
        print("   Please run budget_recommender.py first:")
        print("   python3 budget_recommender.py")
        return
    
    budget_recs = pd.read_csv('../results/budget_recommendations.csv')
    budget_summary = pd.read_csv('../results/budget_summary.csv')
    
    print(f"✓ Loaded {len(budget_recs)} category budgets")
    print(f"✓ Loaded budget summary")
    
    # Create visualizer
    visualizer = BudgetVisualizer(output_dir='../charts')
    
    # Generate charts
    visualizer.create_all_budget_charts(budget_recs, budget_summary)
    
    print("\n" + "="*80)
    print("  ✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📊 Generated budget visualization charts:")
    print("   • Chart 19: Budget Recommendations")
    print("   • Chart 20: Savings Analysis")
    print(f"\n📂 Find them in: ../charts/\n")
    
    print("="*80)
    print("  🎉 CONGRATULATIONS! 100% PROJECT COMPLETION!")
    print("="*80)
    print("\n✨ ALL 20 CHARTS CREATED!")
    print("   Charts 1-8:   Data Analysis & Prediction")
    print("   Charts 9-13:  Anomaly Detection")
    print("   Charts 14-16: Fraud Detection")
    print("   Charts 17-18: User Clustering")
    print("   Charts 19-20: Budget Recommendations")
    print("\n🏆 PERFECT SCORE: 100% COMPLETE!")
    print()

if __name__ == "__main__":
    main()
