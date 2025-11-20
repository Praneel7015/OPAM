"""
OPAM - Budget Recommendations Module
AI-powered smart budget suggestions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BudgetRecommender:
    """Generate personalized budget recommendations"""
    
    def __init__(self):
        self.recommendations = {}
        
    def print_header(self, text, char="="):
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def generate_recommendations(self, df):
        """Generate comprehensive budget recommendations"""
        
        self.print_header("💰 SMART BUDGET RECOMMENDATIONS")
        
        print(f"📊 Analyzing spending patterns for budget optimization...")
        print(f"   Total Transactions: {len(df):,}")
        print(f"   Total Amount: ₹{df['amount'].sum():,.2f}\n")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Calculate current spending
        print("1️⃣  Analyzing current spending patterns...")
        current_spending = self.analyze_current_spending(df)
        
        # 2. Generate category budgets
        print("\n2️⃣  Generating category-wise budget recommendations...")
        category_budgets = self.generate_category_budgets(df, current_spending)
        
        # 3. Calculate savings potential
        print("\n3️⃣  Calculating savings opportunities...")
        savings_analysis = self.analyze_savings_potential(df, category_budgets)
        
        # 4. Create spending goals
        print("\n4️⃣  Creating personalized spending goals...")
        spending_goals = self.create_spending_goals(current_spending, savings_analysis)
        
        # 5. Generate alerts
        print("\n5️⃣  Setting up budget alerts...")
        budget_alerts = self.generate_budget_alerts(category_budgets)
        
        # Show recommendations
        self.show_recommendations(current_spending, category_budgets, 
                                 savings_analysis, spending_goals)
        
        # Save results
        self.save_recommendations(current_spending, category_budgets, 
                                 savings_analysis, spending_goals, budget_alerts)
        
        return category_budgets, savings_analysis, spending_goals
    
    def analyze_current_spending(self, df):
        """Analyze current spending patterns"""
        
        # Calculate time period
        days_in_data = (df['date'].max() - df['date'].min()).days + 1
        months_in_data = days_in_data / 30.44
        
        # Overall spending
        total_spent = df['amount'].sum()
        monthly_avg = total_spent / months_in_data
        daily_avg = total_spent / days_in_data
        
        # Category spending
        category_spending = df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        category_spending.columns = ['total', 'avg_transaction', 'count']
        category_spending['monthly_avg'] = (category_spending['total'] / months_in_data).round(2)
        category_spending = category_spending.sort_values('total', ascending=False)
        
        analysis = {
            'total_spent': total_spent,
            'monthly_avg': monthly_avg,
            'daily_avg': daily_avg,
            'days_analyzed': days_in_data,
            'months_analyzed': months_in_data,
            'category_spending': category_spending
        }
        
        print(f"   ✓ Analyzed {days_in_data} days of spending")
        print(f"   💰 Monthly Average: ₹{monthly_avg:,.2f}")
        print(f"   💰 Daily Average: ₹{daily_avg:,.2f}")
        
        return analysis
    
    def generate_category_budgets(self, df, current_spending):
        """Generate recommended budgets for each category"""
        
        category_spending = current_spending['category_spending']
        
        budgets = []
        
        for category in category_spending.index:
            cat_data = category_spending.loc[category]
            current_monthly = cat_data['monthly_avg']
            
            # Calculate recommended budget (different strategies for different categories)
            if category in ['grocery_pos', 'grocery_net', 'food_dining']:
                # Essential categories: Current spending + 5% buffer
                recommended = current_monthly * 1.05
                priority = 'Essential'
            elif category in ['shopping_pos', 'shopping_net', 'entertainment']:
                # Discretionary: Reduce by 10%
                recommended = current_monthly * 0.90
                priority = 'Discretionary'
            elif category in ['gas_transport', 'travel']:
                # Transportation: Current + 10% buffer
                recommended = current_monthly * 1.10
                priority = 'Necessary'
            else:
                # Other: Maintain current
                recommended = current_monthly
                priority = 'Standard'
            
            # Calculate potential savings
            potential_savings = current_monthly - recommended
            
            budgets.append({
                'category': category,
                'current_monthly': round(current_monthly, 2),
                'recommended_budget': round(recommended, 2),
                'potential_savings': round(potential_savings, 2),
                'priority': priority,
                'transactions_per_month': round(cat_data['count'] / current_spending['months_analyzed'], 1)
            })
        
        budgets_df = pd.DataFrame(budgets)
        budgets_df = budgets_df.sort_values('current_monthly', ascending=False)
        
        print(f"   ✓ Generated budgets for {len(budgets_df)} categories")
        print(f"   📊 Total recommended monthly: ₹{budgets_df['recommended_budget'].sum():,.2f}")
        
        return budgets_df
    
    def analyze_savings_potential(self, df, category_budgets):
        """Analyze potential savings opportunities"""
        
        total_current = category_budgets['current_monthly'].sum()
        total_recommended = category_budgets['recommended_budget'].sum()
        monthly_savings = total_current - total_recommended
        annual_savings = monthly_savings * 12
        
        # Identify top saving categories
        savings_by_category = category_budgets[
            category_budgets['potential_savings'] > 0
        ].sort_values('potential_savings', ascending=False)
        
        # Calculate savings percentage
        savings_percentage = (monthly_savings / total_current) * 100 if total_current > 0 else 0
        
        analysis = {
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'savings_percentage': savings_percentage,
            'top_saving_categories': savings_by_category.head(5),
            'total_current': total_current,
            'total_recommended': total_recommended
        }
        
        print(f"   ✓ Identified savings opportunities")
        print(f"   💵 Monthly Savings Potential: ₹{monthly_savings:,.2f}")
        print(f"   💵 Annual Savings Potential: ₹{annual_savings:,.2f}")
        print(f"   📊 Savings Rate: {savings_percentage:.1f}%")
        
        return analysis
    
    def create_spending_goals(self, current_spending, savings_analysis):
        """Create personalized spending goals"""
        
        current_monthly = savings_analysis['total_current']
        
        # Goal 1: Maintain current spending
        maintain_goal = {
            'name': '🎯 Maintain Current',
            'target': current_monthly,
            'description': 'Keep spending at current level',
            'difficulty': 'Easy',
            'monthly_target': current_monthly
        }
        
        # Goal 2: Reduce by 5%
        reduce_5_goal = {
            'name': '💪 Reduce 5%',
            'target': current_monthly * 0.95,
            'description': 'Small reduction in discretionary spending',
            'difficulty': 'Moderate',
            'monthly_target': current_monthly * 0.95,
            'savings': current_monthly * 0.05
        }
        
        # Goal 3: Reduce by 10%
        reduce_10_goal = {
            'name': '🏆 Reduce 10%',
            'target': current_monthly * 0.90,
            'description': 'Significant spending optimization',
            'difficulty': 'Challenging',
            'monthly_target': current_monthly * 0.90,
            'savings': current_monthly * 0.10
        }
        
        # Goal 4: Optimal budget
        optimal_goal = {
            'name': '⭐ Optimal Budget',
            'target': savings_analysis['total_recommended'],
            'description': 'AI-recommended optimal spending',
            'difficulty': 'Strategic',
            'monthly_target': savings_analysis['total_recommended'],
            'savings': savings_analysis['monthly_savings']
        }
        
        goals = [maintain_goal, reduce_5_goal, reduce_10_goal, optimal_goal]
        
        print(f"   ✓ Created {len(goals)} spending goals")
        
        return goals
    
    def generate_budget_alerts(self, category_budgets):
        """Generate budget alert thresholds"""
        
        alerts = []
        
        for _, row in category_budgets.iterrows():
            budget = row['recommended_budget']
            
            alerts.append({
                'category': row['category'],
                'budget': budget,
                'alert_80': budget * 0.80,  # Warning at 80%
                'alert_90': budget * 0.90,  # Caution at 90%
                'alert_100': budget,        # Limit at 100%
                'priority': row['priority']
            })
        
        alerts_df = pd.DataFrame(alerts)
        
        print(f"   ✓ Configured {len(alerts_df)} budget alerts")
        
        return alerts_df
    
    def show_recommendations(self, current_spending, category_budgets, 
                            savings_analysis, spending_goals):
        """Display comprehensive recommendations"""
        
        self.print_header("📊 BUDGET ANALYSIS SUMMARY")
        
        # Current situation
        print("💳 Current Spending:")
        print(f"   Monthly Average:         ₹{current_spending['monthly_avg']:>12,.2f}")
        print(f"   Daily Average:           ₹{current_spending['daily_avg']:>12,.2f}")
        print(f"   Analysis Period:         {current_spending['days_analyzed']:>12,} days")
        
        # Recommendations
        print(f"\n💰 Recommended Budget:")
        print(f"   Monthly Target:          ₹{savings_analysis['total_recommended']:>12,.2f}")
        print(f"   Potential Savings:       ₹{savings_analysis['monthly_savings']:>12,.2f}")
        print(f"   Savings Rate:            {savings_analysis['savings_percentage']:>12.1f}%")
        print(f"   Annual Savings:          ₹{savings_analysis['annual_savings']:>12,.2f}")
        
        # Top categories
        self.print_header("📋 TOP SPENDING CATEGORIES")
        
        top_5 = category_budgets.head(5)
        print(f"{'Category':20s} {'Current':>12s} {'Recommended':>12s} {'Savings':>12s} {'Priority'}")
        print("─" * 80)
        
        for _, row in top_5.iterrows():
            print(f"{row['category']:20s} "
                  f"₹{row['current_monthly']:>10,.2f} "
                  f"₹{row['recommended_budget']:>10,.2f} "
                  f"₹{row['potential_savings']:>10,.2f} "
                  f"{row['priority']}")
        
        # Savings opportunities
        if not savings_analysis['top_saving_categories'].empty:
            self.print_header("💡 TOP SAVINGS OPPORTUNITIES")
            
            for _, row in savings_analysis['top_saving_categories'].head(3).iterrows():
                print(f"🎯 {row['category']}")
                print(f"   Current: ₹{row['current_monthly']:,.2f}/month")
                print(f"   Target: ₹{row['recommended_budget']:,.2f}/month")
                print(f"   Potential Savings: ₹{row['potential_savings']:,.2f}/month "
                      f"(₹{row['potential_savings'] * 12:,.2f}/year)")
                print()
        
        # Spending goals
        self.print_header("🎯 PERSONALIZED SPENDING GOALS")
        
        for goal in spending_goals:
            print(f"{goal['name']}")
            print(f"   Target: ₹{goal['monthly_target']:,.2f}/month")
            print(f"   Description: {goal['description']}")
            print(f"   Difficulty: {goal['difficulty']}")
            if 'savings' in goal:
                print(f"   Monthly Savings: ₹{goal['savings']:,.2f}")
                print(f"   Annual Savings: ₹{goal['savings'] * 12:,.2f}")
            print()
        
        # Recommendations
        self.print_header("✨ ACTIONABLE RECOMMENDATIONS")
        
        print("Based on your spending analysis:\n")
        
        print("1️⃣  IMMEDIATE ACTIONS:")
        print("   • Set up budget alerts for top 5 spending categories")
        print("   • Review discretionary spending this week")
        print(f"   • Target to save ₹{savings_analysis['monthly_savings']:,.2f}/month")
        
        print("\n2️⃣  SHORT-TERM (This Month):")
        discretionary = category_budgets[category_budgets['priority'] == 'Discretionary']
        if len(discretionary) > 0:
            print(f"   • Reduce discretionary spending by 10%")
            print(f"   • Focus on: {', '.join(discretionary.head(2)['category'].values)}")
        print("   • Track spending daily using the app")
        print("   • Set weekly spending limits")
        
        print("\n3️⃣  LONG-TERM (Next 3 Months):")
        print(f"   • Achieve {spending_goals[2]['name']} goal")
        print(f"   • Save ₹{savings_analysis['annual_savings']/4:,.2f} per quarter")
        print("   • Review and adjust budgets monthly")
        print("   • Build emergency fund (3 months expenses)")
        
        print("\n4️⃣  CATEGORY-SPECIFIC TIPS:")
        for _, row in category_budgets.head(3).iterrows():
            print(f"\n   {row['category'].title()}:")
            if row['priority'] == 'Essential':
                print("   • Look for bulk purchase discounts")
                print("   • Compare prices across stores")
            elif row['priority'] == 'Discretionary':
                print("   • Set a monthly cap")
                print("   • Wait 24 hours before purchases")
            else:
                print("   • Plan ahead to avoid premium pricing")
                print("   • Explore subscription alternatives")
    
    def save_recommendations(self, current_spending, category_budgets, 
                           savings_analysis, spending_goals, budget_alerts):
        """Save recommendations to files"""
        
        self.print_header("💾 SAVING RECOMMENDATIONS")
        
        import os
        os.makedirs('../results', exist_ok=True)
        
        # Save category budgets
        category_budgets.to_csv('../results/budget_recommendations.csv', index=False)
        print(f"✓ Budget recommendations: ../results/budget_recommendations.csv")
        
        # Save budget alerts
        budget_alerts.to_csv('../results/budget_alerts.csv', index=False)
        print(f"✓ Budget alerts: ../results/budget_alerts.csv")
        
        # Save spending goals
        goals_df = pd.DataFrame(spending_goals)
        goals_df.to_csv('../results/spending_goals.csv', index=False)
        print(f"✓ Spending goals: ../results/spending_goals.csv")
        
        # Save summary
        summary = {
            'metric': ['Current Monthly', 'Recommended Monthly', 'Monthly Savings', 
                      'Annual Savings', 'Savings Percentage'],
            'value': [
                current_spending['monthly_avg'],
                savings_analysis['total_recommended'],
                savings_analysis['monthly_savings'],
                savings_analysis['annual_savings'],
                savings_analysis['savings_percentage']
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('../results/budget_summary.csv', index=False)
        print(f"✓ Budget summary: ../results/budget_summary.csv")
        
        print(f"\n✓ Saved 4 recommendation files")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  💰 OPAM: Smart Budget Recommendations")
    print("  🎯 AI-Powered Spending Optimization")
    print("="*80)
    
    # Load data
    print("\n📁 Loading transaction data...")
    df = pd.read_csv('../data/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Generate recommendations
    recommender = BudgetRecommender()
    category_budgets, savings_analysis, spending_goals = recommender.generate_recommendations(df)
    
    # Complete
    print("\n" + "="*80)
    print("  ✅ RECOMMENDATIONS COMPLETE!")
    print("="*80)
    print("\n📂 Results saved in: ../results/")
    print("\n📊 Files created:")
    print("   • budget_recommendations.csv - Category budgets")
    print("   • budget_alerts.csv - Alert thresholds")
    print("   • spending_goals.csv - Personalized goals")
    print("   • budget_summary.csv - Overall summary")
    print("\nNext steps:")
    print("  1. Review budget recommendations")
    print("  2. Run: python3 visualize_budgets.py")
    print("  3. Implement budget tracking")
    print("  4. Set up monthly reviews")
    print()

if __name__ == "__main__":
    main()
