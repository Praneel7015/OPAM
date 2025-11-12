"""
OPAM - Next Month Expense Predictor
Complete ML Pipeline with Terminal Outputs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib

class OPAMExpensePredictor:
    """AI/ML Model for predicting next month's expenses"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.category_models = {}
        
    def print_header(self, text, char="="):
        """Print formatted header"""
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def load_and_prepare_data(self, df):
        """Load and prepare transaction data"""
        self.print_header("📊 DATA LOADING & PREPARATION")
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Total Transactions: {len(df):,}")
        print(f"✓ Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✓ Total Spending: ₹{df['amount'].sum():,.2f}")
        print(f"✓ Average Transaction: ₹{df['amount'].mean():.2f}")
        print(f"✓ Categories: {df['category'].nunique()}")
        print(f"✓ Merchants: {df['merchant'].nunique()}")
        
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_summary = df.groupby('year_month')['amount'].agg(['sum', 'count', 'mean'])
        
        print("\n📈 Monthly Spending Pattern:")
        print(monthly_summary.to_string())
        
        print("\n💰 Category-wise Spending:")
        category_summary = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
        category_summary['percentage'] = (category_summary['sum'] / df['amount'].sum() * 100).round(2)
        print(category_summary.to_string())
        
        return df
    
    def engineer_features(self, df):
        """Feature engineering"""
        self.print_header("🔧 FEATURE ENGINEERING")
        
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = (df['day'] <= 7).astype(int)
        df['is_month_end'] = (df['day'] >= 24).astype(int)
        
        print("✓ Time-based features created:")
        print("  - Year, Month, Day, Day of week")
        print("  - Weekend indicator, Quarter")
        print("  - Month start/end indicators")
        
        return df
    
    def create_monthly_features(self, df):
        """Create monthly aggregated features"""
        self.print_header("📊 MONTHLY FEATURE AGGREGATION")
        
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly = df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'is_recurring': 'sum',
            'is_weekend': 'mean'
        }).reset_index()
        
        monthly.columns = ['year_month', 'total_amount', 'avg_amount', 'std_amount', 
                          'min_amount', 'max_amount', 'num_transactions', 
                          'recurring_count', 'weekend_ratio']
        
        category_pivot = df.groupby(['year_month', 'category'])['amount'].sum().unstack(fill_value=0)
        monthly = monthly.merge(category_pivot, left_on='year_month', right_index=True, how='left')
        
        print("\n✓ Creating lag features:")
        for lag in [1, 2, 3, 6]:
            monthly[f'total_lag_{lag}m'] = monthly['total_amount'].shift(lag)
            monthly[f'avg_lag_{lag}m'] = monthly['avg_amount'].shift(lag)
            monthly[f'count_lag_{lag}m'] = monthly['num_transactions'].shift(lag)
            print(f"  - Lag {lag} months")
        
        print("\n✓ Creating rolling features:")
        for window in [2, 3, 6]:
            monthly[f'rolling_mean_{window}m'] = monthly['total_amount'].rolling(window=window).mean()
            monthly[f'rolling_std_{window}m'] = monthly['total_amount'].rolling(window=window).std()
            print(f"  - {window} month rolling")
        
        monthly['month_over_month_growth'] = monthly['total_amount'].pct_change()
        monthly = monthly.fillna(0)
        
        print(f"\n✓ Total features: {len(monthly.columns) - 1}")
        print(f"✓ Training samples: {len(monthly)}")
        
        return monthly
    
    def train_models(self, monthly_df):
        """Train multiple ML models"""
        self.print_header("🤖 MODEL TRAINING")
        
        monthly_df = monthly_df.iloc[6:]
        
        if len(monthly_df) < 5:
            print("⚠️  ERROR: Need at least 12 months of data")
            return None
        
        exclude_cols = ['year_month', 'total_amount']
        feature_cols = [col for col in monthly_df.columns if col not in exclude_cols]
        
        X = monthly_df[feature_cols].values
        y = monthly_df['total_amount'].values
        
        self.feature_columns = feature_cols
        
        print(f"✓ Feature Matrix: {X.shape}")
        print(f"✓ Target Vector: {y.shape}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        test_size = max(1, int(len(X) * 0.2))
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        print(f"\n✓ Train Set: {len(X_train)} months")
        print(f"✓ Test Set: {len(X_test)} months")
        
        predictions = {}
        
        # 1. Linear Regression
        print("\n" + "─" * 80)
        print("1️⃣  Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        predictions['Linear Regression'] = lr_pred
        self.models['linear'] = lr
        self._print_metrics("Linear Regression", y_test, lr_pred)
        
        # 2. Ridge
        print("\n" + "─" * 80)
        print("2️⃣  Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        predictions['Ridge'] = ridge_pred
        self.models['ridge'] = ridge
        self._print_metrics("Ridge", y_test, ridge_pred)
        
        # 3. Random Forest
        print("\n" + "─" * 80)
        print("3️⃣  Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        predictions['Random Forest'] = rf_pred
        self.models['random_forest'] = rf
        self._print_metrics("Random Forest", y_test, rf_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print("\n📊 Top 10 Important Features:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")
        
        # 4. Gradient Boosting
        print("\n" + "─" * 80)
        print("4️⃣  Training Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        predictions['Gradient Boosting'] = gb_pred
        self.models['gradient_boosting'] = gb
        self._print_metrics("Gradient Boosting", y_test, gb_pred)
        
        # 5. XGBoost
        print("\n" + "─" * 80)
        print("5️⃣  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        predictions['XGBoost'] = xgb_pred
        self.models['xgboost'] = xgb_model
        self._print_metrics("XGBoost", y_test, xgb_pred)
        
        # 6. Ensemble
        print("\n" + "─" * 80)
        print("6️⃣  Creating Ensemble Model...")
        ensemble_pred = (lr_pred * 0.1 + ridge_pred * 0.1 + rf_pred * 0.25 + gb_pred * 0.25 + xgb_pred * 0.3)
        predictions['Ensemble'] = ensemble_pred
        self._print_metrics("Ensemble", y_test, ensemble_pred)
        
        # Model comparison
        self.print_header("📊 MODEL COMPARISON")
        
        comparison = []
        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            mape = mean_absolute_percentage_error(y_test, pred) * 100
            
            comparison.append({
                'Model': name,
                'RMSE': f'₹{rmse:,.2f}',
                'MAE': f'₹{mae:,.2f}',
                'R² Score': f'{r2:.4f}',
                'MAPE': f'{mape:.2f}%'
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
        
        self.training_history = {
            'X_test': X_test,
            'y_test': y_test,
            'predictions': predictions,
            'monthly_df': monthly_df
        }
        
        return predictions
    
    def _print_metrics(self, model_name, y_true, y_pred):
        """Print model metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        print(f"  RMSE:  ₹{rmse:,.2f}")
        print(f"  MAE:   ₹{mae:,.2f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        if rmse < 500:
            accuracy = "Excellent"
        elif rmse < 1000:
            accuracy = "Very Good"
        elif rmse < 2000:
            accuracy = "Good"
        elif rmse < 3000:
            accuracy = "Fair"
        else:
            accuracy = "Needs Improvement"
        
        print(f"  Accuracy: {accuracy}")
    
    def predict_next_month(self, monthly_df):
        """Predict next month"""
        self.print_header("🔮 NEXT MONTH PREDICTION")
        
        last_row = monthly_df.iloc[-1]
        X_pred = last_row[self.feature_columns].values.reshape(1, -1)
        X_pred_scaled = self.scalers['main'].transform(X_pred)
        
        print("🎯 Individual Model Predictions:")
        print("─" * 80)
        
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_pred_scaled)[0]
            predictions[name] = pred
            print(f"{name:20s}: ₹{pred:,.2f}")
        
        ensemble_pred = (
            predictions['linear'] * 0.1 +
            predictions['ridge'] * 0.1 +
            predictions['random_forest'] * 0.25 +
            predictions['gradient_boosting'] * 0.25 +
            predictions['xgboost'] * 0.3
        )
        
        print("─" * 80)
        print(f"{'🎖️  ENSEMBLE':20s}: ₹{ensemble_pred:,.2f}")
        
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        confidence = max(0, min(100, 100 - (pred_std / pred_mean) * 100))
        
        lower_bound = ensemble_pred - pred_std
        upper_bound = ensemble_pred + pred_std
        
        print(f"\n📊 Prediction Analysis:")
        print(f"  Confidence Score: {confidence:.1f}%")
        print(f"  Prediction Range: ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}")
        print(f"  Standard Deviation: ₹{pred_std:,.2f}")
        
        hist_avg = monthly_df['total_amount'].mean()
        diff_from_avg = ((ensemble_pred - hist_avg) / hist_avg) * 100
        
        print(f"\n📈 Historical Comparison:")
        print(f"  Historical Average: ₹{hist_avg:,.2f}")
        print(f"  Predicted vs Average: {diff_from_avg:+.1f}%")
        
        if diff_from_avg > 10:
            print(f"  ⚠️  WARNING: {diff_from_avg:.1f}% higher than usual!")
        elif diff_from_avg < -10:
            print(f"  ✅ GOOD: {abs(diff_from_avg):.1f}% lower!")
        else:
            print(f"  📊 NORMAL: Spending in expected range")
        
        return {
            'predicted_amount': ensemble_pred,
            'confidence': confidence,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'individual_predictions': predictions
        }
    
    def train_category_models(self, df):
        """Train category models"""
        self.print_header("🎯 CATEGORY-WISE PREDICTION")
        
        df['year_month'] = df['date'].dt.to_period('M')
        category_predictions = {}
        
        for category in sorted(df['category'].unique()):
            cat_df = df[df['category'] == category]
            monthly_cat = cat_df.groupby('year_month')['amount'].sum().reset_index()
            monthly_cat.columns = ['year_month', 'amount']
            
            if len(monthly_cat) < 4:
                continue
            
            for lag in [1, 2, 3]:
                monthly_cat[f'lag_{lag}'] = monthly_cat['amount'].shift(lag)
            
            monthly_cat = monthly_cat.dropna()
            
            if len(monthly_cat) < 3:
                continue
            
            X = monthly_cat[[f'lag_{i}' for i in [1, 2, 3]]].values
            y = monthly_cat['amount'].values
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            last_features = X[-1].reshape(1, -1)
            prediction = model.predict(last_features)[0]
            
            self.category_models[category] = model
            category_predictions[category] = max(0, prediction)
            
            hist_avg = monthly_cat['amount'].mean()
            
            print(f"\n{category}:")
            print(f"  Predicted: ₹{prediction:,.2f}")
            print(f"  Historical Avg: ₹{hist_avg:,.2f}")
            print(f"  Change: {((prediction - hist_avg) / hist_avg * 100):+.1f}%")
        
        print("\n" + "─" * 80)
        print(f"📊 Total Predicted: ₹{sum(category_predictions.values()):,.2f}")
        
        return category_predictions
    
    def generate_insights(self, df, prediction_result, category_predictions):
        """Generate insights"""
        self.print_header("💡 ACTIONABLE INSIGHTS")
        
        df['year_month'] = df['date'].dt.to_period('M')
        last_month = df['year_month'].max()
        last_month_spending = df[df['year_month'] == last_month]['amount'].sum()
        
        predicted = prediction_result['predicted_amount']
        diff = predicted - last_month_spending
        pct_diff = (diff / last_month_spending) * 100
        
        print(f"1️⃣  SPENDING TREND:")
        print(f"   Last Month: ₹{last_month_spending:,.2f}")
        print(f"   Next Month Predicted: ₹{predicted:,.2f}")
        print(f"   Change: ₹{diff:,.2f} ({pct_diff:+.1f}%)")
        
        if pct_diff > 10:
            print(f"   ⚠️  HIGH ALERT: Significant increase!")
        elif pct_diff < -10:
            print(f"   ✅ GOOD: Spending decreasing!")
        
        print(f"\n2️⃣  TOP CATEGORIES:")
        last_month_df = df[df['year_month'] == last_month]
        top_categories = last_month_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
        
        for idx, (cat, amount) in enumerate(top_categories.items(), 1):
            pct = (amount / last_month_spending) * 100
            print(f"   {idx}. {cat:20s} ₹{amount:,.2f} ({pct:.1f}%)")
        
        print(f"\n3️⃣  SAVINGS OPPORTUNITIES:")
        
        for category, pred_amount in category_predictions.items():
            cat_df = df[df['category'] == category]
            recent_avg = cat_df[cat_df['year_month'] == last_month]['amount'].sum()
            
            if recent_avg > 0:
                increase = ((pred_amount - recent_avg) / recent_avg) * 100
                
                if increase > 15:
                    savings_potential = (pred_amount - recent_avg) * 0.3
                    print(f"   • {category}: {increase:.0f}% increase")
                    print(f"     💰 Potential savings: ₹{savings_potential:,.2f}")
        
        print(f"\n4️⃣  RECOMMENDATIONS:")
        print(f"   ✓ Set budget alert at ₹{predicted * 0.9:,.2f}")
        print(f"   ✓ Review unused subscriptions")
        print(f"   ✓ Plan major purchases carefully")
        print(f"   ✓ Track daily spending")

# Main execution
def main():
    print("\n" + "="*80)
    print("  🚀 OPAM: Next Month Expense Predictor")
    print("="*80)
    
    predictor = OPAMExpensePredictor()
    
    print("\n📁 Loading data...")
    df = pd.read_csv('../data/transactions.csv')
    
    df = predictor.load_and_prepare_data(df)
    df = predictor.engineer_features(df)
    monthly_df = predictor.create_monthly_features(df)
    predictor.train_models(monthly_df)
    prediction_result = predictor.predict_next_month(monthly_df)
    category_predictions = predictor.train_category_models(df)
    predictor.generate_insights(df, prediction_result, category_predictions)
    
    predictor.print_header("✅ ANALYSIS COMPLETE")
    print("All models trained successfully!\n")

if __name__ == "__main__":
    main()
