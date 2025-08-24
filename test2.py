# Solar Panel Power Output Prediction - Optimized ML Pipeline
# Domain: Electrical & Electronics Engineering

"""
PROBLEM STATEMENT: Predict solar panel power output for grid management optimization
DATASET: Solar Power Generation (1200+ records, 8 features)
TARGET: Power output prediction using environmental & system parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SolarPowerMLPipeline:
    def __init__(self, n_samples=1500):
        self.n_samples = n_samples
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        
    def create_dataset(self):
        """Generate realistic solar power dataset with enhanced features"""
        np.random.seed(42)
        
        # Environmental parameters
        data = {
            'Solar_Irradiance': np.clip(np.random.normal(800, 250, self.n_samples), 50, 1400),
            'Ambient_Temp': np.clip(np.random.normal(25, 10, self.n_samples), -5, 50),
            'Wind_Speed': np.clip(np.random.exponential(4, self.n_samples), 0, 20),
            'Humidity': np.clip(np.random.normal(55, 25, self.n_samples), 10, 95),
            'Cloud_Cover': np.clip(np.random.beta(2, 3, self.n_samples) * 100, 0, 100),
            'Weather': np.random.choice(['Sunny', 'Partly_Cloudy', 'Cloudy', 'Overcast'], 
                                     self.n_samples, p=[0.35, 0.35, 0.2, 0.1]),
            'Panel_Type': np.random.choice(['Mono', 'Poly', 'Thin'], 
                                         self.n_samples, p=[0.5, 0.35, 0.15]),
            'Age_Years': np.random.randint(0, 25, self.n_samples),
            'Tilt_Angle': np.random.normal(30, 5, self.n_samples)
        }
        
        # Calculate panel temperature
        data['Panel_Temp'] = data['Ambient_Temp'] + np.random.normal(20, 8, self.n_samples)
        
        # Calculate power output with realistic physics
        weather_factors = {'Sunny': 1.0, 'Partly_Cloudy': 0.75, 'Cloudy': 0.45, 'Overcast': 0.25}
        panel_factors = {'Mono': 1.0, 'Poly': 0.88, 'Thin': 0.72}
        
        power = []
        for i in range(self.n_samples):
            base_power = (data['Solar_Irradiance'][i] / 1000) * 120  # 120kW capacity
            
            # Apply factors
            base_power *= weather_factors[data['Weather'][i]]
            base_power *= panel_factors[data['Panel_Type'][i]]
            base_power *= (1 - (data['Cloud_Cover'][i] / 100) * 0.6)
            
            # Temperature efficiency loss
            temp_loss = max(0, (data['Panel_Temp'][i] - 25) * 0.0045)
            base_power *= (1 - temp_loss)
            
            # Wind cooling benefit
            base_power *= (1 + min(0.08, data['Wind_Speed'][i] * 0.012))
            
            # Degradation
            base_power *= (1 - data['Age_Years'][i] * 0.0065)
            
            # Tilt angle optimization (peak at ~30°)
            tilt_factor = 1 - abs(data['Tilt_Angle'][i] - 30) * 0.015
            base_power *= max(0.7, tilt_factor)
            
            # Noise
            base_power += np.random.normal(0, base_power * 0.08)
            power.append(max(0, base_power))
        
        data['Power_Output'] = power
        self.df = pd.DataFrame(data)
        
        # Introduce realistic missing values
        missing_idx = np.random.choice(self.df.index, 80, replace=False)
        self.df.loc[missing_idx[:40], 'Wind_Speed'] = np.nan
        self.df.loc[missing_idx[40:], 'Humidity'] = np.nan
        
        print(f"Dataset created: {self.df.shape[0]} records, {self.df.shape[1]} features")
        return self.df
    
    def preprocess_data(self):
        """Enhanced preprocessing with better encoding strategies"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Handle missing values
        self.df['Wind_Speed'].fillna(self.df['Wind_Speed'].median(), inplace=True)
        self.df['Humidity'].fillna(self.df['Humidity'].mean(), inplace=True)
        
        # Advanced categorical encoding
        # Weather: ordinal encoding based on solar impact
        weather_order = {'Sunny': 3, 'Partly_Cloudy': 2, 'Cloudy': 1, 'Overcast': 0}
        self.df['Weather_Encoded'] = self.df['Weather'].map(weather_order)
        
        # Panel type: based on efficiency
        panel_order = {'Mono': 2, 'Poly': 1, 'Thin': 0}
        self.df['Panel_Encoded'] = self.df['Panel_Type'].map(panel_order)
        
        # Feature engineering
        self.df['Temp_Diff'] = self.df['Panel_Temp'] - self.df['Ambient_Temp']
        self.df['Irradiance_Temp_Ratio'] = self.df['Solar_Irradiance'] / (self.df['Panel_Temp'] + 273.15)
        self.df['Cooling_Index'] = self.df['Wind_Speed'] / (self.df['Humidity'] / 100 + 0.1)
        
        # Select features
        self.feature_cols = ['Solar_Irradiance', 'Ambient_Temp', 'Panel_Temp', 'Wind_Speed', 
                           'Humidity', 'Cloud_Cover', 'Weather_Encoded', 'Panel_Encoded', 
                           'Age_Years', 'Tilt_Angle', 'Temp_Diff', 'Irradiance_Temp_Ratio', 
                           'Cooling_Index']
        
        self.X = self.df[self.feature_cols]
        self.y = self.df['Power_Output']
        
        print(f"Features: {len(self.feature_cols)}, Missing values handled")
        
    def advanced_feature_selection(self):
        """Multi-method feature selection"""
        print("\n=== FEATURE SELECTION ===")
        
        # Method 1: Statistical selection
        selector_stats = SelectKBest(f_regression, k=8)
        X_stats = selector_stats.fit_transform(self.X, self.y)
        stats_features = self.X.columns[selector_stats.get_support()]
        
        # Method 2: Recursive Feature Elimination
        rf_temp = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(rf_temp, n_features_to_select=8)
        rfe.fit(self.X, self.y)
        rfe_features = self.X.columns[rfe.support_]
        
        # Combine both methods
        selected_features = list(set(stats_features) | set(rfe_features))
        self.X_selected = self.X[selected_features]
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    
    def train_models(self):
        """Enhanced model training with pipelines"""
        print("\n=== MODEL TRAINING ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_selected, self.y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        # Enhanced model suite
        self.models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Decision_Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
            'SVR_RBF': SVR(kernel='rbf', C=100)
        }
        
        # Train and evaluate
        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring='r2', n_jobs=-1)
            
            # Fit and predict
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Metrics
            self.results[name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'predictions': y_pred_test,
                'model': model
            }
            
            print(f"{name:18} | CV R²: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | "
                  f"Test R²: {self.results[name]['test_r2']:.3f}")
    
    def hyperparameter_tuning(self):
        """Advanced hyperparameter optimization"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Best performing models for tuning
        param_grids = {
            'Random_Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'Gradient_Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        tuned_models = {}
        for model_name, params in param_grids.items():
            print(f"Tuning {model_name}...")
            
            if model_name == 'Random_Forest':
                base_model = RandomForestRegressor(random_state=42)
            else:
                base_model = GradientBoostingRegressor(random_state=42)
            
            grid_search = GridSearchCV(base_model, params, cv=5, scoring='r2', 
                                     n_jobs=-1, verbose=0)
            grid_search.fit(self.X_train, self.y_train)
            
            # Evaluate tuned model
            y_pred_tuned = grid_search.predict(self.X_test)
            tuned_r2 = r2_score(self.y_test, y_pred_tuned)
            tuned_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
            
            tuned_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'test_r2': tuned_r2,
                'test_rmse': tuned_rmse,
                'improvement': tuned_r2 - self.results[model_name]['test_r2']
            }
            
            print(f"  Best R²: {tuned_r2:.3f} (↑{tuned_models[model_name]['improvement']:+.3f})")
        
        # Select best overall model
        best_model_name = max(tuned_models.keys(), 
                            key=lambda x: tuned_models[x]['test_r2'])
        self.best_model = tuned_models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name} (R² = {self.best_model['test_r2']:.3f})")
        
    def create_visualizations(self):
        """Comprehensive visualization suite"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Solar Power ML Pipeline - Complete Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dataset distribution
        axes[0,0].hist(self.y, bins=40, alpha=0.7, color='gold', edgecolor='black')
        axes[0,0].set_title('Power Output Distribution')
        axes[0,0].set_xlabel('Power Output (kW)')
        
        # 2. Feature correlations
        corr_matrix = self.X_selected.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Feature Correlations')
        
        # 3. Model comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_r2'] for name in model_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        bars = axes[0,2].bar(range(len(model_names)), r2_scores, color=colors, alpha=0.8)
        axes[0,2].set_title('Model Performance (R²)')
        axes[0,2].set_xticks(range(len(model_names)))
        axes[0,2].set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=8)
        axes[0,2].set_ylabel('R² Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Best model predictions
        best_predictions = self.best_model['model'].predict(self.X_test)
        axes[1,0].scatter(self.y_test, best_predictions, alpha=0.6, color='darkblue')
        axes[1,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1,0].set_title(f'Actual vs Predicted\n({self.best_model_name})')
        axes[1,0].set_xlabel('Actual Power (kW)')
        axes[1,0].set_ylabel('Predicted Power (kW)')
        
        # 5. Residuals analysis
        residuals = self.y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6, color='green')
        axes[1,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1,1].set_title('Residuals Plot')
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('Residuals')
        
        # 6. Feature importance
        if hasattr(self.best_model['model'], 'feature_importances_'):
            importances = self.best_model['model'].feature_importances_
            feature_names = self.X_selected.columns
            
            indices = np.argsort(importances)[::-1][:8]  # Top 8 features
            
            axes[1,2].barh(range(len(indices)), importances[indices], color='purple', alpha=0.7)
            axes[1,2].set_yticks(range(len(indices)))
            axes[1,2].set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            axes[1,2].set_title('Feature Importance')
            axes[1,2].set_xlabel('Importance')
        else:
            axes[1,2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                          ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('solar_ml_complete_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Comprehensive final report"""
        print("\n" + "="*60)
        print("SOLAR POWER OUTPUT PREDICTION - ML PIPELINE REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Records: {len(self.df):,}")
        print(f"   • Original Features: {len(self.feature_cols)}")
        print(f"   • Selected Features: {len(self.X_selected.columns)}")
        print(f"   • Target Range: {self.y.min():.1f} - {self.y.max():.1f} kW")
        
        print(f"\n🏆 BEST MODEL PERFORMANCE:")
        print(f"   • Algorithm: {self.best_model_name.replace('_', ' ')}")
        print(f"   • Test R²: {self.best_model['test_r2']:.4f}")
        print(f"   • Test RMSE: {self.best_model['test_rmse']:.3f} kW")
        print(f"   • Prediction Accuracy: ~{self.best_model['test_r2']*100:.1f}%")
        
        print(f"\n📈 MODEL COMPARISON:")
        for name, results in self.results.items():
            print(f"   • {name:18}: R² = {results['test_r2']:.3f}, "
                  f"RMSE = {results['test_rmse']:.2f} kW")
        
        if hasattr(self.best_model['model'], 'feature_importances_'):
            importances = self.best_model['model'].feature_importances_
            feature_names = self.X_selected.columns
            top_features = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\n🔑 KEY FEATURES (Top 5):")
            for feature, importance in top_features:
                print(f"   • {feature:20}: {importance:.3f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Solar irradiance is the primary power predictor")
        print(f"   • Panel temperature significantly affects efficiency")
        print(f"   • Weather conditions create non-linear relationships")
        print(f"   • System age shows measurable degradation impact")
        
        print(f"\n🎯 BUSINESS APPLICATIONS:")
        print(f"   • Real-time grid load balancing")
        print(f"   • Energy storage optimization")
        print(f"   • Predictive maintenance scheduling")
        print(f"   • Solar farm performance monitoring")
        
        print(f"\n🚀 FUTURE ENHANCEMENTS:")
        print(f"   • Time-series forecasting integration")
        print(f"   • Deep learning for complex patterns")
        print(f"   • Real-time weather API integration")
        print(f"   • Multi-location ensemble modeling")
        
        print("\n" + "="*60)
        print("✅ ML PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

def main():
    """Execute complete ML pipeline"""
    pipeline = SolarPowerMLPipeline(n_samples=1500)
    
    # Execute pipeline
    pipeline.create_dataset()
    pipeline.preprocess_data()
    pipeline.advanced_feature_selection()
    pipeline.train_models()
    pipeline.hyperparameter_tuning()
    pipeline.create_visualizations()
    pipeline.generate_report()
    
    return pipeline

# Execute pipeline
if __name__ == "__main__":
    solar_pipeline = main()