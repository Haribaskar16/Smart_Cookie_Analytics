"""
Machine Learning Models for Smart Cookie Analytics
====================================================

This module provides machine learning models for:
- User behavior prediction
- Cookie performance analysis
- Customer segmentation
- Recommendation systems

Suitable for educational purposes with clear, simplified implementations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Utility class for evaluating machine learning models.
    Provides methods for metrics calculation and visualization.
    """
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model (for logging)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"\n{'='*50}")
        print(f"{model_name} - Classification Metrics")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def evaluate_regression(y_true, y_pred, model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model (for logging)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2-Score': r2_score(y_true, y_pred)
        }
        
        print(f"\n{'='*50}")
        print(f"{model_name} - Regression Metrics")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix for classification models.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Title for the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, title: str = "Predictions vs Actual"):
        """
        Plot actual vs predicted values for regression models.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Title for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()


class UserBehaviorPredictor:
    """
    Predicts user behavior based on cookie analytics data.
    Predicts if a user will make a purchase or continue browsing.
    """
    
    def __init__(self):
        """Initialize the behavior predictor model."""
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the user behavior prediction model.
        
        Args:
            X: Features DataFrame (session_duration, pages_visited, click_count, etc.)
            y: Target Series (purchase: 1, no_purchase: 0)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n[INFO] Training User Behavior Predictor...")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Evaluate
        train_metrics = ModelEvaluator.evaluate_classification(y_train, y_pred_train, "Training Set")
        test_metrics = ModelEvaluator.evaluate_classification(y_test, y_pred_test, "Test Set")
        
        self.is_trained = True
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict user behavior for new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions (1 for purchase, 0 for no purchase)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions for user behavior.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_[0]
        }).sort_values('Coefficient', ascending=False)
        
        return importance


class CookiePerformanceAnalyzer:
    """
    Analyzes and predicts cookie performance metrics.
    Predicts click-through rates, conversion rates, and engagement scores.
    """
    
    def __init__(self):
        """Initialize the cookie performance analyzer."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the cookie performance model.
        
        Args:
            X: Features DataFrame
            y: Target Series (performance metric)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n[INFO] Training Cookie Performance Analyzer...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model (RandomForest doesn't require scaling, but we'll do it for consistency)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluate
        train_metrics = ModelEvaluator.evaluate_regression(y_train, y_pred_train, "Training Set")
        test_metrics = ModelEvaluator.evaluate_regression(y_test, y_pred_test, "Test Set")
        
        self.is_trained = True
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cookie performance for new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Performance predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the Random Forest model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance


class CustomerSegmentationModel:
    """
    Segments customers based on their behavior and engagement patterns.
    Uses clustering to identify customer groups (e.g., loyal, at-risk, potential).
    """
    
    def __init__(self, n_clusters: int = 3):
        """
        Initialize the customer segmentation model.
        
        Args:
            n_clusters: Number of customer segments to create
        """
        from sklearn.cluster import KMeans
        
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the customer segmentation model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary containing training results
        """
        print("\n[INFO] Training Customer Segmentation Model...")
        
        self.feature_names = X.columns.tolist()
        
        # Standardize features (important for KMeans)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Get cluster assignments
        clusters = self.model.labels_
        
        print(f"[INFO] Customers segmented into {self.n_clusters} clusters")
        print("\nCluster Distribution:")
        for i in range(self.n_clusters):
            count = np.sum(clusters == i)
            percentage = (count / len(clusters)) * 100
            print(f"  Cluster {i}: {count} customers ({percentage:.1f}%)")
        
        self.is_trained = True
        
        return {
            'clusters': clusters,
            'cluster_centers': self.model.cluster_centers_,
            'inertia': self.model.inertia_
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Assign new customers to segments.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Cluster assignments
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_cluster_profiles(self, X: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
        """
        Create profiles for each cluster showing average feature values.
        
        Args:
            X: Features DataFrame
            clusters: Cluster assignments
            
        Returns:
            DataFrame with cluster profiles
        """
        X_copy = X.copy()
        X_copy['Cluster'] = clusters
        
        profiles = X_copy.groupby('Cluster').mean()
        
        print("\nCluster Profiles (Average Feature Values):")
        print(profiles.round(3))
        
        return profiles


class RecommendationSystem:
    """
    Generates product/cookie recommendations based on user similarity.
    Uses collaborative filtering approach with simplified implementation.
    """
    
    def __init__(self):
        """Initialize the recommendation system."""
        self.user_item_matrix = None
        self.is_trained = False
    
    def train(self, user_item_data: pd.DataFrame) -> None:
        """
        Train the recommendation system.
        
        Args:
            user_item_data: DataFrame with columns [user_id, item_id, rating/interaction]
        """
        print("\n[INFO] Training Recommendation System...")
        
        # Create user-item interaction matrix
        self.user_item_matrix = user_item_data.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        print(f"[INFO] User-Item matrix shape: {self.user_item_matrix.shape}")
        self.is_trained = True
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific user.
        
        Args:
            user_id: The user ID to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended items with scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in training data")
        
        # Get the user's preferences
        user_prefs = self.user_item_matrix.loc[user_id]
        
        # Items the user hasn't interacted with
        unrated_items = user_prefs[user_prefs == 0].index.tolist()
        
        # Calculate similarity with other users
        user_similarity = self.user_item_matrix.corrwith(user_prefs, method='pearson')
        similar_users = user_similarity.nlargest(10)[1:]  # Exclude the user themselves
        
        # Calculate recommendations
        recommendations = {}
        for item in unrated_items:
            item_ratings = self.user_item_matrix[item]
            # Weight ratings by similarity
            weighted_sum = (item_ratings * similar_users).sum()
            similarity_sum = similar_users.sum()
            
            if similarity_sum > 0:
                recommendation_score = weighted_sum / similarity_sum
                recommendations[item] = recommendation_score
        
        # Sort and return top N
        top_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return [
            {'item_id': item, 'score': float(score)}
            for item, score in top_recommendations
        ]


class ModelPipeline:
    """
    Complete machine learning pipeline integrating all models.
    Manages data preprocessing, training, and evaluation.
    """
    
    def __init__(self):
        """Initialize the model pipeline."""
        self.behavior_predictor = UserBehaviorPredictor()
        self.performance_analyzer = CookiePerformanceAnalyzer()
        self.segmentation_model = CustomerSegmentationModel(n_clusters=3)
        self.recommendation_system = RecommendationSystem()
        self.models_trained = []
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for model training.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n[INFO] Preprocessing data...")
        
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        
        # Remove duplicates
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        print(f"[INFO] Removed {initial_rows - len(df_processed)} duplicate rows")
        
        # Handle categorical variables if present
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'user_id' and col != 'item_id':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        
        print(f"[INFO] Preprocessed data shape: {df_processed.shape}")
        
        return df_processed
    
    def train_all_models(self, behavior_data: pd.DataFrame = None,
                        performance_data: pd.DataFrame = None,
                        customer_data: pd.DataFrame = None,
                        recommendation_data: pd.DataFrame = None) -> Dict[str, Dict]:
        """
        Train all models in the pipeline.
        
        Args:
            behavior_data: DataFrame for behavior prediction (must include target 'purchase')
            performance_data: DataFrame for performance analysis (must include target)
            customer_data: DataFrame for customer segmentation
            recommendation_data: DataFrame for recommendations (user_id, item_id, rating)
            
        Returns:
            Dictionary containing training results for all models
        """
        results = {}
        
        # Train behavior predictor
        if behavior_data is not None:
            print("\n" + "="*60)
            print("TRAINING USER BEHAVIOR PREDICTOR")
            print("="*60)
            
            behavior_data = self.preprocess_data(behavior_data)
            X_behavior = behavior_data.drop('purchase', axis=1)
            y_behavior = behavior_data['purchase']
            
            results['behavior'] = self.behavior_predictor.train(X_behavior, y_behavior)
            self.models_trained.append('behavior')
        
        # Train performance analyzer
        if performance_data is not None:
            print("\n" + "="*60)
            print("TRAINING COOKIE PERFORMANCE ANALYZER")
            print("="*60)
            
            performance_data = self.preprocess_data(performance_data)
            # Assume last column is target
            X_performance = performance_data.iloc[:, :-1]
            y_performance = performance_data.iloc[:, -1]
            
            results['performance'] = self.performance_analyzer.train(X_performance, y_performance)
            self.models_trained.append('performance')
        
        # Train segmentation model
        if customer_data is not None:
            print("\n" + "="*60)
            print("TRAINING CUSTOMER SEGMENTATION MODEL")
            print("="*60)
            
            customer_data = self.preprocess_data(customer_data)
            results['segmentation'] = self.segmentation_model.train(customer_data)
            self.models_trained.append('segmentation')
        
        # Train recommendation system
        if recommendation_data is not None:
            print("\n" + "="*60)
            print("TRAINING RECOMMENDATION SYSTEM")
            print("="*60)
            
            self.recommendation_system.train(recommendation_data)
            self.models_trained.append('recommendation')
        
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE - {len(self.models_trained)} models trained")
        print("="*60)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("Smart Cookie Analytics - Machine Learning Models")
    print("=" * 60)
    
    # Example: Create sample data and train models
    np.random.seed(42)
    
    # Sample behavior prediction data
    n_samples = 500
    behavior_data = pd.DataFrame({
        'session_duration': np.random.uniform(10, 300, n_samples),
        'pages_visited': np.random.randint(1, 20, n_samples),
        'click_count': np.random.randint(0, 50, n_samples),
        'time_on_site': np.random.uniform(1, 60, n_samples),
        'purchase': np.random.randint(0, 2, n_samples)
    })
    
    # Initialize and train behavior predictor
    predictor = UserBehaviorPredictor()
    X = behavior_data.drop('purchase', axis=1)
    y = behavior_data['purchase']
    results = predictor.train(X, y)
    
    print("\nFeature Importance:")
    print(predictor.get_feature_importance())
    
    print("\n[SUCCESS] Models initialized and ready for use!")
