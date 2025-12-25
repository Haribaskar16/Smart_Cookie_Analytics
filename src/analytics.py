"""
Analytics Module for Smart Cookie Analytics
Provides statistical analysis, correlation analysis, and visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Performs statistical analysis on cookie data"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Statistical Analyzer
        
        Args:
            data: DataFrame containing cookie analytics data
        """
        self.data = data
        self.summary_stats = None
    
    def calculate_summary_statistics(self) -> Dict[str, Dict]:
        """
        Calculate summary statistics for numerical columns
        
        Returns:
            Dictionary containing mean, median, std, min, max, quartiles
        """
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numerical_cols:
            summary[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'q25': self.data[col].quantile(0.25),
                'q75': self.data[col].quantile(0.75),
                'variance': self.data[col].var(),
                'skewness': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis()
            }
        
        self.summary_stats = summary
        return summary
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> np.ndarray:
        """
        Detect outliers using IQR or Z-score method
        
        Args:
            column: Column name to check for outliers
            method: 'iqr' or 'zscore'
        
        Returns:
            Boolean array indicating outliers
        """
        if method == 'iqr':
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (self.data[column] < Q1 - 1.5 * IQR) | (self.data[column] > Q3 + 1.5 * IQR)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.data[column].dropna()))
            outliers = z_scores > 3
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return outliers.values
    
    def perform_hypothesis_test(self, column: str, test_type: str = 'ttest', 
                               expected_mean: float = None) -> Dict:
        """
        Perform hypothesis test on a column
        
        Args:
            column: Column name for testing
            test_type: 'ttest', 'mannwhitneyu', 'normaltest'
            expected_mean: Expected mean for t-test
        
        Returns:
            Dictionary with test statistic and p-value
        """
        results = {}
        
        if test_type == 'ttest' and expected_mean is not None:
            stat, pvalue = stats.ttest_1samp(self.data[column].dropna(), expected_mean)
            results = {
                'test': 'One-Sample T-Test',
                'statistic': stat,
                'p_value': pvalue,
                'significant': pvalue < 0.05
            }
        elif test_type == 'normaltest':
            stat, pvalue = stats.normaltest(self.data[column].dropna())
            results = {
                'test': 'Normality Test',
                'statistic': stat,
                'p_value': pvalue,
                'is_normal': pvalue > 0.05
            }
        elif test_type == 'mannwhitneyu':
            # For two groups - split by median
            median = self.data[column].median()
            group1 = self.data[self.data[column] <= median][column]
            group2 = self.data[self.data[column] > median][column]
            stat, pvalue = stats.mannwhitneyu(group1, group2)
            results = {
                'test': 'Mann-Whitney U Test',
                'statistic': stat,
                'p_value': pvalue,
                'significant': pvalue < 0.05
            }
        
        return results
    
    def get_distribution_info(self, column: str) -> Dict:
        """Get distribution information for a column"""
        return {
            'column': column,
            'mean': self.data[column].mean(),
            'median': self.data[column].median(),
            'mode': self.data[column].mode()[0] if len(self.data[column].mode()) > 0 else None,
            'std_dev': self.data[column].std(),
            'variance': self.data[column].var(),
            'range': self.data[column].max() - self.data[column].min(),
            'iqr': self.data[column].quantile(0.75) - self.data[column].quantile(0.25),
            'skewness': self.data[column].skew(),
            'kurtosis': self.data[column].kurtosis(),
        }


class CorrelationAnalyzer:
    """Performs correlation and relationship analysis"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Correlation Analyzer
        
        Args:
            data: DataFrame containing analytics data
        """
        self.data = data
        self.correlation_matrix = None
    
    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            method: 'pearson', 'spearman', or 'kendall'
        
        Returns:
            Correlation matrix
        """
        numerical_data = self.data.select_dtypes(include=[np.number])
        self.correlation_matrix = numerical_data.corr(method=method)
        return self.correlation_matrix
    
    def find_strong_correlations(self, threshold: float = 0.7) -> List[Tuple]:
        """
        Find strongly correlated pairs above threshold
        
        Args:
            threshold: Correlation threshold (0-1)
        
        Returns:
            List of (column1, column2, correlation) tuples
        """
        if self.correlation_matrix is None:
            self.calculate_correlation()
        
        strong_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_pairs.append((
                        self.correlation_matrix.columns[i],
                        self.correlation_matrix.columns[j],
                        corr_value
                    ))
        
        return sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def calculate_partial_correlation(self, x: str, y: str, control_vars: List[str]) -> float:
        """
        Calculate partial correlation between x and y controlling for other variables
        
        Args:
            x: First variable
            y: Second variable
            control_vars: Variables to control for
        
        Returns:
            Partial correlation coefficient
        """
        from sklearn.linear_model import LinearRegression
        
        # Residuals of x
        X_control = self.data[control_vars].values
        y_x = self.data[x].values
        model_x = LinearRegression().fit(X_control, y_x)
        residuals_x = y_x - model_x.predict(X_control)
        
        # Residuals of y
        y_y = self.data[y].values
        model_y = LinearRegression().fit(X_control, y_y)
        residuals_y = y_y - model_y.predict(X_control)
        
        # Correlation of residuals
        partial_corr = np.corrcoef(residuals_x, residuals_y)[0, 1]
        return partial_corr
    
    def analyze_categorical_relationship(self, cat_col: str, num_col: str) -> Dict:
        """
        Analyze relationship between categorical and numerical variables
        
        Args:
            cat_col: Categorical column name
            num_col: Numerical column name
        
        Returns:
            Dictionary with statistics by category
        """
        grouped = self.data.groupby(cat_col)[num_col].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ])
        
        return grouped.to_dict('index')


class VisualizationEngine:
    """Handles visualization of analytics data"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize Visualization Engine
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_distribution(self, data: pd.Series, title: str = None, 
                         bins: int = 30, kde: bool = True) -> plt.Figure:
        """
        Plot distribution of a variable
        
        Args:
            data: Series to plot
            title: Plot title
            bins: Number of bins for histogram
            kde: Whether to show KDE curve
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data.dropna(), bins=bins, alpha=0.7, edgecolor='black')
        if kde:
            data.dropna().plot(kind='kde', ax=ax, secondary_y=False, 
                             linewidth=2, color='red')
        
        ax.set_title(title or f'Distribution of {data.name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                figsize: Tuple = (12, 10),
                                annot: bool = True) -> plt.Figure:
        """
        Plot correlation matrix as heatmap
        
        Args:
            corr_matrix: Correlation matrix
            figsize: Figure size
            annot: Whether to annotate cells
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr_matrix, annot=annot, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_scatter(self, x: pd.Series, y: pd.Series, title: str = None,
                    hue: Optional[pd.Series] = None, figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create scatter plot
        
        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            hue: Series for coloring points
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if hue is not None:
            scatter = ax.scatter(x, y, c=hue, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
            plt.colorbar(scatter, ax=ax, label='Category')
        else:
            ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel(x.name, fontsize=12)
        ax.set_ylabel(y.name, fontsize=12)
        ax.set_title(title or f'{x.name} vs {y.name}', fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(x.dropna(), y.dropna(), 1)
        p = np.poly1d(z)
        ax.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        return fig
    
    def plot_box_plot(self, data: pd.DataFrame, x: str, y: str,
                     figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create box plot
        
        Args:
            data: DataFrame
            x: Categorical column
            y: Numerical column
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.boxplot(data=data, x=x, y=y, ax=ax, palette='Set2')
        
        ax.set_title(f'Box Plot: {y} by {x}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, dates: pd.Series, values: pd.Series,
                        title: str = None, figsize: Tuple = (14, 6)) -> plt.Figure:
        """
        Plot time series data
        
        Args:
            dates: DateTime series
            values: Values series
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(dates, values, linewidth=2, marker='o', markersize=4)
        ax.fill_between(dates, values, alpha=0.3)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or 'Time Series Analysis', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_multi_distribution(self, data: pd.DataFrame, columns: List[str],
                               figsize: Tuple = (15, 10)) -> plt.Figure:
        """
        Plot distributions of multiple columns
        
        Args:
            data: DataFrame
            columns: List of column names to plot
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            axes[idx].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig


class AnalyticsPipeline:
    """Orchestrates the complete analytics workflow"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Analytics Pipeline
        
        Args:
            data: DataFrame containing analytics data
        """
        self.data = data
        self.stats_analyzer = StatisticalAnalyzer(data)
        self.corr_analyzer = CorrelationAnalyzer(data)
        self.viz_engine = VisualizationEngine()
    
    def generate_full_report(self) -> Dict:
        """
        Generate comprehensive analytics report
        
        Returns:
            Dictionary containing all analysis results
        """
        report = {
            'summary_statistics': self.stats_analyzer.calculate_summary_statistics(),
            'correlation_matrix': self.corr_analyzer.calculate_correlation().to_dict(),
            'strong_correlations': self.corr_analyzer.find_strong_correlations(),
            'data_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.astype(str).to_dict()
        }
        
        return report
    
    def export_visualizations(self, output_dir: str, 
                            numerical_cols: Optional[List[str]] = None) -> None:
        """
        Export all visualizations to directory
        
        Args:
            output_dir: Directory to save visualizations
            numerical_cols: List of numerical columns to visualize
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        if numerical_cols is None:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Correlation heatmap
        corr_matrix = self.corr_analyzer.calculate_correlation()
        fig = self.viz_engine.plot_correlation_heatmap(corr_matrix)
        fig.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Individual distributions
        for col in numerical_cols[:5]:  # Limit to first 5
            fig = self.viz_engine.plot_distribution(self.data[col], title=f'Distribution of {col}')
            fig.savefig(f'{output_dir}/distribution_{col}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    # Example usage
    print("Smart Cookie Analytics Module Loaded Successfully")
