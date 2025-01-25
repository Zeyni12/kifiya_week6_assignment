import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

class EDA:
    def __init__(self, df):
        """
        Initialize the EDA class with a DataFrame.
        :param df: pandas DataFrame
        """
        self.df = df

    def univariate_analysis(self, column, plot_type='histogram', bins=30):
        """
        Perform univariate analysis for a given column.
        :param column: Column name for analysis
        :param plot_type: Type of plot ('histogram', 'boxplot', 'kde')
        :param bins: Number of bins for histogram
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in the DataFrame.")
        
        plt.figure(figsize=(6, 3))
        if plot_type == 'histogram':
            self.df[column].hist(bins=bins, edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        elif plot_type == 'boxplot':
            sns.boxplot(x=self.df[column])
            plt.title(f'Boxplot of {column}')
        elif plot_type == 'kde':
            sns.kdeplot(self.df[column], shade=True)
            plt.title(f'Density Plot of {column}')
        else:
            raise ValueError("Invalid plot_type. Use 'histogram', 'boxplot', or 'kde'.")
        plt.show()

    def multivariate_analysis(self, feature1, feature2, plot_type='scatter'):
        """
        Perform multivariate analysis for two features.
        :param feature1: First feature (x-axis)
        :param feature2: Second feature (y-axis)
        :param plot_type: Type of plot ('scatter', 'correlation')
        """
        if feature1 not in self.df.columns or feature2 not in self.df.columns:
            raise ValueError("One or both columns not found in the DataFrame.")
        
        plt.figure(figsize=(6, 3))
        if plot_type == 'scatter':
            sns.scatterplot(x=self.df[feature1], y=self.df[feature2])
            plt.title(f'Scatter Plot: {feature1} vs {feature2}')
        elif plot_type == 'correlation':
            corr = self.df[[feature1, feature2]].corr().iloc[0, 1]
            sns.heatmap(self.df[[feature1, feature2]].corr(), annot=True, cmap='coolwarm', cbar=True)
            plt.title(f'Correlation: {corr:.2f}')
        else:
            raise ValueError("Invalid plot_type. Use 'scatter' or 'correlation'.")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def categorical_analysis(self, column):
        """
        Perform categorical analysis for a given column.
        :param column: Categorical column name
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in the DataFrame.")
        
        if self.df[column].dtype == 'object' or self.df[column].nunique() < 20:
            plt.figure(figsize=(6, 3))
            sns.countplot(x=self.df[column], order=self.df[column].value_counts().index)
            plt.title(f'Count Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()
        else:
            raise ValueError(f"Column {column} does not appear to be categorical.")

    def detect_outliers(self, column, method='zscore', threshold=3):
        """
        Detect outliers in a given column.
        :param column: Column name for outlier detection
        :param method: Method for detection ('zscore', 'iqr')
        :param threshold: Threshold for outlier detection
        :return: DataFrame with outliers
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in the DataFrame.")
        
        if method == 'zscore':
            self.df['zscore'] = zscore(self.df[column])
            outliers = self.df[abs(self.df['zscore']) > threshold]
            self.df.drop(columns=['zscore'], inplace=True)
        elif method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        else:
            raise ValueError("Invalid method. Use 'zscore' or 'iqr'.")
        
        return outliers

    def summary(self):
        """
        Provide a quick summary of the dataset.
        """
        print("Dataset Summary:")
        print(f"Shape: {self.df.shape}")
        print("\nData Types:")
        print(self.df.dtypes)
        print('\nDuplicate values')
        print(self.df.duplicated())
        print("\nMissing Values:")
        print(self.df.isnull().sum())  
        print("\nBasic Statistics (Numerical Data Only):")
        print(self.df.select_dtypes(include=['number']).describe()) 

    def dispersion_analysis(self):
        """
        Calculate and display dispersion metrics for the numerical columns in the dataset.
        """
        numeric_data = self.df.select_dtypes(include=['number'])

        # Range
        range_val = numeric_data.max() - numeric_data.min()
        print("\nRange:\n", range_val)

        # Variance
        variance = numeric_data.var()
        print("\nVariance:\n", variance)

        # Standard Deviation
        std_dev = numeric_data.std()
        print("\nStandard Deviation:\n", std_dev)

        # Interquartile Range (IQR)
        q1 = numeric_data.quantile(0.25)
        q3 = numeric_data.quantile(0.75)
        iqr = q3 - q1
        print("\nInterquartile Range (IQR):\n", iqr)    

