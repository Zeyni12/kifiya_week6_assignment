import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
class FeatureEngineer:
    def __init__(self, scaling_method='minmax'):
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaling method. Use 'minmax' or 'standard'.")

        self.label_encoder = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=5)

    def preprocess_time_features(self, df):
        # Ensure TransactionStartTime is datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Extract Features from TransactionStartTime
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        return df

    def aggregate_features(self, df):
        # Aggregate Features (Group by CustomerId)
        agg_features = df.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),
            avg_transaction_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_transaction_amount=('Amount', 'std')
        ).reset_index()

        return agg_features

    def encode_features(self, df):
        # One-Hot Encoding for Categorical Variables
        df = pd.get_dummies(df, columns=['ChannelId', 'CurrencyCode', 'ProductCategory','PricingStrategy'], drop_first=True)

        # Label Encoding for ProviderId and Pricing strategy
        df['ProviderId'] = self.label_encoder.fit_transform(df['ProviderId'])
        df['CountryCode'] = self.label_encoder.fit_transform(df['CountryCode'])

        return df

    def handle_missing_values(self, df):
        # Imputation for numerical columns
        df[['Amount', 'Value']] = self.imputer.fit_transform(df[['Amount', 'Value']])
        return df

    def scale_features(self, df):
        # Normalize/Standardize Numerical Features
        df[['Amount', 'Value']] = self.scaler.fit_transform(df[['Amount', 'Value']])
        return df
    def dimensionality_reduction(df, n_components=5):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(df[numerical_cols])
    
        # Add PCA features back to DataFrame
        pca_df = pd.DataFrame(pca_features, columns=[f'PCA_{i+1}' for i in range(n_components)])
        df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

        return df

    def feature_selection(df, target_column, num_features=10):
                        
                       # """
                #Select top features using SelectKBest with ANOVA F-value.
                   #  """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Apply SelectKBest
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X_new = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        df_selected = df[selected_features]

        return df_selected

    def fit_transform(self, df):

        # Step 1: Preprocess time-based features
        df = self.preprocess_time_features(df)

        # Step 2: Aggregate features
        agg_features = self.aggregate_features(df)

        # Step 3: Encode categorical features
        df = self.encode_features(df)

        # Step 4: Handle missing values
        df = self.handle_missing_values(df)

        # Step 5: Scale numerical features
        df = self.scale_features(df)

        # Step 6: Merge aggregated features back to the main dataframe
        df = pd.merge(df, agg_features, on='CustomerId', how='left')

        return df

# Example usage:
# from feature_engineering import FeatureEngineer
# df = pd.read_csv('your_dataset.csv')
# fe = FeatureEngineer(scaling_method='minmax')
# df_cleaned = fe.fit_transform(df)
