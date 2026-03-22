# modules/data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import requests
import io
import warnings

warnings.filterwarnings('ignore')


class HealthcareDataPipeline:
    """Clean pipeline for healthcare data integration"""

    def __init__(self):
        self.real_datasets = {
            'heart_disease': self.load_heart_disease,
            'diabetes': self.load_diabetes_data,
            'cancer': self.load_breast_cancer,
            'synthetic': self.generate_synthetic_healthcare
        }

    def load_heart_disease(self):
        """Load UCI Heart Disease dataset"""
        try:
            # Try to load from UCI ML repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]

            df = pd.read_csv(url, names=columns, na_values='?')

            # Clean and preprocess
            df = df.dropna()
            df['target'] = (df['target'] > 0).astype(int)  # Binary classification

            # Feature engineering
            df['bmi'] = np.random.normal(27, 5, len(df))  # Synthetic BMI
            df['income_level'] = np.random.uniform(0, 1, len(df))

            return df

        except:
            # Fallback to built-in dataset
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['age'] = np.random.normal(55, 15, len(df))
            df['income_level'] = np.random.uniform(0, 1, len(df))

            return df

    def load_diabetes_data(self):
        """Load diabetes dataset"""
        try:
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            columns = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi',
                       'pedigree', 'age', 'target']
            df = pd.read_csv(url, names=columns)

            # Add synthetic features
            df['income_level'] = np.random.uniform(0, 1, len(df))
            df['access_score'] = np.random.uniform(0.3, 1, len(df))

            return df

        except:
            return self.generate_synthetic_healthcare(n_samples=1000)

    def load_breast_cancer(self):
        """Load breast cancer dataset"""
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # Add demographic and socioeconomic features
        df['age'] = np.random.normal(55, 15, len(df))
        df['income_level'] = np.random.uniform(0, 1, len(df))
        df['insurance'] = np.random.choice([0, 1], len(df), p=[0.2, 0.8])
        df['education'] = np.random.choice([1, 2, 3, 4], len(df), p=[0.1, 0.3, 0.4, 0.2])

        return df

    def generate_synthetic_healthcare(self, n_samples=1000):
        """Generate comprehensive synthetic healthcare data"""
        np.random.seed(42)

        data = {
            'age': np.random.normal(55, 15, n_samples).clip(18, 100),
            'sex': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'bmi': np.random.normal(27, 6, n_samples).clip(15, 50),
            'blood_pressure': np.random.normal(130, 20, n_samples).clip(80, 200),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(100, 350),
            'glucose': np.random.normal(110, 30, n_samples).clip(60, 300),
            'chronic_conditions': np.random.poisson(1.5, n_samples).clip(0, 8),
            'previous_hospitalizations': np.random.poisson(0.8, n_samples),
            'smoking': np.random.binomial(1, 0.25, n_samples),
            'alcohol': np.random.binomial(1, 0.15, n_samples),
            'exercise': np.random.uniform(0, 1, n_samples),
            'income_level': np.random.uniform(0, 1, n_samples),
            'education': np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.35, 0.35, 0.15]),
            'insurance': np.random.binomial(1, 0.8, n_samples),
            'access_score': np.random.uniform(0.3, 1, n_samples),
            'region': np.random.choice(['urban', 'suburban', 'rural'], n_samples, p=[0.5, 0.3, 0.2]),
            'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.6, 0.2, 0.15, 0.05])
        }

        # Generate target based on risk factors
        risk_score = (
                data['age'] / 100 * 0.2 +
                (data['bmi'] - 25) / 25 * 0.15 +
                (data['blood_pressure'] - 120) / 80 * 0.15 +
                (data['cholesterol'] - 200) / 150 * 0.1 +
                data['chronic_conditions'] / 8 * 0.2 +
                (1 - data['exercise']) * 0.1 +
                data['smoking'] * 0.05 +
                (1 - data['income_level']) * 0.05
        )

        data['target'] = (risk_score + np.random.normal(0, 0.1, n_samples) > np.percentile(risk_score, 60)).astype(int)

        return pd.DataFrame(data)

    def preprocess_data(self, df, target_column='target'):
        """Clean and preprocess the data"""
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y if y is not None else None

    def augment_with_bias(self, df, bias_config):
        """Augment data with specified biases"""
        df_augmented = df.copy()

        # Apply socioeconomic bias
        if 'socioeconomic' in bias_config:
            low_income_mask = df_augmented['income_level'] < 0.3
            df_augmented.loc[low_income_mask, 'access_score'] *= 0.7
            df_augmented.loc[low_income_mask, 'education'] = np.maximum(1, df_augmented.loc[
                low_income_mask, 'education'] - 1)

        # Apply demographic bias
        if 'demographic' in bias_config:
            # Example: bias against certain ethnicities
            if 'ethnicity' in df_augmented.columns:
                minority_mask = df_augmented['ethnicity'].isin(['C', 'D'])
                df_augmented.loc[minority_mask, 'access_score'] *= 0.8

        # Apply geographic bias
        if 'geographic' in bias_config:
            if 'region' in df_augmented.columns:
                rural_mask = df_augmented['region'] == 'rural'
                df_augmented.loc[rural_mask, 'access_score'] *= 0.6

        return df_augmented

    def create_patient_groups(self, df, group_type='income'):
        """Create patient groups for fairness analysis"""
        if group_type == 'income':
            groups = (df['income_level'] < 0.3).astype(int)  # 0 = low income, 1 = higher income
        elif group_type == 'insurance':
            groups = (df['insurance'] == 0).astype(int)  # 0 = uninsured, 1 = insured
        elif group_type == 'education':
            groups = (df['education'] <= 2).astype(int)  # 0 = low education, 1 = higher education
        else:
            groups = np.zeros(len(df))

        return groups

    def get_dataset(self, dataset_name='hybrid', bias_config=None, n_samples=1000):
        """Main method to get processed dataset"""
        if dataset_name == 'hybrid':
            # Blend real and synthetic data
            try:
                real_df = self.load_heart_disease()
                synthetic_df = self.generate_synthetic_healthcare(n_samples=len(real_df))

                # Combine features
                combined_df = pd.concat([real_df, synthetic_df], axis=1)
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

            except:
                combined_df = self.generate_synthetic_healthcare(n_samples=n_samples)
        else:
            if dataset_name in self.real_datasets:
                combined_df = self.real_datasets[dataset_name]()
            else:
                combined_df = self.generate_synthetic_healthcare(n_samples=n_samples)

        # Apply biases if specified
        if bias_config:
            combined_df = self.augment_with_bias(combined_df, bias_config)

        return combined_df