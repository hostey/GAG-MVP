# modules/healthcare_equity.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class HealthcareEquityAnalyzer:
    """Clean analyzer for healthcare equity"""

    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline
        self.models = {}
        self.results = {}

    def prepare_data(self, dataset_name='hybrid', bias_types=None, n_samples=5000):
        """Prepare data for analysis"""
        bias_config = bias_types if bias_types else []
        df = self.data_pipeline.get_dataset(dataset_name, bias_config, n_samples)

        # Preprocess
        X, y = self.data_pipeline.preprocess_data(df)

        # Create patient groups
        income_groups = self.data_pipeline.create_patient_groups(df, 'income')
        insurance_groups = self.data_pipeline.create_patient_groups(df, 'insurance')

        return X, y, {'income': income_groups, 'insurance': insurance_groups}, df

    def train_model(self, X_train, y_train, model_type='rf'):
        """Train healthcare prediction model"""
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        return model

    def calculate_fairness_metrics(self, y_true, y_pred, groups):
        """Calculate comprehensive fairness metrics"""
        metrics = {}

        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (
                                                                                                                                   metrics[
                                                                                                                                       'precision'] +
                                                                                                                                   metrics[
                                                                                                                                       'recall']) > 0 else 0

        # Group-wise metrics
        group_metrics = {}
        for group_name, group_values in groups.items():
            unique_groups = np.unique(group_values)
            group_metrics[group_name] = {}

            for g in unique_groups:
                mask = group_values == g
                if np.sum(mask) > 0:
                    group_true = y_true[mask]
                    group_pred = y_pred[mask]

                    group_metrics[group_name][g] = {
                        'accuracy': accuracy_score(group_true, group_pred),
                        'precision': precision_score(group_true, group_pred, zero_division=0),
                        'recall': recall_score(group_true, group_pred, zero_division=0),
                        'prevalence': np.mean(group_true),
                        'detection_rate': np.mean(group_pred),
                        'sample_size': np.sum(mask)
                    }

        # Fairness disparities
        fairness_disparities = {}
        for group_name, group_metric in group_metrics.items():
            if len(group_metric) >= 2:
                # Demographic parity
                detection_rates = [m['detection_rate'] for m in group_metric.values()]
                fairness_disparities[f'{group_name}_demographic_parity'] = max(detection_rates) - min(detection_rates)

                # Equal opportunity (recall disparity)
                recalls = [m['recall'] for m in group_metric.values()]
                fairness_disparities[f'{group_name}_equal_opportunity'] = max(recalls) - min(recalls)

                # Equalized odds
                precisions = [m['precision'] for m in group_metric.values()]
                fairness_disparities[f'{group_name}_equalized_odds'] = max(
                    max(precisions) - min(precisions),
                    max(recalls) - min(recalls)
                )

        metrics['group_metrics'] = group_metrics
        metrics['fairness_disparities'] = fairness_disparities

        # Overall fairness score
        if fairness_disparities:
            max_disparity = max(fairness_disparities.values())
            metrics['fairness_score'] = 1.0 - min(1.0, max_disparity * 2)
        else:
            metrics['fairness_score'] = 0.5

        return metrics

    def analyze_health_equity(self, dataset_params=None, model_params=None):
        """Main analysis pipeline"""
        # Default parameters
        if dataset_params is None:
            dataset_params = {
                'dataset_name': 'hybrid',
                'bias_types': ['socioeconomic', 'demographic'],
                'n_samples': 5000
            }

        if model_params is None:
            model_params = {
                'model_type': 'rf',
                'test_size': 0.3,
                'random_state': 42
            }

        # Prepare data
        X, y, groups, raw_df = self.prepare_data(
            dataset_name=dataset_params['dataset_name'],
            bias_types=dataset_params['bias_types'],
            n_samples=dataset_params['n_samples']
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=model_params['test_size'],
            random_state=model_params['random_state'],
            stratify=y
        )

        # Also split groups
        income_groups_train, income_groups_test = train_test_split(
            groups['income'],
            test_size=model_params['test_size'],
            random_state=model_params['random_state'],
            stratify=y
        )

        # Train model
        model = self.train_model(X_train, y_train, model_params['model_type'])

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        test_groups = {'income': income_groups_test}
        metrics = self.calculate_fairness_metrics(y_test, y_pred, test_groups)

        # Store results
        self.results = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'X_test': X_test,
            'y_test': y_test,
            'groups_test': test_groups,
            'raw_data': raw_df
        }

        return self.results

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            return None

        metrics = self.results['metrics']

        # 1. Performance gauges
        fig_gauges = go.Figure()

        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['accuracy'] * 100,
            title={'text': "Accuracy", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}},
            domain={'row': 0, 'column': 0}
        ))

        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['fairness_score'] * 100,
            title={'text': "Fairness Score", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}},
            domain={'row': 0, 'column': 1}
        ))

        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['recall'] * 100,
            title={'text': "Sensitivity", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}},
            domain={'row': 0, 'column': 2}
        ))

        fig_gauges.update_layout(
            grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
            height=250
        )

        # 2. Group comparison bar chart
        if 'group_metrics' in metrics and 'income' in metrics['group_metrics']:
            group_data = []
            for group_id, group_metric in metrics['group_metrics']['income'].items():
                group_data.append({
                    'Group': 'Low Income' if group_id == 0 else 'Higher Income',
                    'Accuracy': group_metric['accuracy'],
                    'Sensitivity': group_metric['recall'],
                    'Precision': group_metric['precision'],
                    'Detection Rate': group_metric['detection_rate']
                })

            df_group = pd.DataFrame(group_data)

            fig_groups = px.bar(
                df_group,
                x='Group',
                y=['Accuracy', 'Sensitivity', 'Precision'],
                barmode='group',
                title='Performance by Income Group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            fig_groups = None

        # 3. Fairness disparities radar chart
        if 'fairness_disparities' in metrics:
            fairness_df = pd.DataFrame({
                'Metric': list(metrics['fairness_disparities'].keys()),
                'Disparity': list(metrics['fairness_disparities'].values())
            })

            fig_fairness = px.bar(
                fairness_df,
                x='Metric',
                y='Disparity',
                title='Fairness Disparities',
                color='Disparity',
                color_continuous_scale='RdYlGn_r'
            )
        else:
            fig_fairness = None

        return {
            'gauges': fig_gauges,
            'group_comparison': fig_groups,
            'fairness_disparities': fig_fairness
        }

    def generate_recommendations(self):
        """Generate actionable recommendations"""
        if not self.results or 'metrics' not in self.results:
            return []

        metrics = self.results['metrics']
        recommendations = []

        # Check accuracy
        if metrics['accuracy'] < 0.7:
            recommendations.append({
                'priority': 'High',
                'category': 'Performance',
                'recommendation': 'Improve model performance through feature engineering or hyperparameter tuning',
                'action': 'Review feature importance and consider additional clinical features'
            })

        # Check fairness
        if 'fairness_score' in metrics and metrics['fairness_score'] < 0.7:
            recommendations.append({
                'priority': 'High',
                'category': 'Fairness',
                'recommendation': 'Address algorithmic bias affecting underserved groups',
                'action': 'Implement fairness-aware algorithms or post-processing corrections'
            })

        # Check group disparities
        if 'fairness_disparities' in metrics:
            for metric_name, disparity in metrics['fairness_disparities'].items():
                if disparity > 0.2:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Equity',
                        'recommendation': f'Reduce {metric_name.replace("_", " ").title()}',
                        'action': f'Consider group-specific thresholds or reweighting strategies'
                    })

        # Always include general recommendations
        recommendations.extend([
            {
                'priority': 'Low',
                'category': 'Monitoring',
                'recommendation': 'Establish continuous monitoring of performance across demographic groups',
                'action': 'Set up automated fairness audits and reporting'
            },
            {
                'priority': 'Medium',
                'category': 'Transparency',
                'recommendation': 'Improve model interpretability for clinical stakeholders',
                'action': 'Implement SHAP or LIME explanations for key predictions'
            }
        ])

        return recommendations