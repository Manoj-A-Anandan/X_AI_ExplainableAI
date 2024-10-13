import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import shap
import numpy as np

class TitanicModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        self.explainer_lime = None
        self.explainer_shap = None
        self.feature_names = None

    def train_model(self, data_path):
        # Load and preprocess the data
        data = pd.read_csv(data_path)
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
        data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
        data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

        X = data.drop(['Survived', 'PassengerId', 'Name'], axis=1)
        y = data['Survived']

        self.feature_names = X.columns.tolist()  # Store feature names for LIME and SHAP

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Initialize LIME and SHAP explainers
        self.explainer_lime = LimeTabularExplainer(X_train.values, feature_names=self.feature_names,
                                                   class_names=['Not Survived', 'Survived'],
                                                   mode='classification')
        self.explainer_shap = shap.Explainer(self.model, X_train)

    def explain_instance(self, instance):
        # Ensure instance is a 2D array
        if isinstance(instance, list):
            instance = np.array(instance)

        # Check the shape of the instance and reshape if necessary
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)  # Reshape to 2D if it's a single instance

        # Create a DataFrame from the instance
        instance_df = pd.DataFrame(instance, columns=self.feature_names)

        # LIME explanation
        lime_explanation = self.explainer_lime.explain_instance(instance_df.values[0], self.model.predict_proba)

        # SHAP explanation
        shap_values = self.explainer_shap(instance_df)

        # Convert explanations to be returned in JSON/text format
        lime_explanation_list = lime_explanation.as_list()
        shap_values_serialized = shap_values.values.tolist()

        return {
            'lime_explanation': lime_explanation_list,
            'shap_values': shap_values_serialized,
            'expected_value': shap_values.base_values.tolist()
        }
