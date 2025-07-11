import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

class CybersecurityModel:
    def __init__(self):
        self.model = None
        self.encoders = {}  # Dictionary to hold all LabelEncoders
        self.columns_to_encode = [
            "Permission_Camera", "Permission_Location", "Permission_Contacts",
            "Permission_Microphone", "Permission_Storage", "Suspicious_Patterns",
            "Risk_Factor", "Category", "Update_Frequency", "Source_Type"
        ]

    def train_model(self, dataset_path):
        # Load dataset
        data = pd.read_csv(dataset_path)

        # Encode categorical columns
        for col in self.columns_to_encode:
            if col in data.columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))
                self.encoders[col] = encoder

        # Define features and target
        X = data.drop("Target", axis=1)
        y = data["Target"]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

        # Evaluate model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained successfully! Accuracy: {accuracy}")
        return accuracy

    def predict(self, input_data):
        # Ensure model is trained
        if not self.model:
            raise Exception("Model not trained yet!")

        # Encode input data
        for col, encoder in self.encoders.items():
            if col in input_data:
                try:
                    # Transform using encoder (handles known categories)
                    input_data[col] = encoder.transform([input_data[col]])[0]
                except ValueError:
                    # Handle unseen category by assigning a default value
                    print(f"Warning: Unseen value '{input_data[col]}' for column '{col}', assigning default.")
                    input_data[col] = encoder.transform([encoder.classes_[0]])[0]

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = self.model.predict(input_df)[0]
        confidence = max(self.model.predict_proba(input_df)[0])

        # Generate explanation
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "analysis": "This is a dummy analysis.",
            "summary": "Based on the input, the app's cybersecurity risk is determined."
        }

        return explanation


# Initialize the cybersecurity model
cybersecurity_model = CybersecurityModel()
