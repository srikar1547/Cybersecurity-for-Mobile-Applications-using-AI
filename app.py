from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

class CybersecurityModel:
    def __init__(self):
        self.pca = None
        self.model = None
        self.encoders = {}

    def preprocess_data(self, dataset_path):
        # Load dataset
        data = pd.read_csv(dataset_path)

        # Encode categorical variables and save encoders
        categorical_columns = [
            "Permission_Camera", "Permission_Location", "Permission_Contacts",
            "Permission_Microphone", "Permission_Storage", "Suspicious_Patterns",
            "Risk_Factor", "Category", "Update_Frequency", "Source_Type"
        ]
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
            self.encoders[col] = encoder  # Save the encoder for later use

        # Separate features and target
        X = data.drop(["App_ID", "Is_Malicious"], axis=1)
        y = data["Is_Malicious"]

        return X, y

    def train_model(self, dataset_path):
        # Preprocess the data
        X, y = self.preprocess_data(dataset_path)

        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=5)
        X_reduced = self.pca.fit_transform(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Save the trained model, PCA, and encoders for persistence
        with open("pca.pkl", "wb") as f:
            pickle.dump(self.pca, f)
        with open("random_forest.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open("encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)

        # Evaluate the model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return accuracy

    def predict(self, input_data):
        # Load the trained model, PCA, and encoders if not already loaded
        if self.pca is None or self.model is None or not self.encoders:
            with open("pca.pkl", "rb") as f:
                self.pca = pickle.load(f)
            with open("random_forest.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("encoders.pkl", "rb") as f:
                self.encoders = pickle.load(f)

        # Encode categorical input data
        for col, encoder in self.encoders.items():
            if col in input_data:
                try:
                    input_data[col] = encoder.transform([input_data[col]])[0]
                except ValueError:
                    raise ValueError(f"Unseen label '{input_data[col]}' in column '{col}'. Please provide a valid label.")

        # Transform the input data using PCA
        input_df = pd.DataFrame([input_data])
        input_reduced = self.pca.transform(input_df)

        # Predict using the trained model
        prediction = self.model.predict(input_reduced)
        confidence = self.model.predict_proba(input_reduced).max()

        # Generate dynamic analysis and summary
        prediction_label = "Malicious" if prediction[0] == 1 else "Safe"
        analysis = []

        if prediction_label == "Safe":
            analysis.append("✅ The app has low risks based on its behavior and permissions.")
            analysis.append("✅ Network and API usage are within normal ranges.")
            analysis.append("✅ Battery and data usage appear safe.")
        elif prediction_label == "Malicious":
            analysis.append("❌ The app requests excessive permissions.")
            analysis.append("❌ High network activity indicates potential malicious activity.")
            analysis.append("❌ Abnormal battery and data usage linked to suspicious behavior.")

        app_summary = (
            f"Based on the analysis, the app is categorized as '{prediction_label}' "
            f"with a confidence level of {confidence * 100:.2f}%. "
            "It is recommended to review the app's permissions and usage patterns carefully."
        )

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "analysis": analysis,
            "summary": app_summary
        }

# Initialize the CybersecurityModel instance
cybersecurity_model = CybersecurityModel()

# Route to serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse user input
        input_data = request.json.get("input_data")
        if not input_data:
            return jsonify({"error": "No input data provided!"}), 400

        # Predict using the trained model
        result = cybersecurity_model.predict(input_data)

        # Return prediction, confidence, analysis, and summary
        return jsonify(result)
    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Train the model before starting the server
    print("Training the model...")
    dataset_path = "C:\\Charan\\Money Projects\\Vivek and Srikar (VTU)\\Cyber Security using AI\\enhanced_mobile_app_data_with_source.csv"

    accuracy = cybersecurity_model.train_model(dataset_path)
    print(f"Model trained successfully with accuracy: {accuracy:.2f}")

    # Start the Flask server
    app.run(debug=True)
