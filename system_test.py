import requests

# Define the input data (mock/test input for prediction)
input_data = {
    "input_data": {
        "Permission_Camera": "Yes",
        "Permission_Location": "Yes",
        "Permission_Contacts": "No",
        "Permission_Microphone": "Yes",
        "Permission_Storage": "Yes",
        "Suspicious_Patterns": "Low",
        "Risk_Factor": "Moderate",
        "Category": "Social",
        "Update_Frequency": "Frequent",
        "Source_Type": "PlayStore"
    }
}

# Send a POST request to your Flask API endpoint
response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
