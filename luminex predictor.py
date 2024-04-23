import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        self.dropout1 = nn.Dropout(0)
        self.fc2 = nn.Linear(50, 20)
        self.dropout2 = nn.Dropout(0)
        self.fc3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Load the model
model_path = 'C:/Users/tlee/OneDrive - Claim Return, LLC/Desktop/ML/trained model.pth'
model = Net(num_features=2565)  # Adjust based on the actual number of features
model.load_state_dict(torch.load(model_path))
model.eval()

# Load encoders
mlb_cpt_path = 'C:/Users/tlee/OneDrive - Claim Return, LLC/Desktop/ML/mlb_cpt.joblib'
mlb_rev_path = 'C:/Users/tlee/OneDrive - Claim Return, LLC/Desktop/ML/mlb_rev.joblib'
mlb_cpt = joblib.load(mlb_cpt_path)
mlb_rev = joblib.load(mlb_rev_path)

# Load unseen data
data_path = 'C:/Users/tlee/OneDrive - Claim Return, LLC/Desktop/ML/Modeling/luminex unseen.csv'
unseen_data = pd.read_csv(data_path)
unseen_data['CPTCodeList'] = unseen_data['CPTCodes'].apply(lambda x: x.split(", ") if pd.notna(x) else [])
unseen_data['RevCodeList'] = unseen_data['RevenueCodes'].apply(lambda x: [str(int(x))] if pd.notna(x) else [])
unseen_data['PatientGender'] = unseen_data['PatientGender'].str.lower().str.strip()

# One-hot encode features
one_hot_cpt_codes = mlb_cpt.transform(unseen_data['CPTCodeList'])
one_hot_rev_codes = mlb_rev.transform(unseen_data['RevCodeList'])
gender_dummies = pd.get_dummies(unseen_data['PatientGender'], prefix='Gender', dummy_na=True)
gender_dummies = gender_dummies.astype(int)

# Combine features into one DataFrame
features_cpt_df = pd.DataFrame(one_hot_cpt_codes, columns=mlb_cpt.classes_)
features_rev_df = pd.DataFrame(one_hot_rev_codes, columns=mlb_rev.classes_)
features_gender_df = gender_dummies
features_full_df = pd.concat([features_cpt_df, features_rev_df, features_gender_df], axis=1)

# Convert DataFrames to tensors
features_cpt = torch.tensor(features_cpt_df.values, dtype=torch.float32)
features_rev = torch.tensor(features_rev_df.values, dtype=torch.float32)
features_gender = torch.tensor(features_gender_df.values, dtype=torch.float32)
features_full = torch.cat([features_cpt, features_rev, features_gender], dim=1)

# Predict using the model
with torch.no_grad():
    predictions = model(features_full) > 0.5

# Convert predictions to numpy array and add to DataFrame
predictions_numpy = predictions.numpy().astype(int)
unseen_data['Prediction'] = predictions_numpy

# Optionally save the updated DataFrame to CSV
unseen_data.to_csv('C:/Users/tlee/OneDrive - Claim Return, LLC/Desktop/ML/Modeling/Unseen_with_Predictions.csv', index=False)

# Output predictions
print("Predictions added to DataFrame and saved to CSV.")
