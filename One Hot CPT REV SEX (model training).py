import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load and Prepare the Data
# Load the dataset
df = pd.read_csv(r'C:\Users\Luis Ysturiz\Downloads\luminx_new_slim.csv')  # Update with your actual file path

# Exclude the 'CaseID' column if it's not needed for modeling
df.drop('CaseID', axis=1, inplace=True)

# Standardize the 'PatientGender' column to prevent multiple columns for the same gender due to case differences
gender_mapping = {
    'male': 'm', 'Male': 'm', 'M': 'm',
    'female': 'f', 'Female': 'f', 'F': 'f'
}
df['PatientGender'] = df['PatientGender'].map(gender_mapping)

# Step 2: Create One-Hot Encoded Features
# Splitting the CPTCodes and RevenueCodes columns into lists of codes
df['CPTCodeList'] = df['CPTCodes'].dropna().apply(lambda x: x.split(", "))
df['RevCodeList'] = df['RevenueCodes'].dropna().apply(lambda x: x.split(", "))

# Initialize the MultiLabelBinarizers
mlb_cpt = MultiLabelBinarizer()
mlb_rev = MultiLabelBinarizer()

# One-hot encode the split CPT and Revenue codes
one_hot_cpt_codes = mlb_cpt.fit_transform(df['CPTCodeList'].dropna())
one_hot_rev_codes = mlb_rev.fit_transform(df['RevCodeList'].dropna())

one_hot_cpt_df = pd.DataFrame(one_hot_cpt_codes, columns=mlb_cpt.classes_, dtype=float)
one_hot_rev_df = pd.DataFrame(one_hot_rev_codes, columns=mlb_rev.classes_, dtype=float)

# Aligning the new DataFrames with the original DataFrame
# Adding suffixes to distinguish between columns from CPT and Revenue codes
df = df.join(one_hot_cpt_df, how='left', rsuffix='_CPT').join(one_hot_rev_df, how='left', rsuffix='_REV').fillna(0)

# Dropping original and intermediate columns used for encoding
df.drop(['CPTCodes', 'CPTCodeList', 'RevenueCodes', 'RevCodeList'], axis=1, inplace=True)

# One-hot encode the standardized gender data
gender_dummies = pd.get_dummies(df['PatientGender'], prefix='Gender', dtype=float)
df.drop(['PatientGender'], axis=1, inplace=True)
df = pd.concat([df, gender_dummies], axis=1)

# Save the preprocessed DataFrame to a CSV file (uncomment line below to actually check if one-hot encoding (OHE) worked)
#df.to_csv(r'C:\Users\tlee\OneDrive - Claim Return, LLC\Desktop\ML\Modeling\Preprocessed_Data.csv', index=False)

# Convert DataFrame to tensors, handling potential errors explicitly
try:
    X = torch.tensor(df.drop('ClaimReturned', axis=1).values, dtype=torch.float32)
    y = torch.tensor(df['ClaimReturned'].values, dtype=torch.float32)
except TypeError as e:
    print(e)
    for column in df.columns:
        try:
            torch.tensor(df[column].values, dtype=torch.float32)
        except TypeError as te:
            print(f'Error converting {column}: {te}')

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f"Total number of features: {df.shape[1] - 1}")

# Step 3: Build the PyTorch Model
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        self.dropout1 = nn.Dropout(0)  # Dropout n * 100% of the neurons
        self.fc2 = nn.Linear(50, 20)
        self.dropout2 = nn.Dropout(0)  # Dropout n * 100% of the neurons
        self.fc3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model
model = Net(num_features=X.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Step 4: Train the Model
for epoch in range(1000):
    model.train()  # Ensure the model is in training mode
    outputs_train = model(X_train)
    loss_train = criterion(outputs_train.squeeze(), y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # Validation loss
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs_val = model(X_val)
        loss_val = criterion(outputs_val.squeeze(), y_val)

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Training Loss: {loss_train.item():.4f}, Validation Loss: {loss_val.item():.4f}')

# Step 5: Evaluate the Model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_test = model(X_test) > 0.5
    accuracy = (predicted_test.squeeze().float() == y_test).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}')

# Step 6: Save the trained model + Save encoders and binarizers
joblib.dump(mlb_cpt, r'C:\Users\Luis Ysturiz\Documents\mlb_cpt.joblib')
joblib.dump(mlb_rev, r'C:\Users\Luis Ysturiz\Documents\mlb_rev.joblib')
torch.save(model.state_dict(), r'C:\Users\Luis Ysturiz\Documents\trained model.pth')