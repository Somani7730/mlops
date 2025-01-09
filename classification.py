import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train a RandomForest model to get feature importances
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for better understanding
features_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': feature_importances
})

# Sort by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Select top N features (e.g., top 10 most important features)
top_n = 10  # You can adjust this as needed
top_features = features_df.head(top_n)['Feature'].values

# Print selected features
print(f"Top {top_n} features: {top_features}")

# Use SelectFromModel to select the most important features
sfm = SelectFromModel(model, threshold='mean', max_features=top_n)
X_selected = sfm.transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train a new model using the selected features
model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train, y_train)

# Save the trained model
with open('breast_cancer_model_selected_features.pkl', 'wb') as f:
    pickle.dump(model_selected, f)

print("Model trained with selected features and saved as 'breast_cancer_model_selected_features.pkl'")
