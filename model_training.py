import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/genloc_isolate.csv")

# Rename columns if necessary
df.columns = ['genloc', 'isolate']  # Assuming 'genloc' corresponds to 'Genloc'

# Preprocess data
le = LabelEncoder()
df['genloc'] = le.fit_transform(df['genloc'])
df['isolate'] = le.fit_transform(df['isolate'])

# Split dataset into features (X) and target (y)
X = df[['genloc']]  # Use 'Genloc' as the feature
y = df['isolate']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)


# Save the trained model for future use
import joblib
joblib.dump(model, 'uti_isolate_prediction_model.pkl')
