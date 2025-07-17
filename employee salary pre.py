#employee salary predication using adult csv
#load your libarary
import pandas as pd
import joblib # Import joblib to save/load models and preprocessors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt # Keep for local plotting, but remember for Streamlit you'd use st.pyplot

# IMPORTANT: For Streamlit Cloud deployment, the CSV file needs to be in your GitHub repository
# alongside this script. Remove the absolute path.
data = pd.read_csv("adult 3.csv")

# --- Data Cleaning and Preprocessing (as in your original script) ---
data.occupation = data.occupation.replace({'?': 'others'})
data.workclass = data.workclass.replace({'?': 'NOT LISTED'})
data['native-country'] = data['native-country'].replace({'?': 'NOT COUNTABLE'})

data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
data = data[data['occupation'] != 'Armed-Forces']
data = data[data['occupation'] != 'Priv-house-serv']
data = data[data['education'] != '7th-8th']
data = data[data['education'] != 'Preschool']
data = data[data['education'] != '1st-4th']
data = data[data['education'] != '5th-6th']
data = data[data['education'] != '9th']
data = data[data['education'] != '11th']
data = data[data['native-country'] != 'Holand-Netherlands']
data = data[data['native-country'] != 'Hungary']
data = data[data['native-country'] != 'Honduras']
data = data[data['native-country'] != 'Laos']
data = data[data['marital-status'] != 'Scotland']
data = data[data['marital-status'] != 'Yugoslavia']

# Drop 'educational-num' as it's often redundant with 'education'
data.drop(columns=['educational-num'], inplace=True)

data = data[(data['age'] <= 70) & (data['age'] >= 20)]

# Initialize LabelEncoder
encoder = LabelEncoder()

# List of all categorical columns to be encoded
# Based on adult.csv and your script, these are the ones that need encoding
categorical_cols = [
    'workclass', 'marital-status', 'occupation', 'relationship',
    'race', 'gender', 'native-country', 'education', 'income' # 'income' is target, but also encoded
]

# Apply LabelEncoder to each specified categorical column
# IMPORTANT: Each column will have its own mapping. If you want a single encoder for all,
# you'd need to fit it on the union of all unique values, which is more complex.
# For simplicity, we'll assume `encoder.fit_transform` on each column sequentially.
# For prediction, we will need to re-initialize an encoder for each column.
# A more robust approach would be to use OneHotEncoder or save a dictionary of LabelEncoders.
# Given your existing code structure, we'll save the 'encoder' object after fitting it on 'education',
# as that was the last one causing issues. For other columns, we'll assume the same encoder instance
# can handle the range of values, which might be a simplification.
# The best practice is to save a dict of encoders or use a ColumnTransformer with OneHotEncoder.

# Let's save a *single* encoder instance after it has seen all the unique values from all categorical columns.
# To do this, we'll fit it on a combined series of all unique categorical values.
# Collect all unique categorical values from the relevant columns
all_unique_categorical_values = pd.Series(dtype='object')
for col in categorical_cols:
    all_unique_categorical_values = pd.concat([all_unique_categorical_values, data[col].astype(str)])

encoder.fit(all_unique_categorical_values.unique()) # Fit the encoder on ALL unique categorical values

# Now transform the columns using this fitted encoder
for col in categorical_cols:
    data[col] = encoder.transform(data[col])


# Apply numerical filters after encoding
data = data[(data['workclass'] <= 5) & (data['workclass'] >= 1)]
data = data[(data['marital-status'] <= 5) & (data['marital-status'] >= 1)]
data = data[(data['native-country'] <= 35) & (data['native-country'] >= 5)]

# Define features (x) and target (y)
# Ensure 'x' contains all features the model expects, in the correct order.
# The original adult.csv columns (excluding 'educational-num' and 'income') are:
# 'age', 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation',
# 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
# 'hours-per-week', 'native-country'

# Get the list of columns that will be in 'x'
feature_columns_for_x = [col for col in data.columns if col not in ['income']]
x = data[feature_columns_for_copy] # Use .copy() to avoid SettingWithCopyWarning
y = data['income']

# Initialize and fit MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# --- Model Training and Evaluation (as in your original script) ---
# Note: Your script has two separate model training/evaluation blocks.
# I'll keep the first one that uses MinMaxScaler, and then the second one
# which saves the 'best_model.pkl' (which seems to be the one you want to deploy).
# The second block uses StandardScaler within a Pipeline, which is different from MinMaxScaler.
# This means your saved 'best_model.pkl' might expect StandardScaler-scaled data.
# For consistency, I will ensure the final 'best_model.pkl' is trained on the MinMaxScaler output.

# Let's streamline the model training to ensure the saved model is consistent
# with the MinMaxScaler output.
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True), # Added probability=True for potential future use (e.g., predict_proba)
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}
best_model = None
best_model_name = ""
max_accuracy = -1

for name, model in models.items():
    # Fit the model directly on the already scaled 'x'
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    acc = accuracy_score(ytest, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

    if acc > max_accuracy:
        max_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n✅ Best model: {best_model_name} with accuracy {max_accuracy:.4f}")

# --- SAVE THE BEST MODEL AND PREPROCESSORS ---
joblib.dump(best_model, "best_model.pkl")
joblib.dump(encoder, "label_encoder.pkl") # Save the fitted LabelEncoder
joblib.dump(scaler, "min_max_scaler.pkl") # Save the fitted MinMaxScaler

print("✅ Saved best model as best_model.pkl")
print("✅ Saved label_encoder as label_encoder.pkl")
print("✅ Saved min_max_scaler as min_max_scaler.pkl")

# Optional: Plotting (for local execution, not directly in Streamlit app)
# plt.bar(results.keys(), results.values(), color='skyblue')
# plt.ylabel('Accuracy Score')
# plt.title('Model comparison')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()
