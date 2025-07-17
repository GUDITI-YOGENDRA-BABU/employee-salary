# employee salary prediction using adult csv
# load your library
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib # For saving the model

# --- Data Loading and Initial Cleaning ---
data = pd.read_csv(r"C:\Users\gudit\Downloads\RECOVER\adult 3.csv")

print("Original occupation value counts:")
print(data.occupation.value_counts())

# Handle missing values
data.occupation = data.occupation.replace({'?': 'others'})
data.workclass = data.workclass.replace({'?': 'NOT LISTED'})
data['native-country'] = data['native-country'].replace({'?': 'NOT COUNTABLE'})

# Filter out specific categories
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
data = data[data['occupation'] != 'Armed-Forces']
data = data[data['occupation'] != 'Priv-house-serv']
data = data[data['education'] != '7th-8th']
data = data[data['education'] != 'Preschool']
data = data[data['education'] != '1st-4th']
data = data[data['education'] != '5th-6th']
data = data[data['education'] != '9th']
data = data[data['native-country'] != 'Holand-Netherlands']
data = data[data['native-country'] != 'Hungary']
data = data[data['native-country'] != 'Honduras']
data = data[data['native-country'] != 'Laos']
data = data[data['marital-status'] != 'Scotland'] # Note: 'Scotland' and 'Yugoslavia' are countries, not marital statuses. This might be a typo in your original data or a specific cleaning choice.
data = data[data['marital-status'] != 'Yugoslavia'] # Same as above.

print("\nData after initial cleaning and filtering:")
print(data.head())

print("\nOccupation value counts after cleaning:")
print(data.occupation.value_counts())
print("\nWorkclass value counts after cleaning:")
print(data.workclass.value_counts())
print("\nEducation value counts after cleaning:")
print(data.education.value_counts())
print("\nRace value counts after cleaning:")
print(data.race.value_counts())
print("\nGender value counts after cleaning:")
print(data.gender.value_counts())
print("\nMarital-status value counts after cleaning:")
print(data['marital-status'].value_counts())
print("\nNative-country value counts after cleaning:")
print(data['native-country'].value_counts())

# Drop 'educational-num' if it's a duplicate or redundant, otherwise keep it.
# Your previous code had 'education' dropped, but this version has 'educational-num'.
# Assuming 'educational-num' is the one to drop as 'education' is used for filtering.
# If 'educational-num' is important, you might want to keep it and drop 'education' instead.
if 'educational-num' in data.columns:
    data.drop(columns=['educational-num'], inplace=True)
elif 'education-num' in data.columns: # Check for the common column name
    data.drop(columns=['education-num'], inplace=True)
else:
    print("Warning: 'educational-num' or 'education-num' column not found to drop.")

print("\nData after dropping educational-num (if present):")
print(data.head())

# Filter age
data = data[(data['age'] <= 70) & (data['age'] >= 20)]

# --- Feature Engineering and Encoding ---
encoder = LabelEncoder()

# Store original unique values for app.py
# It's crucial these lists reflect the data *after* your filtering but *before* encoding
# and are consistent with what your app.py expects.
workclass_options = data['workclass'].unique().tolist()
marital_status_options = data['marital-status'].unique().tolist()
occupation_options = data['occupation'].unique().tolist()
relationship_options = data['relationship'].unique().tolist()
race_options = data['race'].unique().tolist()
gender_options = data['gender'].unique().tolist()
native_country_options = data['native-country'].unique().tolist()

# Label Encoding
for col in ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country']:
    if col in data.columns: # Ensure column exists before encoding
        data[col] = encoder.fit_transform(data[col])
    else:
        print(f"Warning: Column '{col}' not found for encoding.")

# Filter based on encoded values (re-evaluate if these filters are truly necessary/correct)
# These filters are applied AFTER encoding, which can be tricky as encoded values are arbitrary.
# Consider if these are intended to filter specific *categories* or numerical ranges.
data = data[(data['workclass'] <= 5) & (data['workclass'] >= 1)]
data = data[(data['marital-status'] <= 5) & (data['marital-status'] >= 1)]
data = data[(data['native-country'] <= 35) & (data['native-country'] >= 5)]

# Define features (x) and target (y)
x = data.drop(columns=['income'])
y = data['income']

print("\nTarget variable (y) head:")
print(y.head())

# --- Model Training and Evaluation (Consolidated with Pipeline) ---

# Perform train-test split once for consistency
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=35, stratify=y) # Added stratify for balanced classes

print("\nShape of xtrain:", xtrain.shape)
print("Shape of xtest:", xtest.shape)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000), # Increased max_iter for convergence
    "RandomForest": RandomForestClassifier(random_state=35),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=35), # probability=True needed for predict_proba
    "GradientBoosting": GradientBoostingClassifier(random_state=35)
}

results = {}
best_accuracy = 0
best_model_name = ""
final_pipeline_to_save = None # Initialize to store the best pipeline

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    # Use StandardScaler in the pipeline for consistent scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(xtrain, ytrain)
    y_pred = pipe.predict(xtest)
    acc = accuracy_score(ytest, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(ytest, y_pred))

    # Keep track of the best model's pipeline
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        final_pipeline_to_save = pipe # Store the actual fitted pipeline instance

print(f"\n✅ Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

# --- Saving the Best Model Pipeline ---
if final_pipeline_to_save:
    # Before saving the pipeline, add these checks:
    print(f"\nType of final_pipeline_to_save: {type(final_pipeline_to_save)}")
    print(f"Does final_pipeline_to_save have predict method? {'predict' in dir(final_pipeline_to_save)}")

    # Try a dummy prediction with the fitted pipeline right before saving
    try:
        dummy_prediction = final_pipeline_to_save.predict(xtest[:1])
        print(f"Dummy prediction successful: {dummy_prediction}")
    except Exception as e:
        print(f"Error during dummy prediction before saving: {e}")
        print("This indicates the pipeline was not correctly fitted.")

    joblib.dump(final_pipeline_to_save, 'salary_prediction_model.pkl')
    print(f"\nSaved {best_model_name} pipeline to salary_prediction_model.pkl")
else:
    print("\nError: No best model pipeline found to save.")

# --- Visualization (Optional, for local execution) ---
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

