# This file contains the machine learning code to train the model.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import time
import joblib
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

start_time = time.time()
data = pd.read_csv('data/merged_cleaned.csv')

# Filter songs by artist name
artist_keywords = ['bladee', 'ecco', 'lean', 'thaiboy']
data = data[data['artist_names'].str.contains('|'.join(artist_keywords), case=False, na=False)]

# Remove rows without lyrics
data = data[data['lyrics'].notna() & data['lyrics'].str.strip().astype(bool)]

# Get the GPT emotions features
gpt_features = data.filter(regex='^GPT_(?!.*_explanation$)')
features = gpt_features

target = data['track_name'] 

# One-hot encode categorical features, label encode the target track names
features_encoded = pd.get_dummies(features)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target)

# Filter out tracks with less than 5 samples
min_samples_per_class = 5
class_counts = pd.Series(y_encoded).value_counts()
valid_classes = class_counts[class_counts >= min_samples_per_class].index
valid_indices = [i for i, y in enumerate(y_encoded) if y in valid_classes]

features_encoded = features_encoded.iloc[valid_indices]
y_encoded = y_encoded[valid_indices]

# Re-encode the filtered target labels
label_encoder_filtered = LabelEncoder()
y_encoded = label_encoder_filtered.fit_transform(y_encoded)

# Use StratifiedKFold for good test/train representation of songs
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(features_encoded, y_encoded):
    X_train, X_test = features_encoded.iloc[train_index], features_encoded.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    break  # Use the first split

# Test / Train split
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Impute missing values in training data by mean
imputer = SimpleImputer(strategy='mean')  
X_train_imputed = imputer.fit_transform(X_train)

# Apply SMOTE to the imputed training data
print("Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)  
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

# Convert the resampled data back to a DataFrame and reindex to match the original feature set
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)

def collinearity():
    # Calculate collinearity
    correlation_matrix = X_train_resampled.corr()
    threshold = 0.85
    collinear_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                collinear_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
    print("Collinear pairs:")
    for pair in collinear_pairs:
        print(f"{pair[0]} and {pair[1]} with correlation {pair[2]:.2f}")

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.savefig('correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  

# Print dataset data
print("Features of the data:")
print(X_train_resampled.columns.tolist())
print(f"Number of rows in the dataset: {X_train_resampled.shape[0]}")
print(f"Number of columns in the dataset: {data.shape[1]}")

def train_and_save_model(X_train, y_train, X_test, y_test, model_path='xgboost_model.joblib'):
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='auc',
        reg_alpha=0.5,  
        reg_lambda=0.5,
        use_label_encoder=False,
        max_depth=10, 
        n_estimators=20, 
        learning_rate=0.1, 
        min_child_weight=3,  
        n_jobs=-1,  # Use all CPU cores
        tree_method='hist', 
        device='cuda'  # This is for GPU but actually slows it down
    )

    # Train model
    print("Training the model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)], 
        verbose=True  # Print training progress
    )
    print(f"Model trained in {time.time() - start_time:.2f} seconds.")

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Predict and evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Plot feature importance
    importance = model.get_booster().get_score(importance_type='weight')
    print("Feature importance (by weight):")
    for feature, score in importance.items():
        print(f"{feature}: {score}")

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='weight', max_num_features=20)
    plt.title('Top 20 Feature Importances')
    plt.show()
    
    # Plot training and validation loss over iterations
    # Note: You need to capture the evaluation results using the evals_result parameter
    evals_result = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['validation_0']['auc'], label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('AUC')
    plt.title('Model Loss Over Time')
    plt.legend()
    plt.show()

# Function to test the model on a specified time
def test_model_on_specified_time(model_path='xgboost_model.joblib', 
                                 sad=0, happy=0, sadlove=0, happylove=0, hope=0, growth=0, ambition=0,
                                 spirit=0, empower=0, lonely = 0, fear = 0, anger = 0, emotion = 0,
                                 creative = 0, metaphor = 0, material = 0):
    model = joblib.load(model_path)

    # Calculate the average GPT features
    avg_gpt_features = data.filter(regex='^GPT_(?!.*_explanation$)')
    numeric_data = avg_gpt_features.select_dtypes(include=[np.number])
    avg_gpt_features = numeric_data.mean().fillna(0)

    # Adjust features based on input parameters
    avg_gpt_features['GPT_Sad'] = avg_gpt_features['GPT_Sad'] + sad
    avg_gpt_features['GPT_Happy'] = avg_gpt_features['GPT_Happy'] + happy
    avg_gpt_features['GPT_Sad_Love'] = avg_gpt_features['GPT_Sad_Love'] + sadlove
    avg_gpt_features['GPT_Happy_Love'] = avg_gpt_features['GPT_Happy_Love'] + happylove
    avg_gpt_features['GPT_Hope'] = avg_gpt_features['GPT_Hope'] + hope
    avg_gpt_features['GPT_Growth'] = avg_gpt_features['GPT_Growth'] + growth
    avg_gpt_features['GPT_Ambition'] = avg_gpt_features['GPT_Ambition'] + ambition
    avg_gpt_features['GPT_Spirituality'] = avg_gpt_features['GPT_Spirituality'] + spirit
    avg_gpt_features['GPT_Empowerment'] = avg_gpt_features['GPT_Empowerment'] + empower
    avg_gpt_features['GPT_Loneliness'] = avg_gpt_features['GPT_Loneliness'] + lonely
    avg_gpt_features['GPT_Fear'] = avg_gpt_features['GPT_Fear'] + fear
    avg_gpt_features['GPT_Anger'] = avg_gpt_features['GPT_Anger'] + anger
    avg_gpt_features['GPT_Emotion'] = avg_gpt_features['GPT_Emotion'] + emotion
    avg_gpt_features['GPT_Creativity'] = avg_gpt_features['GPT_Creativity'] + creative
    avg_gpt_features['GPT_Metaphorical'] = avg_gpt_features['GPT_Metaphorical'] + metaphor
    avg_gpt_features['GPT_Materialism'] = avg_gpt_features['GPT_Materialism'] + material

    # Create a DataFrame for the specified features
    avg_gpt_features_df = pd.DataFrame([avg_gpt_features])

    # Debug line
    # print(avg_gpt_features) 

    # Reindex to match the training data
    avg_gpt_features_df = avg_gpt_features_df.reindex(columns=features_encoded.columns, fill_value=0)
    prediction = model.predict(avg_gpt_features_df)
    prediction = label_encoder.inverse_transform(prediction)

    # Print prediction
    print(f"Prediction: {prediction[0]}")

# Use this to train the model
train_and_save_model(X_train, y_train, X_test, y_test, model_path='xgboost_model.joblib')

# Example tests:

#print('\nAverage:')
#test_model_on_specified_time()

#print('\nSad:')
#test_model_on_specified_time(sad=3, sadlove=3)

#print('\nSad, hopeless, stagnant:')
#test_model_on_specified_time(sad = 3, happylove = -2, sadlove = 2, lonely = 2, emotion = 2,
#                             hope = -3, growth = -2, spirit = -2, ambition = -2, empower = -3)

#print('\nSpiritual, angry, sad, hopeless:')
#test_model_on_specified_time(sad = 2, fear = 1, anger = 2, hope = -2, spirit = 3, emotion = 1)

#print('\nHappy, hopeful, spiritual, metaphorical:')
#test_model_on_specified_time(happy=4, sad = -1, anger = -1, hope = 4, happylove = 3, spirit = 3, 
#                             empower = 1, creative = 2, lonely = -2, metaphor = 2)

#print('\nHappy, sad, lonely:')
#test_model_on_specified_time(lonely = 4, sad = 3, happy = 2, anger = 2, happylove = 2, sadlove = 4, growth = 3) 
