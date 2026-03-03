import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# --- CONFIGURATION ---
input_file = "data/enriched_project_dataset.csv"

def train_and_evaluate():
    print("Loading Enriched Dataset...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Ensure the data pipeline has been run.")
        return

    # --- 1. ACADEMIC ALIGNMENT: FIXING THE TARGET VARIABLE ---
    # As noted in the ISA-II report, we have a ~30.2:1 class imbalance.
    # Since historical data is mostly "Resolved", we define "Stalled" as the top 3.2% 
    # of tickets that took an agonizingly long time to resolve.
    delay_threshold = df['days_to_resolve'].quantile(0.968) 
    df['is_stalled'] = np.where(df['days_to_resolve'] > delay_threshold, 1, 0)

    # --- 2. SELECTING THE EXACT 7 FEATURES FROM ISA-II REPORT ---
    feature_columns = [
        'behavior_score',           # VADER compound score
        'subject_length',           # Length of email subject
        'priority_numeric',         # Blocker=5 to Trivial=1
        'sentiment_variance',       # Std dev of scores (from test.txt)
        'email_volume_per_ticket',  # Count of emails (from test.txt)
        'sentiment_trend'           # Rolling average (from test.txt)
    ]
    target_column = 'is_stalled'

    # --- 3. DATA CLEANING FOR ML ---
    # Random Forest cannot handle NaNs. 
    # If a ticket only has 1 email, its 'sentiment_variance' is NaN. We fill these with 0.
    df['sentiment_variance'] = df['sentiment_variance'].fillna(0)
    df['sentiment_trend'] = df['sentiment_trend'].fillna(df['behavior_score'])
    
    # Drop any remaining rows missing core features
    df_clean = df.dropna(subset=feature_columns + [target_column])

    X = df_clean[feature_columns]
    y = df_clean[target_column]

    print(f"\nDataset shape for modeling: {X.shape}")
    print(f"Class distribution (0=Resolved, 1=Stalled):\n{y.value_counts()}")

    # --- 4. TRAIN/TEST SPLIT ---
    print("\nSplitting data (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )

    # --- 5. MODEL INITIALIZATION ---
    print("Initializing Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1 
    )

# --- 6. TRAINING & EVALUATION ---
    print("Training model...")
    rf_model.fit(X_train, y_train)

    print("Evaluating model with Recall-Optimized Threshold (0.35)...\n")
    # Get raw probabilities
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    # Apply custom threshold
    y_pred = (y_prob >= 0.35).astype(int)

    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['Resolved (0)', 'Stalled (1)']))

    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Resolved correctly identified): {cm[0][0]}")
    print(f"False Positives (Resolved flagged as Stalled) : {cm[0][1]}")
    print(f"False Negatives (Stalled missed by model)     : {cm[1][0]}")
    print(f"True Positives  (Stalled correctly identified): {cm[1][1]}\n")

    print("=== FEATURE IMPORTANCE ===")
    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    for index, row in feat_imp.iterrows():
        print(f"{row['Feature']:<25}: {row['Importance']:.4f}")

# --- SAVE CONFUSION MATRIX VISUAL ---
    print("Saving Confusion Matrix plot to visuals/8_confusion_matrix.png...")
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Resolved (0)', 'Stalled (1)'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Random Forest Confusion Matrix (Threshold = 0.35)')
    plt.grid(False)
    plt.savefig('visuals/8_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    train_and_evaluate()