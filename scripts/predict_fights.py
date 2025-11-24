"""
Advanced fight prediction system combining Elo ratings with statistical features.
Uses multiple models and ensemble methods for improved accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.insert(0, 'scripts')
from full_genetic_with_k_denom_mov import run_basic_elo, latest_ratings_from_trained_df, find_fighter_match, normalize_name
from elo_utils import add_bout_counts


def extract_features(df, elo_df=None, is_test_data=False):
    """
    Extract features for prediction from the fight data.
    Combines Elo ratings with statistical features.
    
    Args:
        df: DataFrame with fight data
        elo_df: DataFrame with Elo ratings (for training data)
        is_test_data: If True, df has 'fighter'/'opp_fighter' columns instead of 'FIGHTER'/'opp_FIGHTER'
    """
    features = []
    
    # Build name matching set if test data
    all_training_fighters = None
    if is_test_data and elo_df is not None:
        all_training_fighters = set()
        for _, r in elo_df.iterrows():
            all_training_fighters.add(r["FIGHTER"])
            all_training_fighters.add(r["opp_FIGHTER"])
    
    for _, row in df.iterrows():
        feat = {}
        
        # Handle column name differences between train and test
        if is_test_data:
            f1_name = row.get('fighter', '')
            f2_name = row.get('opp_fighter', '')
            date_col = 'date'
        else:
            f1_name = row.get('FIGHTER', '')
            f2_name = row.get('opp_FIGHTER', '')
            date_col = 'DATE'
        
        # Elo features
        if elo_df is not None:
            # Get Elo ratings for this fight
            fight_date = pd.to_datetime(row[date_col])
            rating_lookup = latest_ratings_from_trained_df(elo_df, as_of_date=fight_date)
            
            # Match fighter names if test data
            if is_test_data and all_training_fighters:
                f1_matched = find_fighter_match(f1_name, all_training_fighters) or f1_name
                f2_matched = find_fighter_match(f2_name, all_training_fighters) or f2_name
            else:
                f1_matched = f1_name
                f2_matched = f2_name
            
            f1_elo = rating_lookup.get(f1_matched, 1500)
            f2_elo = rating_lookup.get(f2_matched, 1500)
            
            feat['elo_diff'] = f1_elo - f2_elo
            feat['elo_avg'] = (f1_elo + f2_elo) / 2
            feat['elo_ratio'] = f1_elo / f2_elo if f2_elo > 0 else 1.0
        else:
            feat['elo_diff'] = 0
            feat['elo_avg'] = 1500
            feat['elo_ratio'] = 1.0
        
        # Helper to safely get numeric value
        def safe_get(key, default=0):
            val = row.get(key, default)
            try:
                return float(val) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default
        
        # Physical attributes (test data doesn't have these, so use 0)
        if not is_test_data:
            feat['height_diff'] = safe_get('HEIGHT') - safe_get('opp_HEIGHT')
            feat['reach_diff'] = safe_get('REACH') - safe_get('opp_REACH')
            feat['age_diff'] = safe_get('age') - safe_get('opp_age')
            height = safe_get('HEIGHT', 1)
            feat['reach_advantage'] = feat['reach_diff'] / height if height > 0 else 0
        else:
            # Test data doesn't have physical attributes
            feat['height_diff'] = 0
            feat['reach_diff'] = 0
            feat['age_diff'] = 0
            feat['reach_advantage'] = 0
        
        # Experience features (test data doesn't have these either)
        if not is_test_data:
            feat['boutcount_diff'] = safe_get('precomp_boutcount') - safe_get('opp_precomp_boutcount')
            feat['win_rate_diff'] = safe_get('precomp_winavg') - safe_get('opp_precomp_winavg')
            
            # Striking features (precomp = before this fight)
            feat['sigstr_pm_diff'] = safe_get('precomp_sigstr_pm') - safe_get('opp_precomp_sigstr_pm')
            feat['sigstr_acc_diff'] = safe_get('precomp_sigstr_perc') - safe_get('opp_precomp_sigstr_perc')
            feat['str_def_diff'] = safe_get('precomp_strdef') - safe_get('opp_precomp_strdef')
            
            # Grappling features
            feat['td_avg_diff'] = safe_get('precomp_tdavg') - safe_get('opp_precomp_tdavg')
            feat['td_acc_diff'] = safe_get('precomp_tdacc_perc') - safe_get('opp_precomp_tdacc_perc')
            feat['td_def_diff'] = safe_get('precomp_tddef') - safe_get('opp_precomp_tddef')
            
            # Finish rate (KO/Sub ability)
            feat['finish_rate_diff'] = safe_get('precomp_finish_rate') - safe_get('opp_precomp_finish_rate')
            
            # Recent form (last 3 fights)
            feat['recent_win_rate_diff'] = safe_get('precomp_winavg3') - safe_get('opp_precomp_winavg3')
            feat['recent_sigstr_pm_diff'] = safe_get('precomp_sigstr_pm3') - safe_get('opp_precomp_sigstr_pm3')
            
            # Control time
            feat['ctrl_per_min_diff'] = safe_get('precomp_ctrl_per_min') - safe_get('opp_precomp_ctrl_per_min')
        else:
            # Test data - set to 0 (we'll rely on Elo mainly)
            feat['boutcount_diff'] = 0
            feat['win_rate_diff'] = 0
            feat['sigstr_pm_diff'] = 0
            feat['sigstr_acc_diff'] = 0
            feat['str_def_diff'] = 0
            feat['td_avg_diff'] = 0
            feat['td_acc_diff'] = 0
            feat['td_def_diff'] = 0
            feat['finish_rate_diff'] = 0
            feat['recent_win_rate_diff'] = 0
            feat['recent_sigstr_pm_diff'] = 0
            feat['ctrl_per_min_diff'] = 0
        
        features.append(feat)
    
    return pd.DataFrame(features)


def prepare_training_data(df, test_start_date, best_params=None):
    """
    Prepare training data with Elo ratings and features.
    """
    # Filter to training data only
    df_train = df[df['DATE'] < test_start_date].copy()
    
    # Calculate Elo ratings
    if best_params:
        mov_params = {
            "w_ko": best_params.get("w_ko", 1.4),
            "w_sub": best_params.get("w_sub", 1.3),
            "w_udec": best_params.get("w_udec", 1.0),
            "w_sdec": best_params.get("w_sdec", 0.7),
            "w_mdec": best_params.get("w_mdec", 0.9),
        }
        k = best_params.get("k", 32)
    else:
        mov_params = None
        k = 32
    
    df_train_elo = run_basic_elo(df_train, k=k, mov_params=mov_params)
    
    # Extract features
    features_df = extract_features(df_train, elo_df=df_train_elo)
    
    # Get labels
    labels = df_train['result'].values
    
    # Filter out invalid results
    valid_mask = (labels == 0) | (labels == 1)
    features_df = features_df[valid_mask]
    labels = labels[valid_mask]
    
    return features_df, labels, df_train_elo


def prepare_test_data(test_df, df_train_elo, best_params=None):
    """
    Prepare test data with Elo ratings and features.
    """
    test_df_copy = test_df.copy()
    
    # Extract features using training Elo (mark as test data for name matching)
    features_df = extract_features(test_df_copy, elo_df=df_train_elo, is_test_data=True)
    
    # Get labels
    labels = pd.to_numeric(test_df_copy['result'], errors='coerce').values
    
    # Filter out invalid results
    valid_mask = (labels == 0) | (labels == 1)
    features_df = features_df[valid_mask]
    labels = labels[valid_mask]
    
    return features_df, labels


def train_ensemble_models(X_train, y_train):
    """
    Train multiple models and return them.
    """
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models
    trained_models = {}
    trained_models['logistic'] = (models['logistic'].fit(X_train_scaled, y_train), scaler)
    trained_models['random_forest'] = (models['random_forest'].fit(X_train, y_train), None)
    trained_models['gradient_boosting'] = (models['gradient_boosting'].fit(X_train, y_train), None)
    
    return trained_models


def predict_ensemble(trained_models, X_test):
    """
    Make predictions using ensemble of models (voting).
    """
    predictions = []
    
    for model_name, (model, scaler) in trained_models.items():
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict_proba(X_test_scaled)[:, 1]
        else:
            pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
    
    # Average probabilities
    avg_pred = np.mean(predictions, axis=0)
    final_pred = (avg_pred > 0.5).astype(int)
    
    return final_pred, avg_pred


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """
    Evaluate prediction accuracy and return detailed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n=== Prediction Accuracy ===")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if y_proba is not None:
        # Calculate accuracy by confidence level
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n=== Accuracy by Confidence Level ===")
        for i in range(len(confidence_bins) - 1):
            mask = (y_proba >= confidence_bins[i]) & (y_proba < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_acc = accuracy_score(y_true[mask], y_pred[mask])
                print(f"Confidence [{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}]: "
                      f"{bin_acc:.4f} ({mask.sum()} fights)")
    
    return accuracy


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Add bout counts
    df = add_bout_counts(df)
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    # Load test data
    test_df = pd.read_csv("data/past3_events.csv", low_memory=False)
    test_start_date = pd.to_datetime(test_df['date']).min()
    
    # Use best params from GA (you can update these)
    best_params = {
        'k': 248.02243902953285,
        'w_ko': 1.0455271375631492,
        'w_sub': 1.7827988917883353,
        'w_udec': 1.0390668072379363,
        'w_sdec': 0.7344131878792415,
        'w_mdec': 0.968346164521973
    }
    
    print(f"Test start date: {test_start_date}")
    print("\nPreparing training data...")
    X_train, y_train, df_train_elo = prepare_training_data(df, test_start_date, best_params)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {list(X_train.columns)}")
    
    print("\nTraining models...")
    models = train_ensemble_models(X_train, y_train)
    
    print("\nPreparing test data...")
    X_test, y_test = prepare_test_data(test_df, df_train_elo, best_params)
    
    print(f"Test samples: {len(X_test)}")
    
    print("\nMaking predictions...")
    y_pred, y_proba = predict_ensemble(models, X_test)
    
    print("\nEvaluating predictions...")
    accuracy = evaluate_predictions(y_test, y_pred, y_proba)
    
    print(f"\n=== Comparison ===")
    print(f"Baseline (random): 50.00%")
    print(f"Elo-only baseline: ~51.5%")
    print(f"Ensemble model: {accuracy*100:.2f}%")
    print(f"Improvement: {accuracy*100 - 51.5:.2f} percentage points")

