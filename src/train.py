import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_loader import get_dataloaders, merge_and_create_features
from model import MarchMadnessNN
import config
def train_model():
    print("Preparing data...")
    X_scaled, y_full, feature_names, scaler = get_dataloaders(batch_size=32)
    # Task 1: Synchronize Feature Order
    joblib.dump(feature_names, os.path.join(config.PREPROCESSING_DIR, 'feature_list.pkl'))
    # We need to re-derive seeds to calculate sample weights (upset emphasis)
    # Since X_scaled is already processed, we'll pull seeds from the original matchups if possible
    # or just use the Diff_Barthag/Diff_AdjOE as a proxy for 'Favorite'
    
    matchups, _ = merge_and_create_features()
    # Align matchups with y_full (Standardized 2015-2025)
    matchups = matchups[(matchups['YEAR'] >= 2015) & (matchups['YEAR'] <= 2025)]
    
    sample_weights = []
    for _, row in matchups.iterrows():
        seed_a = float(row.get('SEED_A', 8))
        seed_b = float(row.get('SEED_B', 8))
        winner_a = int(row.get('TeamAwins', 0))
        round_val = int(row.get('ROUND', 0))
        
        # Base weight
        w = 1.0
        # If Upset (Lower rank/Higher number seed wins)
        if (seed_a > seed_b and winner_a == 1) or (seed_b > seed_a and winner_a == 0):
            upset_margin = abs(seed_a - seed_b)
            w += upset_margin / 4.0 # Scale weight by severity of upset
            
            # Step 4: Meta-Learner 'Short' Strategy
            # Targeted weighting for #1-#3 seeds losing in the first weekend (R64/R32)
            is_bust = (seed_a <= 3 and winner_a == 0) or (seed_b <= 3 and winner_a == 1)
            is_first_weekend = (round_val >= 32)
            if is_bust and is_first_weekend:
                w *= 3.0 # Triple weight for "Bust" signatures
                print(f"DEBUG: High-Seed Bust detected ({row['TEAM_A']} vs {row['TEAM_B']}) - Weight boosted.")
            
        sample_weights.append(w)
    
    sample_weights = np.array(sample_weights)
    input_size = len(feature_names)
    
    print(f"\nInitializing Triple-Engine 5-Fold Cross-Validation with input size {input_size}...")
    
    # weights paths
    weights_path_nn = os.path.join(config.WEIGHTS_DIR, 'march_madness_weights.pt')
    weights_path_xgb = os.path.join(config.WEIGHTS_DIR, 'xgb_model.json')
    weights_path_rf = os.path.join(config.WEIGHTS_DIR, 'rf_model.joblib')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_log_losses = []
    global_best_ensemble_loss = float('inf')
    
    # Storage for Meta-Learner Training
    # We'll store (XGB_prob, NN_prob, RF_prob) and the target
    meta_features = []
    meta_targets = []
    meta_weights = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_full)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/5")
        print(f"{'='*60}")
        
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
        w_train_fold = sample_weights[train_idx]
        
        # 1. Train XGBoost (40% Weight)
        print("Training XGBoost with Upset-Weighted Samples...")
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
        xgb_probs = xgb_model.predict_proba(X_val_fold)[:, 1]
        
        # 2. Train Random Forest (25% Weight)
        print("Training Random Forest with Upset-Weighted Samples...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        rf_model.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
        rf_probs = rf_model.predict_proba(X_val_fold)[:, 1]
        
        # 3. Train PyTorch NN (35% Weight)
        print("Training Neural Network...")
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(w_train_fold, dtype=torch.float32)), batch_size=32, shuffle=True)
        
        nn_model = MarchMadnessNN(input_size=input_size)
        criterion = nn.BCEWithLogitsLoss(reduction='none') # None for manual weighting
        optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-3)
        
        nn_best_val_loss = float('inf')
        nn_patience = 15
        nn_no_improve = 0
        
        for epoch in range(150):
            nn_model.train()
            for batch_X, batch_y, batch_w in train_loader:
                optimizer.zero_grad()
                outputs = nn_model(batch_X)
                # Label Smoothing + Manual Weighting
                loss = criterion(outputs, batch_y * 0.9 + 0.05)
                loss = (loss * batch_w.unsqueeze(1)).mean() 
                loss.backward()
                optimizer.step()
            
            # Validation for early stopping
            nn_model.eval()
            with torch.no_grad():
                val_outputs = nn_model(X_val_tensor)
                val_loss = criterion(val_outputs, torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1)).mean().item()
                if val_loss < nn_best_val_loss:
                    nn_best_val_loss = val_loss
                    nn_no_improve = 0
                else:
                    nn_no_improve += 1
            if nn_no_improve >= nn_patience: break
            
        nn_probs = torch.sigmoid(nn_model(X_val_tensor)).detach().numpy().flatten()
        
        # 4. Out-of-Fold Metadata Collection
        # These will be used to train the Meta-Learner
        meta_features_fold = np.column_stack([xgb_probs, nn_probs, rf_probs])
        meta_features.append(meta_features_fold)
        meta_targets.append(y_val_fold)
        
        # Calculate weights for meta-learner (Emphasis on Chaos)
        # We need seeds for the validation indices
        val_matchups = matchups.iloc[val_idx]
        fold_meta_weights = []
        for _, row in val_matchups.iterrows():
            seed_a = float(row.get('SEED_A', 8))
            seed_b = float(row.get('SEED_B', 8))
            # Tournament Chaos: Seed Difference > 4
            if abs(seed_a - seed_b) > 4:
                fold_meta_weights.append(2.5) # Heavy emphasis on learning chaos
            else:
                fold_meta_weights.append(1.0)
        meta_weights.append(fold_meta_weights)

        # 5. Fixed-Weight Eval (Legacy tracking)
        ensemble_probs = (0.40 * xgb_probs) + (0.35 * nn_probs) + (0.25 * rf_probs)
        fold_loss = log_loss(y_val_fold, ensemble_probs)
        fold_log_losses.append(fold_loss)
        
        print(f"  Fold {fold+1} Ensemble Log Loss (Fixed Weights): {fold_loss:.4f}")
        
        # Save base models if this is the globally best ensemble (by fixed weights for consistency)
        if fold_loss < global_best_ensemble_loss:
            global_best_ensemble_loss = fold_loss
            torch.save(nn_model.state_dict(), weights_path_nn)
            xgb_model.save_model(weights_path_xgb)
            joblib.dump(rf_model, weights_path_rf)
            print(f"  *** New Global Best Base Ensemble Loss: {global_best_ensemble_loss:.4f} ***")

    # ---------------------------
    # 5. Train Meta-Learner (Stacking)
    # ---------------------------
    print(f"\n{'='*60}")
    print("TRAINING LOGISTIC REGRESSION META-LEARNER...")
    print(f"{'='*60}")
    
    X_meta = np.vstack(meta_features)
    y_meta = np.concatenate(meta_targets)
    w_meta = np.concatenate(meta_weights)
    
    meta_learner = LogisticRegression()
    meta_learner.fit(X_meta, y_meta, sample_weight=w_meta)
    
    # Eval Meta-Learner
    meta_probs = meta_learner.predict_proba(X_meta)[:, 1]
    meta_loss = log_loss(y_meta, meta_probs)
    
    joblib.dump(meta_learner, os.path.join(config.WEIGHTS_DIR, 'meta_learner.joblib'))
    print(f"Meta-Learner trained on {len(X_meta)} samples.")
    print(f"Meta-Learner Log Loss: {meta_loss:.4f}")
    print(f"Meta-Learner Weights: XGB={meta_learner.coef_[0][0]:.3f}, NN={meta_learner.coef_[0][1]:.3f}, RF={meta_learner.coef_[0][2]:.3f}")
    
    avg_loss = np.mean(fold_log_losses)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS (Target Log Loss < 0.54)")
    print(f"  AVG LOG LOSS: {avg_loss:.4f}")
    if avg_loss < 0.54:
        print(" [SUCCESS] Log Loss threshold met.")
    else:
        print(" [WARNING] Log Loss slightly above threshold.")
    print(f"  Models saved: {weights_path_nn}, {weights_path_xgb}, {weights_path_rf}")

if __name__ == '__main__':
    train_model()
