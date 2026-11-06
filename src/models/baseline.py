import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def chronological_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the dataframe chronologically into train, validation, and test sets.
    Assumes 'DATE_TIME' column exists.
    """
    # Sort by datetime to ensure chronological order
    df_sorted = df.sort_values('DATE_TIME').reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    return train_df, val_df, test_df

def prepare_data(df, target_col='AC_POWER'):
    """
    Prepares features and target for the baseline XGBoost model.
    """
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'TIME_OF_DAY']
    
    # Drop rows where features or target are missing just in case
    df_clean = df.dropna(subset=features + [target_col])
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    return X, y

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, **kwargs):
    """
    Trains the baseline XGBoost model.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance and returns metrics.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MAE': mae, 'R2': r2}

def predict(model, X):
    """
    Generates predictions.
    """
    return model.predict(X)

if __name__ == "__main__":
    pass
