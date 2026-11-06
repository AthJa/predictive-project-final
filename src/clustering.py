import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def prepare_daily_profiles(df, value_col='AC_POWER'):
    """
    Reshapes the dataframe to extract daily profiles for each inverter.
    Expects a preprocessed dataframe with DATE_TIME, INVERTER_ID, AC_POWER.
    Returns a 3D numpy array of shape (n_samples, time_steps, 1) expected by tslearn.
    """
    if 'DATE' not in df.columns:
        df['DATE'] = df['DATE_TIME'].dt.date
        
    # Pivot to get INVERTER_ID and DATE as index, and TIME_OF_DAY as columns
    daily_profiles = df.pivot_table(index=['INVERTER_ID', 'DATE'], 
                                    columns='TIME_OF_DAY', 
                                    values=value_col).fillna(0)
    
    indices = daily_profiles.index
    
    # Reshape to (n_samples, time_steps, 1)
    X = daily_profiles.values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, indices, daily_profiles

def cluster_inverters_dtw(X, n_clusters=3, random_state=42):
    """
    Performs TimeSeries K-Means clustering with DTW metric.
    """
    # Scale the time series (mean=0, variance=1) for clustering shapes rather than absolute magnitudes
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the DTW K-Means model
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=random_state)
    labels = km.fit_predict(X_scaled)
    
    return labels, km

def assign_clusters_to_df(df, indices, labels):
    """
    Maps the cluster labels back to the original dataframe.
    """
    cluster_map = pd.DataFrame(index=indices)
    cluster_map['CLUSTER'] = labels
    cluster_map = cluster_map.reset_index()
    
    if 'DATE' not in df.columns:
        df['DATE'] = df['DATE_TIME'].dt.date
    
    merged_df = pd.merge(df, cluster_map, on=['INVERTER_ID', 'DATE'], how='left')
    return merged_df

if __name__ == "__main__":
    pass
