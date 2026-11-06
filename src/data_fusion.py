import pandas as pd
import numpy as np

def load_and_merge_data(generation_path, weather_path):
    """
    Loads generation and weather data, parses dates, and merges them.
    """
    gen_df = pd.read_csv(generation_path)
    weather_df = pd.read_csv(weather_path)
    
    # Plant 1 format: DD-MM-YYYY HH:MM or YYYY-MM-DD HH:MM:SS
    # Plant 2 might have a different format, so we use infer_datetime_format or multiple formats
    try:
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    except ValueError:
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
        
    try:
        weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
    
    gen_df = gen_df.rename(columns={'SOURCE_KEY': 'INVERTER_ID'})
    weather_df = weather_df.rename(columns={'SOURCE_KEY': 'SENSOR_ID'})
    
    merged_df = pd.merge(gen_df, weather_df, on=['DATE_TIME', 'PLANT_ID'], how='left')
    merged_df = merged_df.sort_values(by=['INVERTER_ID', 'DATE_TIME']).reset_index(drop=True)
    
    return merged_df

def handle_missing_data(df):
    """
    Handles missing values:
    - Linear Interpolation: AC_POWER, DC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE
    - Forward Fill (ffill): DAILY_YIELD, TOTAL_YIELD
    """
    cols_to_interpolate = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    cols_to_ffill = ['DAILY_YIELD', 'TOTAL_YIELD']
    
    for col in cols_to_interpolate:
        if col in df.columns:
            df[col] = df.groupby('INVERTER_ID')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
            
    for col in cols_to_ffill:
        if col in df.columns:
            df[col] = df.groupby('INVERTER_ID')[col].transform(lambda x: x.ffill().bfill())
            
    return df

def feature_engineering(df):
    """
    Adds derived features like time-of-day and DC-to-AC efficiency ratio.
    """
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['MINUTE'] = df['DATE_TIME'].dt.minute
    df['TIME_OF_DAY'] = df['HOUR'] + df['MINUTE'] / 60.0
    
    # DC-to-AC ratio (avoid div by zero, but note that AC is typically smaller than DC)
    # AC power is typically in kW and DC power in kW or W (for this dataset they seem roughly 1:10 ratio, actually DC is *10 of AC usually)
    df['DC_AC_RATIO'] = np.where(df['DC_POWER'] > 0, df['AC_POWER'] / df['DC_POWER'], 0)
    
    return df

def preprocess_pipeline(generation_path, weather_path):
    merged = load_and_merge_data(generation_path, weather_path)
    imputed = handle_missing_data(merged)
    engineered = feature_engineering(imputed)
    return engineered

if __name__ == "__main__":
    import os
    gen_path = os.path.join('..', 'dataset', 'Plant_1_Generation_Data.csv')
    weather_path = os.path.join('..', 'dataset', 'Plant_1_Weather_Sensor_Data.csv')
    
    if os.path.exists(gen_path) and os.path.exists(weather_path):
        df = preprocess_pipeline(gen_path, weather_path)
        print("Preprocessing completed. Dataset shape:", df.shape)
        print(df.head())
