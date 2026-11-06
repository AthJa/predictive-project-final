import pandas as pd
import numpy as np

def calculate_residuals(df, actual_col='AC_POWER', expected_col='PREDICTED_AC_POWER'):
    """
    Calculates the residual (loss) for each timestamp. 
    Positive residual means underperformance (expected > actual).
    We clip at 0 because we only care about underperformance.
    """
    df['RESIDUAL_KW'] = np.maximum(0, df[expected_col] - df[actual_col])
    return df

def maintenance_optimization(df, ppa_rate_per_mwh=50, maintenance_cost=500):
    """
    Calculates rolling 7-day financial loss due to underperformance and flags for maintenance.
    Assumes 15-minute intervals.
    """
    # Convert kW to kWh for each 15-minute interval (divide by 4)
    df['RESIDUAL_KWH'] = df['RESIDUAL_KW'] / 4.0
    
    # Calculate rolling 7-day sum of lost kWh
    # 7 days = 7 * 24 * 4 = 672 intervals
    # Must sort by time before rolling
    df = df.sort_values(by=['INVERTER_ID', 'DATE_TIME'])
    
    df['ROLLING_7D_LOSS_KWH'] = df.groupby('INVERTER_ID')['RESIDUAL_KWH'].transform(
        lambda x: x.rolling(window=672, min_periods=1).sum()
    )
    
    # Convert kWh to MWh
    df['ROLLING_7D_LOSS_MWH'] = df['ROLLING_7D_LOSS_KWH'] / 1000.0
    
    # Calculate financial loss
    df['FINANCIAL_LOSS_7D'] = df['ROLLING_7D_LOSS_MWH'] * ppa_rate_per_mwh
    
    # Trigger maintenance flag if financial loss > maintenance cost
    df['MAINTENANCE_REQUIRED'] = df['FINANCIAL_LOSS_7D'] > maintenance_cost
    
    return df

if __name__ == "__main__":
    pass
