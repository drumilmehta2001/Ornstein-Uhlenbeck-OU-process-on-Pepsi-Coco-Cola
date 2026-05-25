import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pykalman import KalmanFilter

warnings.filterwarnings("ignore")

def run_dynamic_backtest(ticker_y, ticker_x, start_date, end_date):
    print(f"Fetching Data for {ticker_y} and {ticker_x} from {start_date} to {end_date}...")
    data = yf.download([ticker_y, ticker_x], start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1).dropna()
    
    y = data[ticker_y].values
    x = data[ticker_x].values
    
    print("Initializing Kalman Filter...")
    # Observation matrix needs to be reshaped for pykalman: [X_t, 1] for Beta and Alpha
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                      initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=np.eye(2) * 1e-4) # Random walk smoothness
                      
    print("Running continuous Kalman Filter (Day-by-Day updating)...")
    # We use 'filter' (forward pass only) to strictly prevent lookahead bias!
    state_means, state_covs = kf.filter(y)
    
    hedge_ratios = state_means[:, 0]
    intercepts = state_means[:, 1]
    
    # Calculate the Spread. The prediction error of the Kalman filter IS the spread.
    predictions = (hedge_ratios * x) + intercepts
    errors = y - predictions
    
    df = pd.DataFrame(index=data.index)
    df['Y'] = y
    df['X'] = x
    df['Dynamic_Beta'] = hedge_ratios
    df['Spread'] = errors 
    
    # We still need a rolling standard deviation to calculate a Z-score signal
    df['Rolling_Std'] = df['Spread'].rolling(window=30).std()
    
    # Create columns for risk tracking
    df['Z_Score'] = np.where(df['Rolling_Std'] == 0, 0, df['Spread'] / df['Rolling_Std'])
    df['Position'] = 0.0
    
    # State tracking variables
    days_in_trade = 0
    lockout = False
    
    # Stats tracking
    time_stops = 0
    z_stops = 0
    normal_exits = 0
    
    # --- TRADING LOGIC WITH RISK MANAGEMENT ---
    for t in range(30, len(df)):
        current_z = df['Z_Score'].iloc[t]
        prev_z = df['Z_Score'].iloc[t-1]
        prev_pos = df['Position'].iloc[t-1]
        
        # Check for Lockout clear (Z-score crosses 0)
        if lockout:
            if (current_z >= 0 and prev_z < 0) or (current_z <= 0 and prev_z > 0):
                lockout = False
            df.iloc[t, df.columns.get_loc('Position')] = 0
            continue # Skip trading this day
            
        if prev_pos == 0:
            if current_z > 2.0:
                df.iloc[t, df.columns.get_loc('Position')] = -1 # Short Y, Long X
                days_in_trade = 1
            elif current_z < -2.0:
                df.iloc[t, df.columns.get_loc('Position')] = 1  # Long Y, Short X
                days_in_trade = 1
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                
        else: # We are in an open trade
            days_in_trade += 1
            
            # 1. Z-Score Stop Loss
            if abs(current_z) > 4.0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                lockout = True
                z_stops += 1
                days_in_trade = 0
                
            # 2. Time Stop Loss
            elif days_in_trade > 21:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                lockout = True
                time_stops += 1
                days_in_trade = 0
                
            # 3. Normal Exit (Short)
            elif prev_pos == -1 and current_z <= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                normal_exits += 1
                days_in_trade = 0
                
            # 3. Normal Exit (Long)
            elif prev_pos == 1 and current_z >= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                normal_exits += 1
                days_in_trade = 0
                
            # 4. Hold
            else:
                df.iloc[t, df.columns.get_loc('Position')] = prev_pos

    df = df.dropna()
    
    # --- DYNAMIC PNL CALCULATION ---
    df['Y_Return_Abs'] = df['Y'].diff()
    df['X_Return_Abs'] = df['X'].diff()
    
    df['Daily_PnL'] = df['Position'].shift(1) * (df['Y_Return_Abs'] - df['Dynamic_Beta'].shift(1) * df['X_Return_Abs'])
    df['Daily_PnL'] = df['Daily_PnL'].fillna(0)
    df['Cumulative_PnL'] = df['Daily_PnL'].cumsum()
    
    print("\n" + "="*40)
    print("BACKTEST COMPLETE (WITH RISK MANAGEMENT)")
    print("="*40)
    print(f"Total Return (Dynamically Hedged): ${df['Cumulative_PnL'].iloc[-1]:.2f}")
    print(f"Normal Take-Profit Exits: {normal_exits}")
    print(f"Stop-Loss Triggers (Z-Score > 4): {z_stops}")
    print(f"Stop-Loss Triggers (Time > 21d): {time_stops}")
    print("="*40 + "\n")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 12))
    
    # 1. Dynamic Beta
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Dynamic_Beta'], color='orange', label=r'Kalman Filter Hedge Ratio ($\beta$)')
    plt.title(f"Dynamic Hedge Ratio ({ticker_y} / {ticker_x})")
    plt.legend()
    plt.grid(True)
    
    # 2. Z-Score and Signals
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Z_Score'], label='Z-Score', color='purple')
    plt.axhline(4.0, color='red', linestyle='--', label='Stop Loss Limit (+4.0)', linewidth=2)
    plt.axhline(-4.0, color='red', linestyle='--', label='Stop Loss Limit (-4.0)', linewidth=2)
    plt.axhline(2.0, color='grey', linestyle=':', label='+2.0')
    plt.axhline(-2.0, color='grey', linestyle=':', label='-2.0')
    plt.axhline(0, color='black')
    
    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
    plt.scatter(long_entries.index, long_entries['Z_Score'], color='green', marker='^', s=100, label="Enter Long")
    plt.scatter(short_entries.index, short_entries['Z_Score'], color='red', marker='v', s=100, label="Enter Short")
    plt.title("Z-Score Trading Signals (With Stop Losses)")
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 3. Equity Curve
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Cumulative_PnL'], label='Cumulative Profit ($)', color='blue', linewidth=2)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] >= 0), color='green', alpha=0.3)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] < 0), color='red', alpha=0.3)
    plt.title("Dynamically Hedged Equity Curve (Risk Managed)")
    plt.xlabel("Date")
    plt.ylabel("Profit ($)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("kalman_risk_backtest.png")
    print("Saved equity curve to 'kalman_risk_backtest.png'")

if __name__ == "__main__":
    run_dynamic_backtest('JPM', 'PNC', '2022-01-01', '2024-01-01')
