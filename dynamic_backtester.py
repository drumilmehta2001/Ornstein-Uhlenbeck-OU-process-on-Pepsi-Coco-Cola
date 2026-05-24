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
    
    # Protect against divide by zero
    df['Z_Score'] = np.where(df['Rolling_Std'] == 0, 0, df['Spread'] / df['Rolling_Std'])
    df['Position'] = 0.0
    
    # --- TRADING LOGIC ---
    for t in range(30, len(df)):
        current_z = df['Z_Score'].iloc[t]
        prev_pos = df['Position'].iloc[t-1]
        
        if prev_pos == 0:
            if current_z > 2.0:
                df.iloc[t, df.columns.get_loc('Position')] = -1 # Short Y, Long X
            elif current_z < -2.0:
                df.iloc[t, df.columns.get_loc('Position')] = 1  # Long Y, Short X
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 0
        elif prev_pos == -1:
            if current_z <= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
            else:
                df.iloc[t, df.columns.get_loc('Position')] = -1
        elif prev_pos == 1:
            if current_z >= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 1

    df = df.dropna()
    
    # --- DYNAMIC PNL CALCULATION ---
    # Holding 1 share of Y means holding -beta shares of X. 
    # Beta changes daily! PnL depends on the exact beta at time t-1.
    df['Y_Return_Abs'] = df['Y'].diff()
    df['X_Return_Abs'] = df['X'].diff()
    
    # Daily PnL = Position Yesterday * [Change in Y today - (Beta Yesterday * Change in X today)]
    df['Daily_PnL'] = df['Position'].shift(1) * (df['Y_Return_Abs'] - df['Dynamic_Beta'].shift(1) * df['X_Return_Abs'])
    df['Daily_PnL'] = df['Daily_PnL'].fillna(0)
    df['Cumulative_PnL'] = df['Daily_PnL'].cumsum()
    
    print("\nBacktest Complete!")
    print(f"Total Return (Dynamically Hedged): ${df['Cumulative_PnL'].iloc[-1]:.2f}")
    
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
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.axhline(0, color='black')
    
    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
    plt.scatter(long_entries.index, long_entries['Z_Score'], color='green', marker='^', s=100, label="Enter Long")
    plt.scatter(short_entries.index, short_entries['Z_Score'], color='red', marker='v', s=100, label="Enter Short")
    plt.title("Z-Score Trading Signals")
    plt.legend()
    plt.grid(True)
    
    # 3. Equity Curve
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Cumulative_PnL'], label='Cumulative Profit ($)', color='blue', linewidth=2)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] >= 0), color='green', alpha=0.3)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] < 0), color='red', alpha=0.3)
    plt.title("Dynamically Hedged Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Profit ($)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("kalman_backtest.png")
    print("Saved equity curve to 'kalman_backtest.png'")

if __name__ == "__main__":
    run_dynamic_backtest('JPM', 'PNC', '2022-01-01', '2024-01-01')
