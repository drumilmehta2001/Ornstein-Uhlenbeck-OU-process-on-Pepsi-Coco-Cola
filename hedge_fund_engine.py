import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pykalman import KalmanFilter
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

def run_hedge_fund_engine(ticker_y, ticker_x, start_date, end_date):
    print(f"Fetching Data for {ticker_y} and {ticker_x} from {start_date} to {end_date}...")
    data = yf.download([ticker_y, ticker_x], start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1).dropna()
    
    y = data[ticker_y].values
    x = data[ticker_x].values
    
    # 1. Kalman Filter (Dynamic Hedging)
    print("1. Running continuous Kalman Filter...")
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                      initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=np.eye(2) * 1e-4) 
                      
    state_means, _ = kf.filter(y)
    hedge_ratios = state_means[:, 0]
    intercepts = state_means[:, 1]
    
    predictions = (hedge_ratios * x) + intercepts
    errors = y - predictions
    
    df = pd.DataFrame(index=data.index)
    df['Y'] = y
    df['X'] = x
    df['Dynamic_Beta'] = hedge_ratios
    df['Spread'] = errors 
    df['Rolling_Std'] = df['Spread'].rolling(window=30).std()
    df['Z_Score'] = np.where(df['Rolling_Std'] == 0, 0, df['Spread'] / df['Rolling_Std'])
    
    # 2. Hidden Markov Model (Regime Detection)
    print("2. Training Hidden Markov Model on volatility...")
    # Drop NaNs before training HMM
    hmm_data = df[['Rolling_Std']].dropna()
    
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    hmm.fit(hmm_data)
    regimes = hmm.predict(hmm_data)
    
    # Identify which regime is "Panic" (the one with the higher mean volatility)
    if hmm.means_[0][0] > hmm.means_[1][0]:
        panic_regime_id = 0
    else:
        panic_regime_id = 1
        
    df['Regime'] = np.nan
    df.loc[hmm_data.index, 'Regime'] = regimes
    df['Is_Panic'] = df['Regime'] == panic_regime_id
    df['Is_Panic'] = df['Is_Panic'].fillna(False)

    df['Position'] = 0.0
    
    # Risk Management Tracking
    days_in_trade = 0
    lockout = False
    time_stops = 0
    z_stops = 0
    normal_exits = 0
    
    print("3. Executing Trades (with Slippage & AI Dynamic Entry)...")
    for t in range(30, len(df)):
        current_z = df['Z_Score'].iloc[t]
        prev_z = df['Z_Score'].iloc[t-1]
        prev_pos = df['Position'].iloc[t-1]
        is_panic = df['Is_Panic'].iloc[t]
        
        # AI Dynamic Entry Threshold
        ENTRY_Z = 3.0 if is_panic else 2.0
        
        # Lockout clear
        if lockout:
            if (current_z >= 0 and prev_z < 0) or (current_z <= 0 and prev_z > 0):
                lockout = False
            df.iloc[t, df.columns.get_loc('Position')] = 0
            continue 
            
        if prev_pos == 0:
            if current_z > ENTRY_Z:
                df.iloc[t, df.columns.get_loc('Position')] = -1
                days_in_trade = 1
            elif current_z < -ENTRY_Z:
                df.iloc[t, df.columns.get_loc('Position')] = 1
                days_in_trade = 1
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 0
                
        else:
            days_in_trade += 1
            
            # 1. Z-Score Stop Loss
            if abs(current_z) > 4.5: # Widen slightly because panic enters at 3.0
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
    
    # 4. SLIPPAGE AND TRANSACTION COSTS
    df['Y_Return_Abs'] = df['Y'].diff()
    df['X_Return_Abs'] = df['X'].diff()
    
    # Raw Daily PnL
    df['Raw_Daily_PnL'] = df['Position'].shift(1) * (df['Y_Return_Abs'] - df['Dynamic_Beta'].shift(1) * df['X_Return_Abs'])
    df['Raw_Daily_PnL'] = df['Raw_Daily_PnL'].fillna(0)
    
    # Calculate Slippage (Transaction Cost = 0.1% (0.001) per positional change)
    # Total nominal exposure = 1 share of Y + abs(Beta) shares of X. 
    df['Nominal_Exposure'] = df['Y'] + abs(df['Dynamic_Beta']) * df['X']
    
    # The "amount" of trading we did today is proportional to the change in our Position scalar
    df['Position_Change'] = df['Position'].diff().fillna(0).abs()
    
    # Slippage penalty: 0.1% of the nominal exposure every time we enter or exit a trade
    df['Slippage_Penalty'] = df['Position_Change'] * df['Nominal_Exposure'] * 0.001
    
    # Final Net PnL
    df['Net_Daily_PnL'] = df['Raw_Daily_PnL'] - df['Slippage_Penalty']
    df['Cumulative_Net_PnL'] = df['Net_Daily_PnL'].cumsum()
    df['Cumulative_Raw_PnL'] = df['Raw_Daily_PnL'].cumsum()
    
    print("\n" + "="*45)
    print("HEDGE FUND ENGINE COMPLETE (SLIPPAGE + HMM AI)")
    print("="*45)
    print(f"Total Raw Return: ${df['Cumulative_Raw_PnL'].iloc[-1]:.2f}")
    print(f"Total Slippage Paid: -${df['Slippage_Penalty'].sum():.2f}")
    print(f"Total NET Return: ${df['Cumulative_Net_PnL'].iloc[-1]:.2f}")
    print(f"Normal Take-Profit Exits: {normal_exits}")
    print(f"Stop-Loss Triggers (Z-Score > 4.5): {z_stops}")
    print(f"Stop-Loss Triggers (Time > 21d): {time_stops}")
    print("="*45 + "\n")
    
    # VISUALIZATION
    plt.figure(figsize=(16, 14))
    
    # 1. HMM Regimes
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Rolling_Std'], color='black', label='Spread Volatility')
    
    # Highlight Panic Regimes
    plt.fill_between(df.index, plt.ylim()[0], plt.ylim()[1], where=df['Is_Panic'], color='red', alpha=0.3, label='Panic Regime (HMM)')
    plt.fill_between(df.index, plt.ylim()[0], plt.ylim()[1], where=~df['Is_Panic'], color='green', alpha=0.1, label='Quiet Regime (HMM)')
    
    plt.title("HMM Regime Detection (Volatility)")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # 2. Z-Score and Dynamic Signals
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Z_Score'], label='Z-Score', color='purple')
    
    # Dynamic thresholds based on regime
    dynamic_upper = np.where(df['Is_Panic'], 3.0, 2.0)
    dynamic_lower = np.where(df['Is_Panic'], -3.0, -2.0)
    plt.plot(df.index, dynamic_upper, color='orange', linestyle='--', label='AI Dynamic Entry (+)')
    plt.plot(df.index, dynamic_lower, color='orange', linestyle='--', label='AI Dynamic Entry (-)')
    plt.axhline(0, color='black')
    
    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
    plt.scatter(long_entries.index, long_entries['Z_Score'], color='green', marker='^', s=100, label="Enter Long")
    plt.scatter(short_entries.index, short_entries['Z_Score'], color='red', marker='v', s=100, label="Enter Short")
    plt.title("Z-Score Trading Signals (With Dynamic AI Thresholds)")
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 3. Equity Curve (Raw vs Net)
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Cumulative_Raw_PnL'], label='Raw Profit (No Slippage)', color='grey', linestyle='--')
    plt.plot(df.index, df['Cumulative_Net_PnL'], label='NET Profit (After 0.1% Slippage)', color='blue', linewidth=2)
    
    plt.fill_between(df.index, df['Cumulative_Net_PnL'], 0, where=(df['Cumulative_Net_PnL'] >= 0), color='green', alpha=0.3)
    plt.fill_between(df.index, df['Cumulative_Net_PnL'], 0, where=(df['Cumulative_Net_PnL'] < 0), color='red', alpha=0.3)
    
    plt.title("Hedge Fund Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Profit ($)")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("hedge_fund_backtest.png")
    print("Saved equity curve to 'hedge_fund_backtest.png'")

if __name__ == "__main__":
    run_hedge_fund_engine('JPM', 'PNC', '2022-01-01', '2024-01-01')
