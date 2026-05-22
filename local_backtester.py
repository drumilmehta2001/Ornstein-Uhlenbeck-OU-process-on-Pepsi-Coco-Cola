import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

def run_backtest(ticker_y, ticker_x, start_date, end_date):
    print(f"Fetching Out-Of-Sample Data for {ticker_y} and {ticker_x} from {start_date} to {end_date}...")
    data = yf.download([ticker_y, ticker_x], start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1).dropna() # Ensure clean data
    
    if ticker_y not in data.columns or ticker_x not in data.columns:
        print("Data fetch failed.")
        return
        
    y = data[ticker_y]
    x = data[ticker_x]
    
    # We use a 60-day Rolling Window to prevent Look-Ahead Bias
    window = 60
    
    df = pd.DataFrame(index=data.index)
    df['Y'] = y
    df['X'] = x
    df['Hedge_Ratio'] = np.nan
    df['Spread'] = np.nan
    df['Rolling_Mean'] = np.nan
    df['Rolling_Std'] = np.nan
    df['Z_Score'] = np.nan
    df['Position'] = 0.0 # 1 = Long Spread, -1 = Short Spread, 0 = Flat
    
    print("Simulating trading days... (Calculating rolling logic strictly on past data)")
    # Loop over time. On day 't', we can only look at data from [t-60] to [t-1].
    for t in range(window, len(df)):
        # Historical window strictly BEFORE today
        hist_y = df['Y'].iloc[t-window:t]
        hist_x = df['X'].iloc[t-window:t]
        
        # Calculate Hedge Ratio using OLS on historical window
        hist_x_with_constant = sm.add_constant(hist_x)
        model = sm.OLS(hist_y, hist_x_with_constant).fit()
        hedge_ratio = model.params.iloc[1]
        
        # Calculate historical spread to find mean/std
        hist_spread = hist_y - (hedge_ratio * hist_x)
        rolling_mean = hist_spread.mean()
        rolling_std = hist_spread.std()
        
        # Now step into TODAY (t)
        current_y = df['Y'].iloc[t]
        current_x = df['X'].iloc[t]
        
        # Calculate TODAY's spread using YESTERDAY'S hedge ratio (realistic trading)
        current_spread = current_y - (hedge_ratio * current_x)
        
        # Protect against ZeroDivisionError
        if rolling_std == 0:
            current_z = 0
        else:
            current_z = (current_spread - rolling_mean) / rolling_std
        
        # Store values
        df.iloc[t, df.columns.get_loc('Hedge_Ratio')] = hedge_ratio
        df.iloc[t, df.columns.get_loc('Spread')] = current_spread
        df.iloc[t, df.columns.get_loc('Rolling_Mean')] = rolling_mean
        df.iloc[t, df.columns.get_loc('Rolling_Std')] = rolling_std
        df.iloc[t, df.columns.get_loc('Z_Score')] = current_z
        
        # --- TRADING LOGIC ---
        prev_pos = df['Position'].iloc[t-1]
        
        if prev_pos == 0:
            # Enter Trade
            if current_z > 2.0:
                df.iloc[t, df.columns.get_loc('Position')] = -1 # Short Spread
            elif current_z < -2.0:
                df.iloc[t, df.columns.get_loc('Position')] = 1  # Long Spread
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 0
        elif prev_pos == -1:
            # Exit Short Trade when Z crosses 0
            if current_z <= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
            else:
                df.iloc[t, df.columns.get_loc('Position')] = -1 # Hold
        elif prev_pos == 1:
            # Exit Long Trade when Z crosses 0
            if current_z >= 0:
                df.iloc[t, df.columns.get_loc('Position')] = 0
            else:
                df.iloc[t, df.columns.get_loc('Position')] = 1 # Hold

    # Drop the initial 60 days where we had no positions
    df = df.dropna()
    
    # --- PnL CALCULATION ---
    # Daily PnL = Position yesterday * Change in Spread today
    df['Spread_Diff'] = df['Spread'].diff()
    df['Daily_PnL'] = df['Position'].shift(1) * df['Spread_Diff']
    df['Daily_PnL'] = df['Daily_PnL'].fillna(0)
    df['Cumulative_PnL'] = df['Daily_PnL'].cumsum()
    
    print("\nBacktest Complete!")
    print(f"Total Return (Cumulative PnL for trading 1 share of {ticker_y}): ${df['Cumulative_PnL'].iloc[-1]:.2f}")
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 10))
    
    # Top Subplot: Z-Score and Signals
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Z_Score'], label='Rolling Z-Score', color='purple')
    plt.axhline(2.0, color='red', linestyle='--', label='+2 (Short Entry)')
    plt.axhline(-2.0, color='green', linestyle='--', label='-2 (Long Entry)')
    plt.axhline(0, color='black', linestyle='-', label='Mean (Exit)')
    plt.title(f"Out-of-Sample Walk-Forward Backtest: {ticker_y} / {ticker_x} (2022-2024)")
    plt.ylabel("Z-Score")
    
    # Overlay entry signals
    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
    plt.scatter(long_entries.index, long_entries['Z_Score'], color='green', marker='^', s=100, zorder=5)
    plt.scatter(short_entries.index, short_entries['Z_Score'], color='red', marker='v', s=100, zorder=5)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Bottom Subplot: Equity Curve
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Cumulative_PnL'], label='Cumulative Profit ($)', color='blue', linewidth=2)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] >= 0), color='green', alpha=0.3)
    plt.fill_between(df.index, df['Cumulative_PnL'], 0, where=(df['Cumulative_PnL'] < 0), color='red', alpha=0.3)
    plt.title("Strategy Equity Curve (Out of Sample)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL ($)")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("oos_backtest.png")
    print("Saved equity curve to 'oos_backtest.png'")

if __name__ == "__main__":
    # We test JPM and PNC since they won the 2020-2022 scan!
    run_backtest('JPM', 'PNC', '2022-01-01', '2024-01-01')
