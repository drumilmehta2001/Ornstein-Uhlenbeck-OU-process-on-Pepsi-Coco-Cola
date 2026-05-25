import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

warnings.filterwarnings("ignore")

def run_johansen_basket(tickers, start_date, end_date):
    print(f"Fetching Data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna()
    
    print("Running Johansen Cointegration Test on Multi-Dimensional Space...")
    # det_order=0 (no deterministic trend), k_ar_diff=1 (lags)
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
    
    # The eigenvectors are in the columns of `evec`.
    # We want the eigenvector associated with the largest eigenvalue (the most stationary portfolio)
    weights = johansen_test.evec[:, 0]
    
    print("\n" + "="*50)
    print("JOHANSEN COINTEGRATION WEIGHTS (THE BASKET)")
    print("="*50)
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {weights[i]:.4f} shares")
        
    print("\nThis means our perfectly hedged, multi-asset portfolio is:")
    portfolio_equation = " + ".join([f"({weights[i]:.2f} * {ticker})" for i, ticker in enumerate(tickers)])
    print(portfolio_equation)
    print("="*50 + "\n")
    
    # Calculate the Spread of the Basket
    # Spread = w1*P1 + w2*P2 + w3*P3 + w4*P4
    spread = np.dot(data.values, weights)
    
    df = pd.DataFrame(index=data.index)
    df['Spread'] = spread
    df['Mean'] = df['Spread'].mean()
    df['Upper'] = df['Mean'] + 2 * df['Spread'].std()
    df['Lower'] = df['Mean'] - 2 * df['Spread'].std()
    
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Spread'], label='4-Asset Basket Spread', color='purple')
    plt.axhline(df['Mean'].iloc[0], color='black', label='Mean')
    plt.plot(df.index, df['Upper'], color='red', linestyle='--', label='+2 Std Dev')
    plt.plot(df.index, df['Lower'], color='green', linestyle='--', label='-2 Std Dev')
    
    plt.title(f"Johansen Cointegration: 4-Asset Statistical Arbitrage\n{portfolio_equation}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("johansen_basket.png")
    print("Saved basket chart to 'johansen_basket.png'")

if __name__ == "__main__":
    # Scanning the 2020-2022 Formation Period for a 4-stock basket
    run_johansen_basket(['JPM', 'BAC', 'PNC', 'WFC'], '2020-01-01', '2022-01-01')
