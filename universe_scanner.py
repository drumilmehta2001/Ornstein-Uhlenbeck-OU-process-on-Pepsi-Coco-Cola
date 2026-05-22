import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from ou_basic import estimate_ou_parameters

# Ignore warnings from statsmodels for cleaner terminal output
warnings.filterwarnings("ignore")

def fetch_universe_data(tickers, start_date, end_date):
    print(f"Fetching daily close prices for {len(tickers)} Bank stocks...")
    # Download data silently
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # Drop columns (tickers) that have NaN values to ensure perfectly matching arrays
    data = data.dropna(axis=1)
    
    successful_tickers = list(data.columns)
    print(f"Successfully fetched complete data for {len(successful_tickers)} stocks.")
    return data, successful_tickers

def calculate_spread(y, x):
    x_with_constant = sm.add_constant(x)
    model = sm.OLS(y, x_with_constant).fit()
    hedge_ratio = model.params.iloc[1]
    spread = y - (hedge_ratio * x)
    return spread, hedge_ratio

def scan_universe(data, tickers):
    print(f"\nGenerating pairs and scanning... (This will run hundreds of regressions)")
    
    # Generate all unique combinations of the tickers
    pairs = list(itertools.combinations(tickers, 2))
    print(f"Total unique pairs to scan: {len(pairs)}\n")
    
    results = []
    dt = 1.0 / 252.0 # Daily data assumption
    
    # Loop through every pair
    for count, (ticker_y, ticker_x) in enumerate(pairs):
        if count > 0 and count % 50 == 0:
            print(f"Scanned {count} / {len(pairs)} pairs...")
            
        y = data[ticker_y]
        x = data[ticker_x]
        
        # 1. Run Cointegration Test
        try:
            score, pvalue, _ = coint(y, x)
        except:
            continue # Skip if math fails on weird data
            
        # 2. Filter: Only proceed if there is strong mathematical cointegration
        if pvalue < 0.05:
            # 3. Calculate Spread
            spread, hedge_ratio = calculate_spread(y, x)
            S = spread.values
            
            # 4. Estimate OU Parameters
            try:
                theta, mu, sigma = estimate_ou_parameters(S, dt)
            except:
                continue
                
            # We only want pairs that actually snap back (positive theta)
            if theta > 0:
                half_life = np.log(2) / theta * 252
                results.append({
                    'Pair': f"{ticker_y} / {ticker_x}",
                    'P-Value': round(pvalue, 4),
                    'Theta (Speed)': round(theta, 2),
                    'Half-Life (Days)': round(half_life, 1),
                    'Hedge Ratio': round(hedge_ratio, 3)
                })
                
    return pd.DataFrame(results)

def main():
    # Define a basket of US Banks (Major and Regional)
    bank_tickers = [
        'JPM', 'BAC', 'WFC', 'C', 'MS', 'GS', 'USB', 'PNC', 'TFC', 
        'COF', 'BK', 'STT', 'FITB', 'MTB', 'HBAN', 'KEY', 'RF', 
        'CFG', 'SYF', 'DFS', 'CMA'
    ]
    
    start_date = '2020-01-01'
    end_date = '2022-01-01'
    
    print("=" * 60)
    print("PHASE 2: BANKING UNIVERSE PAIRS SCANNER")
    print("=" * 60)
    
    # Run the pipeline
    data, valid_tickers = fetch_universe_data(bank_tickers, start_date, end_date)
    results_df = scan_universe(data, valid_tickers)
    
    if results_df.empty:
        print("\nNo highly cointegrated pairs found in this universe during this timeframe.")
        return
        
    # Sort the results to find the Absolute Best pairs
    # We want the lowest p-value (highest certainty) and the highest Theta (fastest snap-back)
    
    top_by_pvalue = results_df.sort_values(by='P-Value', ascending=True).head(5)
    top_by_theta = results_df.sort_values(by='Theta (Speed)', ascending=False).head(5)
    
    print("\n" + "=" * 60)
    print("TOP 5 PAIRS BY STATISTICAL CERTAINTY (Lowest P-Value)")
    print("=" * 60)
    print(top_by_pvalue[['Pair', 'P-Value', 'Theta (Speed)', 'Half-Life (Days)']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TOP 5 PAIRS BY MEAN REVERSION SPEED (Highest Theta)")
    print("=" * 60)
    print(top_by_theta[['Pair', 'P-Value', 'Theta (Speed)', 'Half-Life (Days)']].to_string(index=False))
    print("\nScan Complete!")

if __name__ == "__main__":
    main()
