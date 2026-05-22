import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Import our OU estimation function from the previous script
from ou_basic import estimate_ou_parameters

# Step 1: Data Fetching
def fetch_data(tickers, start_date, end_date):
    print(f"Fetching daily close prices for {tickers} from {start_date} to {end_date}...")
    # We download 'Close' prices (could also use 'Adj Close')
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    # Drop missing values to ensure the arrays match perfectly in length
    data = data.dropna()
    print("Data successfully fetched.\n")
    return data

# Step 2: Cointegration Test
def test_cointegration(y, x):
    print("Running Engle-Granger Cointegration Test...")
    # The coint function returns the t-statistic, p-value, and critical values
    score, pvalue, _ = coint(y, x)
    print(f"Cointegration p-value: {pvalue:.4f}")
    
    if pvalue < 0.05:
        print("Result: STRONG evidence of cointegration (p < 0.05). Safe to trade.")
    else:
        print("Result: WEAK/NO evidence of cointegration (p > 0.05). Do NOT trade.")
    print("-" * 40)
    return pvalue

# Step 3: Calculate Spread and Hedge Ratio
def calculate_spread(y, x):
    # We use Ordinary Least Squares (OLS) regression to find the Hedge Ratio
    # Equation: y = beta * x + alpha
    x_with_constant = sm.add_constant(x)
    model = sm.OLS(y, x_with_constant).fit()
    
    # The 'beta' coefficient is our Hedge Ratio
    hedge_ratio = model.params.iloc[1] 
    
    print(f"Calculated Hedge Ratio: {hedge_ratio:.4f}")
    print(f"This means: For every 1 share of PEP shorted, you buy {hedge_ratio:.4f} shares of KO.")
    
    # Construct our synthetic "Spread" asset
    spread = y - (hedge_ratio * x)
    print("-" * 40)
    return spread, hedge_ratio

def main():
    # Define our universe and timeframe
    ticker_y = 'PEP'
    ticker_x = 'KO'
    
    # We will test on a 2-year historical window
    start_date = '2022-01-01'
    end_date = '2024-01-01' 
    
    # Execute Pipeline
    data = fetch_data([ticker_y, ticker_x], start_date, end_date)
    y = data[ticker_y]
    x = data[ticker_x]
    
    pvalue = test_cointegration(y, x)
    spread, hedge_ratio = calculate_spread(y, x)
    
    # Step 4: OU Parameter Estimation
    print("Estimating OU Parameters on the Spread...")
    # Convert pandas series to numpy array for the math function
    S = spread.values
    
    # For daily trading data, dt is often represented as 1/252 (trading days in a year)
    dt = 1.0 / 252.0 
    
    theta, mu, sigma = estimate_ou_parameters(S, dt)
    
    print(f"Long-term Mean (Mu): {mu:.4f}")
    print(f"Volatility (Sigma): {sigma:.4f}")
    print(f"Mean Reversion Speed (Theta): {theta:.4f}")
    
    # Calculate half-life of mean reversion (how long it typically takes to snap back half the distance)
    if theta > 0:
        half_life_years = np.log(2) / theta
        half_life_days = half_life_years * 252
        print(f"Half-life of mean reversion: ~{half_life_days:.1f} trading days.")
    else:
        print("Warning: Theta is negative. The spread is structurally diverging, NOT reverting.")
    print("-" * 40)
    
    # Step 5: Generate Z-Scores and Plot
    print("Generating trading signals and rendering plot...")
    # Normalize the spread into a Z-score: (Current - Mean) / Volatility
    z_score = (spread - mu) / sigma
    
    plt.figure(figsize=(12, 8))
    
    # --- Top Plot: Raw Spread ---
    plt.subplot(2, 1, 1)
    plt.plot(spread.index, spread.values, label='Historical Spread', color='blue')
    plt.axhline(mu, color='black', linestyle='--', label=f'Mean ($\mu$ = {mu:.2f})')
    
    # Trading Bands: Enter short at +2 sigma, Enter long at -2 sigma
    plt.axhline(mu + (2 * sigma), color='red', linestyle='--', label='Sell Signal (+2$\sigma$)')
    plt.axhline(mu - (2 * sigma), color='green', linestyle='--', label='Buy Signal (-2$\sigma$)')
    
    plt.title(f'Pairs Trading Spread: {ticker_y} - ({hedge_ratio:.2f} * {ticker_x})')
    plt.ylabel('Spread Price ($)')
    plt.legend()
    plt.grid(True)
    
    # --- Bottom Plot: Z-Score ---
    plt.subplot(2, 1, 2)
    plt.plot(z_score.index, z_score.values, label='Z-Score', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2.0, color='red', linestyle='--', label='Short Entry Threshold')
    plt.axhline(-2.0, color='green', linestyle='--', label='Long Entry Threshold')
    
    plt.title('Normalized Z-Score (Trading Signals)')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("pairs_trading_plot.png")
    print("Analysis complete. Plot saved as 'pairs_trading_plot.png'.")

if __name__ == "__main__":
    main()
