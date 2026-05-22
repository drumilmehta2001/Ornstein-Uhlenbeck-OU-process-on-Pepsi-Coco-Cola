# Mean Reversion via Ornstein-Uhlenbeck Process

This repository contains a foundational implementation of a mean-reverting trading model using the Ornstein-Uhlenbeck (OU) stochastic process, along with real-world market scanners and out-of-sample backtesting engines.

## 1. What's going on under the hood?

If you are new to quantitative finance, predicting whether a stock will go up or down on a given day is incredibly difficult. However, some financial instruments (like two highly correlated stocks) tend to wander away from their average price but eventually snap back to it. This is called **Mean Reversion**.

### The Core Concept: Ergodicity
Before writing any code, we must assume the system is **ergodic**. 
Ergodicity means that the long-term historical average of an asset is a reliable predictor of its future expected average. If a process is *not* ergodic, a mean reversion strategy will fail spectacularly because the "mean" it is waiting to revert to no longer exists.

### The Math: Ornstein-Uhlenbeck (OU) Process
The OU process is the mathematical gold standard for modeling this "snap-back" behavior. It models the dynamic using three key parameters:
*   `$\mu$` (**Mu**): The long-term mean (the equilibrium point).
*   `$\theta$` (**Theta**): The speed of mean reversion (how fast it snaps back).
*   `$\sigma$` (**Sigma**): Volatility (random market noise).

---

## 2. The Universe Scanner (`universe_scanner.py`)
To find pairs that actually work, we built an automated scanner. Instead of guessing, this script downloads historical data for an entire sector (e.g., Regional and Major US Banks), mathematically generates every possible pairwise combination, and runs the **Engle-Granger Cointegration Test**. 

It ranks pairs by statistical certainty (p-value) and Mean Reversion Speed ($\theta$) to find the pairs with the tightest "leash".

---

## 3. Out-of-Sample Backtesting (`local_backtester.py`)
To avoid **Selection Bias** (cherry-picking a winning pair after the fact), we implemented a strict Walk-Forward backtest:
*   **Formation Period (2020-2022):** We ran the scanner and identified `JPM` and `PNC` as the most cointegrated banking pair of that era.
*   **Trading Period (2022-2024):** We locked in that pair and simulated trading it on unseen future data. We used a strict 60-day rolling window to calculate the Hedge Ratio and Mean, ensuring the algorithm could not "cheat" by looking ahead.

### The Real-World Lesson: Structural Breaks
During our out-of-sample backtest of `JPM` and `PNC`, the algorithm technically closed with a profit, but the Equity Curve revealed massive drawdowns and holding periods lasting up to 5 months. 

**Why? The Spring 2023 Regional Banking Crisis.** 
When Silicon Valley Bank collapsed, investors panic-sold regional banks (`PNC`) and fled to "safe haven" mega-banks (`JPM`). The fundamental, macroeconomic relationship between the two stocks structurally broke down. The statistical "rubber band" snapped.

### The Solution: Risk Management
This backtest perfectly highlighted a fatal flaw in raw statistical arbitrage. A pure math algorithm will blindly hold a broken pair until it blows up your account. To make this strategy viable for live trading, it requires strict risk management:
1.  **Time Stops:** If a pair's historical Half-Life is 11 days, and the trade hasn't reverted in 22 days, force-close the position. 
2.  **Z-Score Stop Loss:** If we enter a trade at a Z-Score of `+2.0`, and the spread blows out to `+4.0`, a structural macroeconomic break has occurred. Cut the loss immediately to avoid margin calls.
