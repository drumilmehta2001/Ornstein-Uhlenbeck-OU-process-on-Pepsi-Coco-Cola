# Mean Reversion via Ornstein-Uhlenbeck Process

This repository contains a foundational implementation of a mean-reverting trading model using the Ornstein-Uhlenbeck (OU) stochastic process.

## What's going on under the hood?

If you are new to quantitative finance, predicting whether a stock will go up or down on a given day is incredibly difficult. However, some financial instruments (or combinations of them, like two highly correlated stocks) tend to wander away from their average price but eventually snap back to it. This is called **Mean Reversion**.

### 1. The Core Concept: Ergodicity
Before writing any code, we must assume the system is **ergodic**. 
In simple terms, ergodicity means that the long-term historical average of an asset is a reliable predictor of its future expected average. If a process is *not* ergodic (e.g., a company's fundamentals change permanently and the stock drops to 0), a mean reversion strategy will fail spectacularly because the "mean" it is waiting to revert to no longer exists.

### 2. The Math: Ornstein-Uhlenbeck (OU) Process
The OU process is the mathematical gold standard for modeling this "snap-back" behavior continuously over time. 

Imagine a rubber band attached to a wall. As you pull it away, the resistance gets stronger the further you pull. 
The OU process models this exact dynamic using three key parameters:
*   `$\mu$` (**Mu**): The long-term mean (the wall where the rubber band is attached).
*   `$\theta$` (**Theta**): The speed of mean reversion (how stiff the rubber band is). A high Theta means it snaps back violently; a low Theta means it wanders slowly.
*   `$\sigma$` (**Sigma**): Volatility (random market noise trying to push the price around).

### 3. How the Code Works
The `ou_basic.py` script does two things:
1.  **Simulation:** The `simulate_ou_process` function creates artificial, randomized price data that mathematically behaves exactly like our rubber band.
2.  **Estimation:** The `estimate_ou_parameters` function acts like a detective. If we give it raw price data, it uses Ordinary Least Squares (OLS) Linear Regression to reverse-engineer the data and calculate the exact stiffness of the rubber band ($\theta$) and the equilibrium point ($\mu$).

In a live trading system, instead of using simulated data, we feed historical stock prices into this estimation function to determine if a stock exhibits strong mean-reverting properties.
