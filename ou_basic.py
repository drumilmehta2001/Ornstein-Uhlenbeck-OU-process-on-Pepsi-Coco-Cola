import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def simulate_ou_process(theta, mu, sigma, S0, dt, T):
    """
    Simulates an Ornstein-Uhlenbeck process.
    Equation: dS_t = theta * (mu - S_t) * dt + sigma * dW_t
    
    Parameters:
    theta: Rate of mean reversion (how fast it pulls back to the mean)
    mu: Long-term mean level
    sigma: Volatility (randomness)
    S0: Initial starting value
    dt: Time step
    T: Total time
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = S0
    
    for i in range(1, N):
        # dW is a standard Brownian motion step
        dW = np.random.normal(0, np.sqrt(dt))
        # Euler-Maruyama discretization method for solving the SDE
        S[i] = S[i-1] + theta * (mu - S[i-1]) * dt + sigma * dW
        
    return t, S

def estimate_ou_parameters(S, dt):
    """
    Estimates OU parameters (theta, mu, sigma) from a time series using linear regression.
    Rearranging the Euler-Maruyama discretization gives a linear equation:
    S_{t} - S_{t-1} = (theta * mu * dt) - (theta * dt) * S_{t-1} + error
    """
    # y is the change in price from one step to the next
    y = np.diff(S)
    # X is the previous price
    X = S[:-1].reshape(-1, 1)
    
    # Fit OLS regression
    reg = LinearRegression().fit(X, y)
    
    # The regression equation is y = a + b * X
    # Where a = theta * mu * dt
    # And b = -theta * dt
    b = reg.coef_[0]
    a = reg.intercept_
    
    # Back out the parameters from the regression coefficients
    theta = -b / dt
    mu = a / (theta * dt)
    
    # Calculate residuals to estimate the volatility (sigma)
    residuals = y - reg.predict(X)
    sigma = np.std(residuals) / np.sqrt(dt)
    
    return theta, mu, sigma

if __name__ == "__main__":
    # 1. Simulate an OU process
    print("Simulating OU Process...")
    true_theta = 5.0   # Speed of mean reversion
    true_mu = 100.0    # Long-term mean
    true_sigma = 2.0   # Volatility
    
    dt = 0.01 # Time step
    T = 10.0  # Total duration
    
    # Start the price at 105 to show it reverting down to the mean of 100
    t, S = simulate_ou_process(true_theta, true_mu, true_sigma, S0=105.0, dt=dt, T=T)
    
    # 2. Estimate parameters from the simulated data
    # In a real trading system, you'd feed historical price data into this function
    est_theta, est_mu, est_sigma = estimate_ou_parameters(S, dt)
    
    print("\nParameter Estimation Results:")
    print(f"True Theta: {true_theta:.2f} | Estimated Theta: {est_theta:.2f}")
    print(f"True Mu:    {true_mu:.2f} | Estimated Mu:    {est_mu:.2f}")
    print(f"True Sigma: {true_sigma:.2f} | Estimated Sigma: {est_sigma:.2f}")
    
    # 3. Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(t, S, label='Simulated Price Path')
    plt.axhline(true_mu, color='r', linestyle='--', label=r'Long-term Mean ($\mu$)')
    plt.title('Ornstein-Uhlenbeck (OU) Process Simulation')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig("ou_plot.png")
    print("Plot saved to ou_plot.png")
