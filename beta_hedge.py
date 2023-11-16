import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

tickers = "CVNA, QQQ"
start_date = "2023-01-01"
end_date = "2023-10-01"
data = yf.download(tickers, start=start_date, end=end_date)

asset = data['Adj Close'].CVNA
benchmark = data['Adj Close'].QQQ

asset_returns = asset.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()

asset_returns.plot()
benchmark_returns.plot()
plt.ylabel("Daily Return")
plt.legend()
plt.show()

X = benchmark_returns.values
Y = asset_returns.values

def linreg(x, y):
    # Add a column of 1s to fit alpha
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    # Remove the constant now that we're done
    x = x[:, 1]
    return model.params[0], model.params[1]

alpha, beta = linreg(X, Y)
print(f"Alpha: {alpha}\nBeta: {beta}")

X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2 * beta + alpha
# Plot the raw data
plt.scatter(X, Y, alpha=0.3)
plt.xlabel("QQQ Daily Returns")
plt.ylabel("CVNA Daily Returns")
# Add the regression line
plt.plot(X2, Y_hat, 'r', alpha=0.9)
plt.show()

portfolio = -1 * beta * benchmark_returns + asset_returns
portfolio.name = "CVNA + Hedge"
portfolio.plot(alpha=0.9)
benchmark_returns.plot(alpha=0.5)
asset_returns.plot(alpha=0.5)
plt.ylabel("Daily Return")
plt.legend()
plt.show()

P = portfolio.values
alpha_port, beta_port = linreg(X, P)
print(f"Alpha: {alpha_port}\nBeta: {beta_port}") # resulting beta of (or very near) zero indicates fully hedged