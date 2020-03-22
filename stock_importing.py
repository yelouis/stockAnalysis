# Import the plotting library
import matplotlib.pyplot as plt

# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf

# Get the data of the stock AAPL
data = yf.download('AAPL','2016-01-01','2018-01-01')

# Plot the close price of the AAPL
data.Close.plot()
plt.show()
