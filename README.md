# Future Sales Prediction

Two short Google Colab notebooks from April 2022: a linear regression that predicts product sales from advertising spend, and an LSTM fitted to Apple daily stock prices.

## Notebooks

### `Future_Sales_Prediction.ipynb`

Predicts `Sales` from advertising spend on `TV`, `Radio` and `Newspaper` using the advertising dataset.

- Checks for missing values, plots each channel against sales with Plotly (OLS trendlines), and computes correlations with sales.
- Trains a scikit-learn `LinearRegression` on an 80/20 train/test split (`random_state=42`).

Results from the saved notebook outputs:

| | Value |
|---|---|
| Correlation with Sales: TV / Radio / Newspaper | 0.90 / 0.35 / 0.16 |
| Test-set R2 (`model.score`) | 0.906 |

### `Stock_Price_Prediction_with_LSTM.ipynb`

Downloads about 5,000 days of AAPL daily prices with `yfinance` (3,446 trading days ending 2022-04-01 in the saved run), plots a candlestick chart, and trains a Keras model with two LSTM layers (128 and 64 units) and two dense layers (117,619 parameters) to predict the `Close` price from the same day's `Open`, `High`, `Low` and `Volume`.

The four features are fed to the LSTM as a length-4 sequence and the rows are split randomly, so this is a same-day regression rather than a forecast of future prices. The features are not scaled. The notebook reports only the training MSE loss (4.11 after 30 epochs, from the saved output) and does not evaluate on the held-out split.

## Running

Both notebooks were written for Google Colab; open them from GitHub in Colab or run them in Jupyter.

- `Future_Sales_Prediction.ipynb` reads the data from `/content/drive/MyDrive/Datasets/FutureSalesPrediction.txt` (a Google Drive path). The data file is not included in this repository; change the path in the data-loading cell to your own copy of the advertising CSV (columns `TV, Radio, Newspaper, Sales`).
- `Stock_Price_Prediction_with_LSTM.ipynb` downloads its data at run time, so results change with the run date.

Dependencies: numpy, pandas, scikit-learn, plotly (and statsmodels for the trendlines); the LSTM notebook also needs yfinance and TensorFlow/Keras.

## Data

- Advertising data: the advertising dataset (TV, radio and newspaper budgets vs. sales) from *An Introduction to Statistical Learning* (James, Witten, Hastie and Tibshirani).
- Stock data: Yahoo Finance via the `yfinance` package.
