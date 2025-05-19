# Stock Price Forecasting with LSTM (Single and Multiple Company Modes)

## Project Overview
This project applies deep learning (LSTM models) to forecast stock prices based on historical market data. It supports:

1. **Single Company Forecasting**: Visualizes and evaluates predictions for a selected stock.
2. **Multi-Company Evaluation**: Trains one model on all stocks and evaluates prediction accuracy across multiple companies.

The goal is to measure the forecasting ability of LSTM models and analyze their accuracy using a custom metric.

---

## Data
- **Preprocessed CSV files** under:
  - `Preprocessed_Data/Large_Cap/`
  - `Preprocessed_Data/Mid_Cap/`
  - `Preprocessed_Data/Small_Cap/`
- Each file must include columns: `Date`, `Open Price`, `High Price`, `Low Price`, `Close Price`.

---

## Methodology

1. **Preprocessing**
   - Parse `Date`, set it as index, sort chronologically.
   - Retain only OHLC values (`Open`, `High`, `Low`, `Close`).
   - Normalize using `MinMaxScaler`.

2. **Sequence Generation**
   - Use a sliding window of 50 timesteps to create training sequences.
   - Predict the next `Close` price from each window.

3. **Model Architecture**
   - 2 stacked LSTM layers with dropout.
   - Fully connected dense layers for regression output.

4. **Training & Evaluation**
   - Trains for 50 epochs (configurable).
   - Option to save and reuse previously trained models.
   - Custom accuracy is calculated as percentage of predictions within ±5% of actual values.

---

## Project Structure

```
Code/ 
    ├── Prediction_Code/  
        ├── forecast_single_company.py # Run prediction on one company (plot included)
        ├── stock_forecast.py # Train on all companies, evaluate & save metrics/plots
    ├── Preprocessed_Data/ 
        ├── Large_Cap/ 
        ├── Mid_Cap/
        ├── Small_Cap/ 
    ├── saved_models/ # Auto-created folder for saved .h5 models 
    ├── plots_50/ # Auto-created folder for forecast plots of 50 epochs model
    ├── plots_100/ # Auto-created folder for forecast plots of 100 epochs model
    ├── plots_500/ # Auto-created folder for forecast plots of 500 epochs model
    ├── evaluation_metrics.csv # Custom accuracy report per stock 
    ├── README.md # This file 
    └── requirements.txt # Python dependencies
```
---

## Requirements
- Python 3.8+
- TensorFlow & Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

---


## Usage

### Single Stock Forecast
To predict and visualize stock price for a single company:

```bash
python Code/forecast_single_company.py
```

Edit the script to choose your stock file from one of the cap folders (Large_Cap, Mid_Cap, or Small_Cap).

### Multi-Stock Forecast and Evaluation
To train on all available stock CSVs and generate evaluation plots:

```bash
python Code/stock_forecast.py
```
1. **Use saved model**
```bash
force_retrain = False
```
- Choose the model based on the index:
    - index 0 for 100 epochs
    - index 1 for 500 epochs
    - index 2 for 50 epochs

Plots saved under plots_50/ or plots_100/ or plots_500/, metrics in evaluation_metrics.csv.

2. **Train New Model**
```bash
force_retrain = True
```
- Choose number of epochs and change learning rate if needed
- Create a directory to store plots based on the number of epochs

```bash
plot_dir = "plots_folder"
```
---

## Results
- For each stock, prediction plots are saved comparing actual vs. forecasted prices.
- The script also reports a "Custom Accuracy" (how close predictions are to actual values within 5% tolerance).
---

## Future Work
- Incorporate technical indicators (RSI, MACD, etc.) as features.
- Experiment with Transformer or GRU-based architectures.
- Add support for real-time prediction using live stock data APIs.

---

## License
This project is released under the MIT License.