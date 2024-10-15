import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Data validation and preprocessing
def validate_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Validate data: check required columns
    required_columns = ['performance', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill missing values
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    
    # Remove outliers (IQR method)
    Q1 = df['performance'].quantile(0.25)
    Q3 = df['performance'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['performance'] < (Q1 - 1.5 * IQR)) | (df['performance'] > (Q3 + 1.5 * IQR)))]

    # Create rolling average
    df['rolling_avg'] = df['performance'].rolling(window=3).mean()

    # Ensure time column is properly parsed
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Normalize data
    scaler = MinMaxScaler()
    df[['performance', 'rolling_avg']] = scaler.fit_transform(df[['performance', 'rolling_avg']])

    return df, scaler

# Train the LSTM model
def train_model(filepath):
    df, scaler = validate_and_preprocess(filepath)

    X, y = [], []
    time_steps = 10  # Reduced time steps from 50 to 10 to support smaller datasets

    # Prepare data for LSTM
    for i in range(time_steps, len(df)):
        X.append(df[['performance', 'rolling_avg']].iloc[i-time_steps:i].values)
        y.append(df['performance'].iloc[i])

    X, y = np.array(X), np.array(y)

    print("Shape of X (samples, time_steps, features):", X.shape)
    print("First 5 rows of the input data for LSTM:", df[['performance', 'rolling_avg']].head())

    # Error handling for insufficient data
    if X.shape[0] == 0:
        raise ValueError("Not enough data to create the required time steps for LSTM. Try lowering the time_steps value or add more data to your dataset.")

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)

    return model, scaler

# Predict future performance and visualize the result
def predict_and_visualize(model, scaler, df):
    last_10_steps = df[['performance', 'rolling_avg']].iloc[-10:].values.reshape(1, 10, 2)

    predictions = []
    for _ in range(10):
        pred = model.predict(last_10_steps)
        predictions.append(pred[0][0])

        # Reshape prediction to match the 3D shape for concatenation
        pred_reshaped = np.array([[pred[0][0], pred[0][0]]]).reshape(1, 1, 2)
        
        # Concatenate the prediction to the existing sequence
        last_10_steps = np.append(last_10_steps[:, 1:, :], pred_reshaped, axis=1)

    # Ensure predictions are inverse-transformed correctly
    predictions = np.array(predictions).reshape(-1, 1)  # Reshape to match expected scaler input
    predictions = scaler.inverse_transform(np.hstack([predictions, predictions]))[:, 0]

    print("Raw Predictions from the Model:", predictions)  # Debugging print statement

    # Visualization using Plotly
    trace_avg = go.Scatter(
        x=df.index,
        y=df['performance'],
        mode='lines',
        name='Historical Performance'
    )
    future_dates = pd.date_range(start=df.index[-1], periods=len(predictions) + 1, freq='D')[1:]
    trace_pred = go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Performance'
    )

    data = [trace_avg, trace_pred]
    layout = go.Layout(
        title="Failure Prediction with Time-Series Analysis",
        xaxis_title="Time",
        yaxis_title="Performance"
    )

    return predictions, data, layout

# Generate a PDF report of the predictions
def generate_pdf_report(predictions, filename='static/prediction_report.pdf'):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Failure Prediction Report")
    c.drawString(100, 730, f"Total Predictions: {len(predictions)}")

    for i, pred in enumerate(predictions, 1):
        c.drawString(100, 710 - i * 20, f"Prediction {i}: {pred:.2f}")

    c.showPage()
    c.save()


