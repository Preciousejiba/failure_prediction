from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    data = pd.read_csv(file)

    # Visualization with Plotly
    fig = px.line(data, x='time', y='failure', title='Equipment Failure Data')
    graph_html = fig.to_html(full_html=False)

    # Get user-selected hyperparameters from the form
    lstm_units = int(request.form.get('lstm_units', 50))
    sequence_length = int(request.form.get('sequence_length', 10))
    epochs = int(request.form.get('epochs', 5))

    # Dynamically adjust sequence length if necessary
    sequence_length = min(sequence_length, len(data) - 1)

    # Check if there are enough data points for sequence creation
    if sequence_length < 1:
        return f"Error: The dataset is too small to create valid sequences. Please upload more data."

    # Preparing data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['failure']])

    # Creating sequences of data for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    # Convert lists to numpy arrays
    X, y = np.array(X), np.array(y)

    # Ensure X has three dimensions (samples, time steps, features)
    if len(X.shape) == 2:
        X = np.expand_dims(X, -1)

    # Check if X is empty or not properly shaped
    if X.size == 0:
        return "Error: No valid sequences to train the model."

    # Define the LSTM model with Input layer
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(lstm_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=epochs, verbose=1)

    # Save the model in the new Keras format
    model.save('saved_model/failure_model.keras')

    # Plot loss curve
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    
    # Convert plot to PNG image and encode it to display on the page
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Get model performance metrics
    final_loss = history.history['loss'][-1]

    return render_template('result.html', graph_html=graph_html, plot_url=plot_url, final_loss=final_loss)


if __name__ == '__main__':
    app.run(debug=True)


