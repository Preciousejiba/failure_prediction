# Failure Prediction with Time-Series Analysis

This project uses Flask, TensorFlow, Pandas, and Plotly to create an interactive web application that predicts the failure of medical equipment using time-series analysis. The model is built using an LSTM neural network, trained on historical failure data to predict when equipment is likely to fail.

## Features

- Upload a CSV file containing failure data.
- Visualize equipment failure data interactively using Plotly.
- Train an LSTM model to predict failures based on time-series data.
- Display training loss curve and final model performance after training.
- Download and save the trained model in Keras format (`.keras`).

## Tech Stack

- **Backend**: Flask
- **Machine Learning**: TensorFlow, Keras, LSTM neural networks
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly (for interactive charts), Matplotlib (for training loss curves)

## Installation

### Prerequisites

Before running this project, you will need to have the following installed:

- Python 3.x
- Pip (Python package manager)

### Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/Preciousejiba/failure-prediction.git
    cd failure-prediction
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip3 install -r requirements.txt
    ```

4. Run the Flask app:

    ```bash
    python3 app.py
    ```

The application will now be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

1. Open the application in your browser.
2. Upload a CSV file containing your time-series data (failure data).
3. Set the model hyperparameters:
   - Number of LSTM units.
   - Sequence length.
   - Number of training epochs.
4. Click "Train Model" to visualize the data and train the LSTM model.
5. After training:
   - View the interactive failure data plot.
   - See the training loss curve.
   - View the final model loss after training.

## CSV File Format

The CSV file should have the following columns:

| time        | failure |
|-------------|---------|
| 2024-10-01  | 0       |
| 2024-10-02  | 1       |
| ...         | ...     |

- **time**: The timestamp for each observation.
- **failure**: A binary column representing whether the equipment failed (1 for failure, 0 for no failure).
