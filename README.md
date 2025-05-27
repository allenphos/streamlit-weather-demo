# Streamlit Aussie Rain Predictor

This project demonstrates how to deploy a machine learning model using Streamlit to predict whether it will rain tomorrow in Australia, based on weather data.
The app allows users to enter weather parameters and get a prediction and probability using a pre-trained ML model.

You can test the application here: [https://your-app-link.streamlit.app/](https://allenphos-streamlit-weather-demo-app-n0p56x.streamlit.app/)
If you see the message "This app has gone to sleep due to inactivity. Would you like to wake it back up?", simply click "Yes, get this app back up!" and wait about 30 seconds.

## Project Structure

- **data/**: Директорія, що містить набір даних Iris (`weatherAUS.csv`).
- **images/**: Директорія для зберігання зображень, які використовуються в додатку.
- **models/**: Директорія, що містить навчену ML-модель.
- **app.py**: Основний файл застосунку Streamlit.
- **requirements.txt**: Список необхідних Python-пакетів.

## Setup

### Prerequisites

- Python 3.12 or newer

- Required packages listed in requirements.txt `requirements.txt`.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/allenphos/streamlit-weather-demo.git
   cd streamlit-weather-demo
   ```

2. **(Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

Run the Streamlit application locally with the following command:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.
