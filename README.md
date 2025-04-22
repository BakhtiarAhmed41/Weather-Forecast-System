
# 🌦️ Weather Forecast System

An interactive weather simulation and prediction tool built with **Streamlit** and powered by **Markov Chains**. This application allows users to enter historical or synthetic weather data, visualize patterns, compute transition probabilities, and forecast future weather using stochastic modeling.

---

## 📌 Features

- **Data Entry & Visualization**
  - Add daily weather conditions and temperature readings
  - View bar charts, pie charts, severity trends, and custom plots

- **Weather Report Tab**
  - Summarizes dataset statistics: total days, unique weather types, most frequent weather
  - Displays complete tabular data for review and export

- **Transition Probability Matrix (TPM)**
  - Calculates probabilities of moving from one weather state to another
  - Visualizes how the weather shifts between: Sunny, Cloudy, Rainy, Stormy

- **Markov Chain Forecasting**
  - Simulates future weather paths based on current state and TPM
  - Graphical Markov Chain visualization using NetworkX

- **Data Upload & Export**
  - Upload existing CSV files with weather data
  - Export current dataset with all fields as a downloadable CSV

---

## 📊 Weather States

| Weather | Severity |
|---------|----------|
| Sunny   | 1        |
| Cloudy  | 2        |
| Rainy   | 3        |
| Stormy  | 4        |

Severity scores are used to track and visualize the intensity of weather patterns over time.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed.

### Installation

```bash
git clone https://github.com/your-username/weather-forecast-system.git
cd weather-forecast-system
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run weather_forecast_app.py
```

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) – for UI and interaction
- [Pandas](https://pandas.pydata.org/) – data manipulation
- [NumPy](https://numpy.org/) – numerical operations
- [Matplotlib](https://matplotlib.org/) – plotting graphs
- [NetworkX](https://networkx.org/) – Markov Chain graph visualization

---

## 📁 Example CSV Format

```csv
Day,Weather,Temp_Morning,Temp_Afternoon,Temp_Night
1,Sunny,25,30,20
2,Cloudy,24,28,21
3,Rainy,22,26,19
...
```

---

## 📈 Future Enhancements

- Real-time weather API integration
- Additional weather states (e.g., Snowy, Foggy, Windy)
- Machine learning-based weather predictions
- Export PDF reports with charts and summaries
- Seasonal weather analysis features
