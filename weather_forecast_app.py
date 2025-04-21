# Filename: weather_forecast_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Weather Forecast System", layout="wide")

# Constants
WEATHER_STATES = ['Sunny', 'Cloudy', 'Rainy', 'Stormy']
SEVERITY_MAP = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3, 'Stormy': 4}
TEMP_COLUMNS = ["Morning_Temp", "Afternoon_Temp", "Evening_Temp"]
REQUIRED_COLUMNS = ['Day', 'Weather'] + TEMP_COLUMNS

# Initialize session state
for key, default in {
    'weather_data': pd.DataFrame(columns=REQUIRED_COLUMNS + ['Severity']),
    'simulation_data': pd.DataFrame(columns=REQUIRED_COLUMNS + ['Severity']),
    'live_simulation': False,
    'simulation_log': [],
    'simulation_speed': 1.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Helper Functions
def ensure_columns(df):
    df = df.copy()
    for col in ['Severity'] + TEMP_COLUMNS:
        if col not in df.columns:
            if col == 'Severity':
                df['Severity'] = df['Weather'].map(SEVERITY_MAP)
            else:
                df[col] = np.nan
    return df

def validate_uploaded_data(df):
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        st.error(f"Uploaded CSV missing required columns: {', '.join(missing)}")
        return False
    if df['Weather'].isin(WEATHER_STATES).sum() != len(df):
        st.error("CSV contains invalid weather states.")
        return False
    if df['Day'].duplicated().any():
        st.error("CSV contains duplicate day entries.")
        return False
    return True

def compute_tpm(data):
    if len(data) < 2:
        return pd.DataFrame()
    tpm = pd.crosstab(data['Weather'].shift(), data['Weather'], normalize=0).reindex(
        index=WEATHER_STATES, columns=WEATHER_STATES, fill_value=0).round(2)
    return tpm

def add_weather_entry(df_key, day, weather, temps):
    """Add entry to specified dataframe in session_state (weather_data or simulation_data)."""
    if day in st.session_state[df_key]['Day'].values:
        return False
    severity = SEVERITY_MAP.get(weather, 1)
    new_data = {
        "Day": day,
        "Weather": weather,
        "Severity": severity,
        "Morning_Temp": round(temps["Morning_Temp"]),
        "Afternoon_Temp": round(temps["Afternoon_Temp"]),
        "Evening_Temp": round(temps["Evening_Temp"])
    }
    st.session_state[df_key] = pd.concat([st.session_state[df_key], pd.DataFrame([new_data])], ignore_index=True)
    return True

# Plotting Functions
def plot_weather_frequency(data):
    fig, ax = plt.subplots()
    data['Weather'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel("Count")
    ax.set_title("Weather Frequency")
    return fig

def plot_weather_distribution(data):
    fig, ax = plt.subplots()
    counts = data['Weather'].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax.axis('equal')
    return fig

def plot_severity_over_time(data):
    fig, ax = plt.subplots()
    sorted_df = data.sort_values("Day")
    ax.plot(sorted_df["Day"], sorted_df["Severity"], marker='o', color='teal')
    ax.set_xlabel("Day")
    ax.set_ylabel("Severity")
    ax.set_title("Weather Severity Over Time")
    ax.grid(True)
    return fig

def plot_temperature_trends(data):
    fig, ax = plt.subplots()
    sorted_df = data.sort_values("Day")
    for col in TEMP_COLUMNS:
        ax.plot(sorted_df["Day"], sorted_df[col], marker='o', label=col)
    ax.set_title("Temperature Trends")
    ax.set_xlabel("Day")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    return fig

def plot_markov_chain(tpm):
    G = nx.DiGraph()
    for i, row in tpm.iterrows():
        for j, prob in row.items():
            if prob > 0:
                G.add_edge(i, j, weight=prob)
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    return fig

# Sidebar Upload/Download
st.sidebar.header("📂 Data Management")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if validate_uploaded_data(df):
            st.session_state.weather_data = ensure_columns(df)
            st.sidebar.success("Upload successful!")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

if not st.session_state.weather_data.empty:
    st.sidebar.download_button("Download Data", data=st.session_state.weather_data.to_csv(index=False),
                               file_name="weather_data.csv", mime="text/csv")

# Title and Tabs
st.title("🌦️ Weather Forecast System")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Graphs", "Report", "TPM", "Markov Chain", "Live Simulation"])

data = ensure_columns(st.session_state.weather_data)

# ========== Tab 1: Graphs ==========  
with tab1:
    st.subheader("📊 Add New Entry & Visualizations")

    # --- Add Form ---
    with st.form("add_form"):
        cols = st.columns(5)
        day = cols[0].number_input("Day", min_value=1, value=int(st.session_state.weather_data["Day"].max() + 1) if not st.session_state.weather_data.empty else 1)
        weather = cols[1].selectbox("Weather", WEATHER_STATES)
        t_m = cols[2].number_input("Morning Temp (°C)", 0, 50, 25)
        t_a = cols[3].number_input("Afternoon Temp (°C)", 0, 50, 30)
        t_e = cols[4].number_input("Evening Temp (°C)", 0, 50, 20)
        submitted = st.form_submit_button("➕ Add Entry")

        if submitted:
            if add_weather_entry('weather_data', day, weather, {
                "Morning_Temp": t_m,
                "Afternoon_Temp": t_a,
                "Evening_Temp": t_e
            }):
                st.success(f"Entry for Day {day} added!")
                st.rerun()

    # --- If Data Exists ---
    if not st.session_state.weather_data.empty:
        # Ensure numeric columns are cast correctly
        st.session_state.weather_data["Day"] = pd.to_numeric(st.session_state.weather_data["Day"], errors='coerce')
        st.session_state.weather_data["Morning_Temp"] = pd.to_numeric(st.session_state.weather_data["Morning_Temp"], errors='coerce')
        st.session_state.weather_data["Afternoon_Temp"] = pd.to_numeric(st.session_state.weather_data["Afternoon_Temp"], errors='coerce')
        st.session_state.weather_data["Evening_Temp"] = pd.to_numeric(st.session_state.weather_data["Evening_Temp"], errors='coerce')

        # Show standard graphs (one per line)
        st.pyplot(plot_weather_frequency(st.session_state.weather_data), clear_figure=True)
        st.pyplot(plot_severity_over_time(st.session_state.weather_data), clear_figure=True)
        st.pyplot(plot_weather_distribution(st.session_state.weather_data), clear_figure=True)
        st.pyplot(plot_temperature_trends(st.session_state.weather_data), clear_figure=True)

        # --- Custom Graph Section ---
        st.markdown("### 🔧 Custom Graph")
        numeric = [col for col in st.session_state.weather_data.columns if pd.api.types.is_numeric_dtype(st.session_state.weather_data[col])]

        # Check if we have enough numeric columns
        if len(numeric) >= 2:
            x = st.selectbox("X-Axis", numeric, index=0)
            y = st.selectbox("Y-Axis", numeric, index=1)
            fig, ax = plt.subplots()
            ax.plot(st.session_state.weather_data[x], st.session_state.weather_data[y], marker='o', color='darkgreen')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")
            ax.grid(True)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Not enough numeric columns for custom graph.")

# ========== Tab 2: Report ==========
with tab2:
    st.subheader("📋 Report Summary")
    if not data.empty:
        st.markdown(f"""
        - **Total Days:** {len(data)}
        - **Weather Types:** {data['Weather'].nunique()}
        - **Most Common Weather:** {data['Weather'].mode()[0]}
        - **Average Temps:** 
            - Morning: {data['Morning_Temp'].mean():.1f}°C  
            - Afternoon: {data['Afternoon_Temp'].mean():.1f}°C  
            - Evening: {data['Evening_Temp'].mean():.1f}°C  
        """)
        st.dataframe(data, use_container_width=True)

# ========== Tab 3: TPM ==========
with tab3:
    st.subheader("🔄 Transition Probability Matrix")
    tpm = compute_tpm(data)
    if not tpm.empty:
        st.dataframe(tpm, use_container_width=True)
    else:
        st.info("At least two entries are required for TPM.")

# ========== Tab 4: Markov Chain ==========
with tab4:
    st.subheader("🔁 Weather Forecast using Markov Chain")
    if len(data) >= 2:
        current = st.selectbox("Current Weather", WEATHER_STATES)
        steps = st.slider("Forecast Steps", 1, 10, 3)
        forecast = [current]
        for _ in range(steps):
            probs = tpm.loc[forecast[-1]]
            forecast.append(np.random.choice(probs.index, p=probs / probs.sum()) if probs.sum() > 0 else random.choice(WEATHER_STATES))
        st.markdown(" → ".join(forecast))
        fig = plot_markov_chain(tpm)
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Markov Graph", buf.getvalue(), file_name="markov_chain.png", mime="image/png")
    else:
        st.info("Add at least two weather entries for Markov forecasting.")

# ========== Tab 5: Live Simulation ==========
with tab5:
    st.subheader("⏱️ Live Simulation")

    # Controls
    col1, col2, col3 = st.columns(3)
    min_t = col1.number_input("Min Temp (°C)", 0, 30, 15)
    max_t = col1.number_input("Max Temp (°C)", min_t + 5, 50, 40)
    st.session_state.simulation_speed = col2.slider("Speed (seconds)", 0.1, 5.0, 1.0)
    auto = col2.checkbox("Auto-Update Graph", value=True)
    max_entries = col3.number_input("Max Entries", 10, 1000, 100)

    # Buttons
    c1, c2 = st.columns(2)
    if not st.session_state.live_simulation:
        if c1.button("▶️ Start Simulation"):
            st.session_state.live_simulation = True
            st.session_state.simulation_log.clear()
            st.session_state.simulation_data = pd.DataFrame(columns=REQUIRED_COLUMNS + ['Severity'])
            st.rerun()
    else:
        if c1.button("⏹️ Stop Simulation"):
            st.session_state.live_simulation = False
            st.success("Simulation stopped.")
            st.rerun()

    if c2.button("🔁 Reset Data"):
        st.session_state.simulation_data = pd.DataFrame(columns=REQUIRED_COLUMNS + ['Severity'])
        st.session_state.simulation_log.clear()
        st.success("Simulation data reset.")
        st.rerun()

    sim_data = st.session_state.simulation_data
    graph_container = st.empty()
    table_container = st.empty()
    log_container = st.empty()
    tpm_container = st.empty()
    mc_container = st.empty()

    if st.session_state.live_simulation:
        progress = st.progress(0)
        last_day = int(sim_data["Day"].max()) if not sim_data.empty else 0

        for i in range(max_entries):
            if not st.session_state.live_simulation:
                break

            new_day = last_day + i + 1
            weather = random.choice(WEATHER_STATES)
            temps = {
                "Morning_Temp": random.randint(min_t, max_t),
                "Afternoon_Temp": random.randint(min_t, max_t),
                "Evening_Temp": random.randint(min_t, max_t)
            }

            if add_weather_entry('simulation_data', new_day, weather, temps):
                st.session_state.simulation_log.append(f"Day {new_day}: {weather}, Temps={temps}")

            if auto:
                graph_container.pyplot(plot_severity_over_time(st.session_state.simulation_data), clear_figure=True)

            table_container.dataframe(st.session_state.simulation_data.tail(10), use_container_width=True)
            log_container.write("\n".join(st.session_state.simulation_log[-5:]))

            progress.progress((i + 1) / max_entries)
            time.sleep(st.session_state.simulation_speed)
            st.rerun()

    # Post-simulation outputs
    if not st.session_state.live_simulation and not st.session_state.simulation_data.empty:
        graph_container.pyplot(plot_severity_over_time(st.session_state.simulation_data), clear_figure=True)
        table_container.dataframe(st.session_state.simulation_data.tail(10), use_container_width=True)
        log_container.write("\n".join(st.session_state.simulation_log[-5:]))

        tpm_sim = compute_tpm(st.session_state.simulation_data)
        tpm_container.dataframe(tpm_sim)
        mc_container.pyplot(plot_markov_chain(tpm_sim), clear_figure=True)
