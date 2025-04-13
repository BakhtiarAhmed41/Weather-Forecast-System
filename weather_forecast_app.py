import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO

# Weather states and severity
weather_states = ['Sunny', 'Cloudy', 'Rainy', 'Stormy']
severity_map = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3, 'Stormy': 4}

# Initialize session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame(columns=[
        'Day', 'Weather', 'Severity',
        'Morning_Temp', 'Afternoon_Temp', 'Evening_Temp'
    ])

st.title("🌦️ Weather Forecast System")

tab1, tab2, tab3, tab4 = st.tabs(["Graphs", "Report", "TPM", "Markov Chain"])

# Function to add weather entry
def add_weather_entry(day, weather, morning_temp, afternoon_temp, evening_temp):
    severity = severity_map.get(weather, 1)
    entry = {
        "Day": day,
        "Weather": weather,
        "Severity": severity,
        "Morning_Temp": morning_temp,
        "Afternoon_Temp": afternoon_temp,
        "Evening_Temp": evening_temp
    }
    st.session_state.weather_data = pd.concat(
        [st.session_state.weather_data, pd.DataFrame([entry])],
        ignore_index=True
    )

# Upload / Download
st.sidebar.header("📂 Upload / Download Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    st.session_state.weather_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")

if not st.session_state.weather_data.empty:
    csv = st.session_state.weather_data.to_csv(index=False)
    st.sidebar.download_button("Download Data", csv, "weather_data.csv", "text/csv")

# Tab 1: Graphs
with tab1:
    st.subheader("📊 Weather Visualizations")

    with st.form("weather_form_graph_tab"):
        day = st.number_input("Day", min_value=1)
        weather = st.selectbox("Weather Type", weather_states)
        morning_temp = st.number_input("🌅 Morning Temp (°C)", step=0.1)
        afternoon_temp = st.number_input("🌞 Afternoon Temp (°C)", step=0.1)
        evening_temp = st.number_input("🌇 Evening Temp (°C)", step=0.1)
        submitted = st.form_submit_button("Add Entry")
        if submitted:
            add_weather_entry(day, weather, morning_temp, afternoon_temp, evening_temp)
            st.success(f"Day {day} - {weather} with temperatures added.")

    if not st.session_state.weather_data.empty:
        df = st.session_state.weather_data.sort_values("Day")

        # Weather Frequency Bar Plot
        st.markdown("### Weather Frequency")
        counts = df['Weather'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color='skyblue')
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Pie Chart
        st.markdown("### Weather Distribution")
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax2.axis('equal')
        st.pyplot(fig2)

        # Severity Over Time
        st.markdown("### Weather Severity Over Time")
        fig3, ax3 = plt.subplots()
        ax3.plot(df['Day'], df['Severity'], marker='o')
        ax3.set_ylabel("Severity")
        ax3.set_xlabel("Day")
        ax3.set_title("Time Series of Weather Severity")
        st.pyplot(fig3)

        # Temperature Trends
        st.markdown("### 🌡️ Temperature Trends Over Days")
        fig_temp, ax_temp = plt.subplots()
        ax_temp.plot(df["Day"], df["Morning_Temp"], label="Morning", marker='o')
        ax_temp.plot(df["Day"], df["Afternoon_Temp"], label="Afternoon", marker='s')
        ax_temp.plot(df["Day"], df["Evening_Temp"], label="Evening", marker='^')
        ax_temp.set_xlabel("Day")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.legend()
        ax_temp.set_title("Daily Temperature Patterns")
        st.pyplot(fig_temp)

        # Custom Graph
        st.markdown("### Custom Graph: Select X and Y Axes")
        columns = df.columns.tolist()
        x_axis = st.selectbox("X-Axis", columns, index=0)
        y_axis = st.selectbox("Y-Axis", columns, index=1)
        if x_axis and y_axis:
            fig4, ax4 = plt.subplots()
            ax4.plot(df[x_axis], df[y_axis], marker='o')
            ax4.set_xlabel(x_axis)
            ax4.set_ylabel(y_axis)
            ax4.set_title(f"{y_axis} vs {x_axis}")
            st.pyplot(fig4)

# Tab 2: Report
with tab2:
    st.subheader("📋 Data Report")
    df = st.session_state.weather_data
    if not df.empty:
        st.markdown(f"""
        - **Total Days:** {len(df)}
        - **Unique Weather Types:** {df['Weather'].nunique()}
        - **Most Common:** {df['Weather'].mode()[0]}
        """)
        st.dataframe(df)
    else:
        st.warning("No data to show yet.")

# Tab 3: TPM
with tab3:
    st.subheader("🔄 Transition Probability Matrix")
    df = st.session_state.weather_data
    if len(df) > 1:
        tpm = pd.crosstab(df['Weather'].shift(), df['Weather'], normalize=0).reindex(
            index=weather_states, columns=weather_states, fill_value=0).round(2)
        st.dataframe(tpm)
    else:
        st.warning("Please add more data to calculate TPM.")

# Tab 4: Markov Chain
with tab4:
    st.subheader("🔁 Weather Forecast using Markov Chain")
    df = st.session_state.weather_data
    if len(df) > 1:
        tpm = pd.crosstab(df['Weather'].shift(), df['Weather'], normalize=0).reindex(
            index=weather_states, columns=weather_states, fill_value=0).round(2)

        current_state = st.selectbox("Current State", weather_states)
        steps = st.slider("Days Ahead", 1, 10, 3)

        forecast = [current_state]
        for _ in range(steps):
            probs = tpm.loc[forecast[-1]]
            next_state = np.random.choice(probs.index, p=probs.values) if probs.sum() > 0 else np.random.choice(weather_states)
            forecast.append(next_state)

        st.markdown("### Forecast Path")
        st.write(" → ".join(forecast))

        # Visual Graph
        st.markdown("### 📈 Markov Chain Visualization")
        G = nx.DiGraph()
        for from_state in tpm.index:
            for to_state in tpm.columns:
                prob = tpm.loc[from_state, to_state]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=prob)

        pos = nx.spring_layout(G, seed=42)
        fig_mc, ax_mc = plt.subplots()
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw(G, pos, ax=ax_mc, with_labels=True, node_color='lightblue', node_size=2000,
                edge_color='gray', width=edge_weights, arrowsize=20, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_mc)
        st.pyplot(fig_mc)

        # Average temperature per forecast state
        st.markdown("### 🌤️ Avg Temperature Forecast (Based on Weather)")
        if {'Morning_Temp', 'Afternoon_Temp', 'Evening_Temp'}.issubset(df.columns):
            morning_avg = df.groupby("Weather")["Morning_Temp"].mean().round(1)
            afternoon_avg = df.groupby("Weather")["Afternoon_Temp"].mean().round(1)
            evening_avg = df.groupby("Weather")["Evening_Temp"].mean().round(1)

            forecast_temp_df = pd.DataFrame({
                "Morning (°C)": morning_avg,
                "Afternoon (°C)": afternoon_avg,
                "Evening (°C)": evening_avg
            })

            st.dataframe(forecast_temp_df.loc[forecast])
    else:
        st.info("More data needed to run Markov simulation.")
