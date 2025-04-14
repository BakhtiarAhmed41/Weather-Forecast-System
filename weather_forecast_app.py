import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# Initialization
# ---------------------------
st.set_page_config(page_title="Weather Forecast System", layout="wide")
weather_states = ['Sunny', 'Cloudy', 'Rainy', 'Stormy']
severity_map = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3, 'Stormy': 4}
temp_columns = ["Temp_Morning", "Temp_Afternoon", "Temp_Night"]

if 'weather_data' not in st.session_state:
    columns = ['Day', 'Weather', 'Severity'] + temp_columns
    st.session_state.weather_data = pd.DataFrame(columns=columns)

# ---------------------------
# UI Layout
# ---------------------------
st.title("🌦️ Weather Forecast System")

tab1, tab2, tab3, tab4 = st.tabs(["Graphs", "Report", "TPM", "Markov Chain"])

# ---------------------------
# Helper Functions
# ---------------------------
def add_weather_entry(day, weather, temps):
    severity = severity_map.get(weather, 1)
    new_data = {
        "Day": day,
        "Weather": weather,
        "Severity": severity,
        "Temp_Morning": temps.get("Temp_Morning"),
        "Temp_Afternoon": temps.get("Temp_Afternoon"),
        "Temp_Night": temps.get("Temp_Night")
    }
    st.session_state.weather_data = pd.concat([
        st.session_state.weather_data,
        pd.DataFrame([new_data])
    ], ignore_index=True)

def ensure_columns(df):
    for col in ['Severity'] + temp_columns:
        if col not in df.columns:
            if col == "Severity":
                df['Severity'] = df['Weather'].map(severity_map)
            else:
                df[col] = np.nan
    return df

# ---------------------------
# Sidebar Upload/Download
# ---------------------------
st.sidebar.header("📂 Upload / Download Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = ensure_columns(df)
        st.session_state.weather_data = df
        st.sidebar.success("Data uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")

if not st.session_state.weather_data.empty:
    csv = st.session_state.weather_data.to_csv(index=False)
    st.sidebar.download_button("Download Data", csv, "weather_data.csv", "text/csv")

# ---------------------------
# Tab 1: Graphs
# ---------------------------
with tab1:
    st.subheader("📊 Weather Entry and Visualizations")

    with st.form("add_weather_form"):
        cols = st.columns(4)
        day = cols[0].number_input("Day", min_value=1, value=1)
        weather = cols[1].selectbox("Weather Type", weather_states)
        temp_m = cols[2].number_input("Morning Temp", value=25.0)
        temp_a = cols[3].number_input("Afternoon Temp", value=30.0)
        temp_n = st.number_input("Night Temp", value=20.0)
        submitted = st.form_submit_button("➕ Add Weather Entry")
        if submitted:
            add_weather_entry(day, weather, {
                "Temp_Morning": temp_m,
                "Temp_Afternoon": temp_a,
                "Temp_Night": temp_n
            })
            st.success(f"✅ Day {day} - {weather} added successfully.")

    data = ensure_columns(st.session_state.weather_data)

    if not data.empty:
        st.markdown("### 📌 Weather Frequency")
        counts = data['Weather'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.bar(counts.index, counts.values, color='skyblue')
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        st.markdown("### 🥧 Weather Distribution Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax2.axis('equal')
        st.pyplot(fig2)

        st.markdown("### 📈 Weather Severity Over Time")
        sorted_df = data.sort_values("Day")
        fig3, ax3 = plt.subplots()
        ax3.plot(sorted_df["Day"], sorted_df["Severity"], marker='o')
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Severity")
        st.pyplot(fig3)

        st.markdown("### 🔄 Custom Graph: Select X and Y")
        columns = data.columns.tolist()
        x_col = st.selectbox("X-Axis", columns, index=0, key="x_axis_graph")
        y_col = st.selectbox("Y-Axis", columns, index=columns.index("Temp_Afternoon"), key="y_axis_graph")

        fig4, ax4 = plt.subplots()
        ax4.plot(data[x_col], data[y_col], marker='o')
        ax4.set_xlabel(x_col)
        ax4.set_ylabel(y_col)
        ax4.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig4)

# ---------------------------
# Tab 2: Report
# ---------------------------
with tab2:
    st.subheader("📋 Weather Report")
    if not data.empty:
        st.markdown(f"""
        - **Total Days:** {len(data)}
        - **Unique Weather Types:** {data['Weather'].nunique()}
        - **Most Frequent Weather:** {data['Weather'].mode()[0]}
        """)
        st.dataframe(data)
    else:
        st.info("No data to show yet.")

# ---------------------------
# Tab 3: Transition Probability Matrix
# ---------------------------
with tab3:
    st.subheader("🔄 Transition Probability Matrix")
    if len(data) > 1:
        tpm = pd.crosstab(data['Weather'].shift(), data['Weather'], normalize=0).reindex(
            index=weather_states, columns=weather_states, fill_value=0).round(2)
        st.dataframe(tpm)
    else:
        st.warning("Add more data to calculate TPM.")

# ---------------------------
# Tab 4: Markov Chain Forecast
# ---------------------------
with tab4:
    st.subheader("🔁 Weather Forecast using Markov Chain")
    if len(data) > 1:
        tpm = pd.crosstab(data['Weather'].shift(), data['Weather'], normalize=0).reindex(
            index=weather_states, columns=weather_states, fill_value=0).round(2)

        current_state = st.selectbox("Select Current State", weather_states)
        steps = st.slider("Steps (Days Ahead)", 1, 10, 3)

        forecast = [current_state]
        for _ in range(steps):
            probs = tpm.loc[forecast[-1]]
            if probs.sum() == 0:
                forecast.append(np.random.choice(weather_states))
            else:
                forecast.append(np.random.choice(probs.index, p=probs.values))

        st.markdown("### Forecast Path")
        st.write(" → ".join(forecast))

        st.markdown("### 📉 Graphical Markov Chain")
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
        st.pyplot(fig)

    else:
        st.info("Add more entries to use Markov simulation.")
