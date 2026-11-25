import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="JalTrace AI", page_icon="💧", layout="wide")
st.title("💧 JalTrace AI")
st.caption("AI-Powered Community Water Distribution Anomaly & Shortage Intelligence")

# --------------------------------------------------
# 2. GENERATE SIMULATED DATA
# --------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    localities = [
        "Green Park", "Lake View", "Hill Top", "Old Town", "East Colony",
        "West Colony", "River Side", "Sunrise Nagar", "Market Area", "Central Ward"
    ]
    rows = []
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30)
    for date in dates:
        for locality in localities:
            planned = np.random.randint(8000, 15000)
            delivered = planned * np.random.uniform(0.88, 1.05)
            delay = np.random.randint(0, 90)
            if locality in ["Hill Top", "Old Town", "East Colony"] and np.random.random() < 0.30:
                delivered = planned * np.random.uniform(0.40, 0.72)
                delay = np.random.randint(90, 240)
            rows.append({
                "Date": date,
                "Locality": locality,
                "Planned_Liters": int(planned),
                "Delivered_Liters": int(delivered),
                "Delay_Minutes": int(delay)
            })
    return pd.DataFrame(rows)

# --------------------------------------------------
# 3. LOAD DATA / OPTIONAL CSV UPLOAD
# --------------------------------------------------
default_data = generate_data()
st.sidebar.header("⚙️ Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload Water Supply CSV", type=["csv"])
required_columns = ["Date", "Locality", "Planned_Liters", "Delivered_Liters", "Delay_Minutes"]

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)
        missing_columns = [col for col in required_columns if col not in upload_df.columns]
        if not missing_columns:
            df = upload_df[required_columns].copy()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for col in ["Planned_Liters", "Delivered_Liters", "Delay_Minutes"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=required_columns)
            if df.empty:
                st.sidebar.error("CSV has no valid rows. Using demo data.")
                df = default_data.copy()
            else:
                st.sidebar.success("CSV uploaded successfully!")
        else:
            st.sidebar.error("CSV columns are missing.")
            st.sidebar.write("Missing columns:", missing_columns)
            df = default_data.copy()
    except Exception as error:
        st.sidebar.error(f"Could not read CSV: {error}")
        df = default_data.copy()
else:
    df = default_data.copy()
    st.sidebar.info("Using simulated demo data")

# --------------------------------------------------
# 4. DATA PROCESSING
# --------------------------------------------------
df["Shortage_Liters"] = (df["Planned_Liters"] - df["Delivered_Liters"]).clip(lower=0)
df["Fulfillment_Percent"] = (df["Delivered_Liters"] / df["Planned_Liters"].replace(0, np.nan) * 100).fillna(0).clip(upper=100)
df["Status"] = np.where(df["Fulfillment_Percent"] >= 90, "Normal", np.where(df["Fulfillment_Percent"] >= 70, "Warning", "Critical"))

# --------------------------------------------------
# 5. MACHINE LEARNING ANOMALY DETECTION
# --------------------------------------------------
features = df[["Planned_Liters", "Delivered_Liters", "Delay_Minutes", "Shortage_Liters", "Fulfillment_Percent"]].fillna(0)
model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
df["Anomaly"] = model.fit_predict(features)
df["Anomaly_Status"] = np.where(df["Anomaly"] == -1, "Anomaly Detected", "Normal Pattern")

# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "Overview Dashboard",
    "Water Supply Records",
    "Shortage Analysis",
    "AI Anomaly Detection",
    "Shortage Hotspot Map",
    "AI Shortage Insights",
    "Historical Trend Analysis",
    "Community Water Reports",
    "Smart Priority Recommendations",
    "About JalTrace AI"
])


# --------------------------------------------------
# 7. LOCALITY FILTER
# --------------------------------------------------
localities = sorted(df["Locality"].dropna().unique())
selected_localities = st.sidebar.multiselect("Select Localities", localities, default=localities)
filtered_df = df[df["Locality"].isin(selected_localities)].copy()
if filtered_df.empty:
    st.warning("Select at least one locality from the sidebar.")
    st.stop()

# --------------------------------------------------
# 8. OVERVIEW DASHBOARD
# --------------------------------------------------
if page == "Overview Dashboard":
    st.subheader("📊 Community Water Supply Overview")
    total_planned = filtered_df["Planned_Liters"].sum()
    total_delivered = filtered_df["Delivered_Liters"].sum()
    total_shortage = filtered_df["Shortage_Liters"].sum()
    fulfillment = total_delivered / total_planned * 100 if total_planned > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Planned Water", f"{total_planned:,.0f} L")
    col2.metric("Delivered Water", f"{total_delivered:,.0f} L")
    col3.metric("Recorded Shortage", f"{total_shortage:,.0f} L")
    col4.metric("Fulfillment Rate", f"{fulfillment:.1f}%")
    st.divider()
    daily = filtered_df.groupby("Date")[["Planned_Liters", "Delivered_Liters"]].sum().reset_index()
    fig1 = px.line(daily, x="Date", y=["Planned_Liters", "Delivered_Liters"], markers=True, title="Daily Planned vs Delivered Water")
    st.plotly_chart(fig1, use_container_width=True)
    locality_summary = filtered_df.groupby("Locality")[["Planned_Liters", "Delivered_Liters", "Shortage_Liters"]].sum().reset_index()
    fig2 = px.bar(locality_summary, x="Locality", y=["Planned_Liters", "Delivered_Liters"], barmode="group", title="Water Supply Comparison by Locality")
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("🚨 Shortage Hotspots")
    ranking = locality_summary.sort_values("Shortage_Liters", ascending=False)
    fig3 = px.bar(ranking, x="Locality", y="Shortage_Liters", color="Shortage_Liters", title="Locality Shortage Ranking")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(ranking, use_container_width=True)

# --------------------------------------------------
# 9. WATER SUPPLY RECORDS
# --------------------------------------------------
elif page == "Water Supply Records":
    st.subheader("📋 Water Distribution Records")
    st.dataframe(filtered_df.sort_values("Date", ascending=False), use_container_width=True)
    st.download_button("📥 Download Records as CSV", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name="JalTrace_Water_Records.csv", mime="text/csv")

# --------------------------------------------------
# 10. SHORTAGE ANALYSIS
# --------------------------------------------------
elif page == "Shortage Analysis":
    st.subheader("🚨 Locality Shortage Intelligence")
    summary = filtered_df.groupby("Locality").agg(
        Planned=("Planned_Liters", "sum"), Delivered=("Delivered_Liters", "sum"),
        Shortage=("Shortage_Liters", "sum"), Average_Delay=("Delay_Minutes", "mean")
    ).reset_index()
    summary["Fulfillment_Percent"] = (summary["Delivered"] / summary["Planned"].replace(0, np.nan) * 100).fillna(0)
    summary = summary.sort_values("Shortage", ascending=False)
    st.dataframe(summary, use_container_width=True)
    fig = px.bar(summary, x="Locality", y="Shortage", color="Fulfillment_Percent", title="Shortage by Locality")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("AI-Generated Priority Alerts")
    for _, row in summary.iterrows():
        if row["Fulfillment_Percent"] < 70:
            st.error(f"🔴 {row['Locality']}: Critical shortage pattern. Fulfillment is {row['Fulfillment_Percent']:.1f}%. Prioritize verification.")
        elif row["Fulfillment_Percent"] < 90:
            st.warning(f"🟠 {row['Locality']}: Supply below target. Fulfillment is {row['Fulfillment_Percent']:.1f}%. Monitor upcoming deliveries.")
        else:
            st.success(f"🟢 {row['Locality']}: Supply fulfillment is {row['Fulfillment_Percent']:.1f}%.")

# --------------------------------------------------
# 11. AI ANOMALY DETECTION
# --------------------------------------------------
elif page == "AI Anomaly Detection":
    st.subheader("🤖 Machine Learning Anomaly Detection")
    anomaly_df = filtered_df[filtered_df["Anomaly"] == -1].copy()
    normal_df = filtered_df[filtered_df["Anomaly"] == 1].copy()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(filtered_df))
    col2.metric("Anomaly Signals", len(anomaly_df))
    col3.metric("Normal Patterns", len(normal_df))
    st.write("The Isolation Forest model identifies unusual patterns in water delivery quantities and delay records.")
    st.warning("An anomaly is a signal for verification. It is NOT proof of leakage, theft, or misconduct.")
    if not anomaly_df.empty:
        st.subheader("⚠️ Detected Unusual Records")
        st.dataframe(anomaly_df.sort_values("Shortage_Liters", ascending=False), use_container_width=True)
        fig = px.scatter(filtered_df, x="Planned_Liters", y="Delivered_Liters", color="Anomaly_Status", hover_data=["Locality", "Delay_Minutes", "Shortage_Liters"], title="Normal vs Anomalous Water Delivery Patterns")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No anomaly signals found in the selected records.")
    st.subheader("Suggested Verification Actions")
    st.markdown("""\n    - Verify unusually low delivery quantities.
    - Check records of delayed tanker arrivals.
    - Compare repeated shortage patterns across localities.
    - Review the actual delivery records with community representatives.
    - Investigate signals before drawing conclusions.
    """)

# --------------------------------------------------
# 12. SHORTAGE HOTSPOT MAP (SCHEMATIC, NOT GEO-LOCATED)
# --------------------------------------------------
elif page == "Shortage Hotspot Map":
    st.subheader("🗺️ Community Water Shortage Hotspot Map")
    hotspot = filtered_df.groupby("Locality").agg(
        Planned=("Planned_Liters", "sum"), Delivered=("Delivered_Liters", "sum"),
        Shortage=("Shortage_Liters", "sum")
    ).reset_index()
    hotspot["Fulfillment"] = (hotspot["Delivered"] / hotspot["Planned"].replace(0, np.nan) * 100).fillna(0)
    hotspot["Risk"] = np.where(hotspot["Fulfillment"] < 70, "Critical", np.where(hotspot["Fulfillment"] < 90, "Moderate", "Low"))
    # Coordinates are schematic only; they are not real geographic coordinates.
    default_positions = {
        "Green Park": (1, 4), "Lake View": (2, 6), "Hill Top": (4, 8),
        "Old Town": (6, 6), "East Colony": (8, 4), "West Colony": (2, 2),
        "River Side": (4, 1), "Sunrise Nagar": (6, 2), "Market Area": (8, 7),
        "Central Ward": (5, 4)
    }
    positions = dict(default_positions)
    for i, locality in enumerate(hotspot["Locality"]):
        if locality not in positions:
            positions[locality] = ((i % 5) * 2 + 1, (i // 5) * 2 + 1)
    hotspot["X"] = hotspot["Locality"].map(lambda name: positions[name][0])
    hotspot["Y"] = hotspot["Locality"].map(lambda name: positions[name][1])
    st.info("Illustrative hotspot layout only. These point positions are simulated and do not represent real locations on a geographic map.")
    fig_map = px.scatter(
        hotspot, x="X", y="Y", color="Risk", size="Shortage", text="Locality",
        hover_data={"Planned": True, "Delivered": True, "Shortage": True, "Fulfillment": ":.1f", "X": False, "Y": False},
        size_max=55, color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
        title="Water Shortage Risk Across Localities"
    )
    fig_map.update_traces(textposition="top center")
    fig_map.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=600, plot_bgcolor="#f4f7fb", paper_bgcolor="#f4f7fb")
    st.plotly_chart(fig_map, use_container_width=True)
    st.subheader("📊 Hotspot Summary")
    st.dataframe(hotspot.sort_values("Shortage", ascending=False), use_container_width=True)
    st.subheader("🚨 Priority Localities")
    critical = hotspot[hotspot["Risk"] == "Critical"]
    if not critical.empty:
        for _, row in critical.iterrows():
            st.error(f"{row['Locality']} — Critical shortage risk. Fulfillment: {row['Fulfillment']:.1f}%. Recommended: Verify supply records and prioritize community-level review.")
    else:
        st.success("No critical shortage localities in this selection.")

# --------------------------------------------------
# 13. ABOUT PROJECT
# --------------------------------------------------
elif page == "About JalTrace AI":
    st.subheader("💧 About JalTrace AI")
    st.markdown("""\n    **JalTrace AI** is a community water distribution intelligence prototype designed to identify shortages, irregular deliveries, delays, and recurring distribution anomalies.

    ### Problem
    Communities may experience uneven water distribution, delayed deliveries, and recurring shortages. These problems can be difficult to identify when supply records are spread across multiple locations.

    ### Our Solution
    JalTrace AI analyzes water supply records and provides:
    - Water supply monitoring
    - Shortage identification
    - Locality shortage ranking
    - Delivery delay analysis
    - Machine learning anomaly detection
    - AI-based verification recommendations

    ### Technology Stack
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn
    - Isolation Forest

    ### Responsible AI
    An anomaly does not prove theft, leakage, or misconduct. JalTrace AI provides signals for human verification and decision-making. It does not automatically accuse individuals or organizations.

    The default dashboard uses simulated data for demonstration.
    """)

# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------
st.divider()
st.caption("JalTrace AI | AI for Bharat | Community Water Intelligence Prototype")


# --------------------------------------------------
# 15. AI-POWERED SHORTAGE INSIGHTS
# --------------------------------------------------

if page == "AI Shortage Insights":

    st.title("🤖 AI-Powered Shortage Insights")
    st.caption("Locality-level shortage explanations and suggested verification actions")

    if df.empty:
        st.warning("No water supply records available.")
    else:
        insights = (
            df.groupby("Locality")
            .agg(
                Planned=("Planned_Liters", "sum"),
                Delivered=("Delivered_Liters", "sum"),
                Avg_Delay=("Delay_Minutes", "mean"),
                Total_Shortage=("Shortage_Liters", "sum")
            )
            .reset_index()
        )

        insights["Fulfillment"] = (
            insights["Delivered"] / insights["Planned"].replace(0, np.nan) * 100
        ).fillna(0)

        insights = insights.sort_values("Fulfillment", ascending=True)

        st.subheader("📍 Locality Risk Summary")

        for _, row in insights.iterrows():

            locality = row["Locality"]
            fulfillment = row["Fulfillment"]
            shortage = row["Total_Shortage"]
            delay = row["Avg_Delay"]

            if fulfillment < 70:
                risk = "🔴 CRITICAL"
                explanation = "Severe water shortage detected."
                action = "Prioritize water supply and verify the delivery immediately."

            elif fulfillment < 90:
                risk = "🟠 WARNING"
                explanation = "Water delivery is below the expected level."
                action = "Review the delivery schedule and arrange additional supply if required."

            else:
                risk = "🟢 NORMAL"
                explanation = "Water supply is close to the planned quantity."
                action = "Continue monitoring the locality."

            with st.expander(f"{risk} — {locality}", expanded=(fulfillment < 70)):

                col1, col2, col3 = st.columns(3)

                col1.metric("Fulfillment", f"{fulfillment:.1f}%")
                col2.metric("Shortage", f"{shortage:,.0f} L")
                col3.metric("Average Delay", f"{delay:.0f} min")

                st.markdown("### 🔍 Insight")
                st.write(explanation)

                st.markdown("### 📌 Recommended Action")
                st.info(action)

        st.divider()

        st.subheader("🏆 Highest Priority Localities")

        top_risk = insights.head(5)

        st.dataframe(
            top_risk[[
                "Locality",
                "Planned",
                "Delivered",
                "Total_Shortage",
                "Fulfillment",
                "Avg_Delay"
            ]].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These insights are generated using transparent shortage thresholds "
            "and aggregated delivery data. They are decision-support signals, "
            "not proof of leakage, theft, or misconduct."
        )

# --------------------------------------------------
# 16. HISTORICAL TREND ANALYSIS
# --------------------------------------------------

if page == "Historical Trend Analysis":

    st.title("📈 Historical Trend Analysis")
    st.caption("Track water supply performance and shortage patterns over time")

    # Convert Date column to datetime
    trend_df = df.copy()
    trend_df["Date"] = pd.to_datetime(trend_df["Date"])

    # Daily aggregation
    daily = (
        trend_df.groupby("Date")
        .agg(
            Planned=("Planned_Liters", "sum"),
            Delivered=("Delivered_Liters", "sum"),
            Shortage=("Shortage_Liters", "sum")
        )
        .reset_index()
        .sort_values("Date")
    )

    # Calculate fulfillment percentage
    daily["Fulfillment"] = (
        daily["Delivered"] / daily["Planned"].replace(0, np.nan) * 100
    ).fillna(0)

    st.subheader("📊 Daily Water Supply Performance")

    # Fulfillment trend
    fig1 = px.line(
        daily,
        x="Date",
        y="Fulfillment",
        markers=True,
        title="Daily Water Fulfillment (%)"
    )

    fig1.update_layout(
        yaxis_title="Fulfillment (%)",
        xaxis_title="Date"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Shortage trend
    st.subheader("🚰 Daily Water Shortage")

    fig2 = px.area(
        daily,
        x="Date",
        y="Shortage",
        title="Total Daily Water Shortage (Liters)"
    )

    fig2.update_layout(
        yaxis_title="Shortage (Liters)",
        xaxis_title="Date"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Locality comparison
    st.subheader("📍 Locality-wise Fulfillment Comparison")

    locality_trend = (
        trend_df.groupby(["Date", "Locality"])
        .agg(
            Planned=("Planned_Liters", "sum"),
            Delivered=("Delivered_Liters", "sum")
        )
        .reset_index()
    )

    locality_trend["Fulfillment"] = (
        locality_trend["Delivered"]
        / locality_trend["Planned"].replace(0, np.nan)
        * 100
    ).fillna(0)

    fig3 = px.line(
        locality_trend,
        x="Date",
        y="Fulfillment",
        color="Locality",
        markers=True,
        title="Locality-wise Daily Fulfillment Trend"
    )

    fig3.update_layout(
        yaxis_title="Fulfillment (%)",
        xaxis_title="Date"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Trend summary
    st.subheader("🧠 Trend Summary")

    if len(daily) >= 2:

        first_fulfillment = daily.iloc[0]["Fulfillment"]
        last_fulfillment = daily.iloc[-1]["Fulfillment"]

        change = last_fulfillment - first_fulfillment

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "First Day Fulfillment",
            f"{first_fulfillment:.1f}%"
        )

        col2.metric(
            "Latest Fulfillment",
            f"{last_fulfillment:.1f}%"
        )

        col3.metric(
            "Overall Change",
            f"{change:+.1f}%",
            delta=f"{change:+.1f}%"
        )

        if change > 1:
            st.success("Water supply fulfillment is improving overall.")

        elif change < -1:
            st.warning("Water supply fulfillment is declining. Further investigation may be needed.")

        else:
            st.info("Water supply fulfillment is relatively stable.")

    st.caption(
        "Trend results summarize recorded data and do not independently confirm "
        "the cause of a shortage."
    )

# --------------------------------------------------
# 17. COMMUNITY WATER SHORTAGE REPORTING
# --------------------------------------------------

if page == "Community Water Reports":

    st.title("💧 Community Water Shortage Reporting")
    st.caption("Report water supply problems in your locality")

    # Initialize reports in session memory
    if "community_reports" not in st.session_state:
        st.session_state.community_reports = []

    locality_options = sorted(df["Locality"].dropna().unique().tolist())

    st.subheader("📝 Submit a Water Shortage Report")

    with st.form("water_report_form"):

        locality = st.selectbox(
            "Select your locality",
            locality_options
        )

        days_without_water = st.number_input(
            "Days without proper water supply",
            min_value=0,
            max_value=30,
            value=1
        )

        severity = st.selectbox(
            "Shortage Severity",
            ["Low", "Medium", "High", "Critical"]
        )

        description = st.text_area(
            "Describe the water problem",
            placeholder="Explain the shortage or delivery delay..."
        )

        submitted = st.form_submit_button("🚨 Submit Report")

        if submitted:

            report = {
                "Report ID": len(st.session_state.community_reports) + 1,
                "Locality": locality,
                "Days Without Water": days_without_water,
                "Severity": severity,
                "Description": description,
                "Reported At": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            }

            st.session_state.community_reports.append(report)

            st.success("Your water shortage report has been submitted successfully!")

    st.divider()

    st.subheader("📋 Community Submitted Reports")

    reports = st.session_state.community_reports

    if reports:

        reports_df = pd.DataFrame(reports)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Reports", len(reports))

        col2.metric(
            "High/Critical Reports",
            len(reports_df[
                reports_df["Severity"].isin(["High", "Critical"])
            ])
        )

        col3.metric(
            "Affected Localities",
            reports_df["Locality"].nunique()
        )

        st.subheader("🚩 Reports by Locality")

        locality_counts = (
            reports_df.groupby("Locality")
            .size()
            .reset_index(name="Report Count")
            .sort_values("Report Count", ascending=False)
        )

        st.bar_chart(
            locality_counts.set_index("Locality")
        )

        st.subheader("📄 All Submitted Reports")

        st.dataframe(
            reports_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No community reports submitted yet. "
            "Be the first to report a water shortage!"
        )

    st.caption(
        "Community reports are user-submitted information and must be verified "
        "before official action is taken."
    )

# --------------------------------------------------
# 18. SMART PRIORITY RECOMMENDATIONS
# --------------------------------------------------

if page == "Smart Priority Recommendations":

    st.title("🎯 Smart Priority Recommendations")
    st.caption("Combine water supply data and community reports to identify priority localities")

    # Supply data summary
    priority_df = (
        df.groupby("Locality")
        .agg(
            Planned=("Planned_Liters", "sum"),
            Delivered=("Delivered_Liters", "sum"),
            Shortage=("Shortage_Liters", "sum"),
            Avg_Delay=("Delay_Minutes", "mean")
        )
        .reset_index()
    )

    priority_df["Fulfillment"] = (
        priority_df["Delivered"]
        / priority_df["Planned"].replace(0, np.nan)
        * 100
    ).fillna(0).clip(upper=100)

    priority_df["Supply_Risk"] = (
        100 - priority_df["Fulfillment"]
    ).clip(lower=0, upper=100)

    # Initialize complaint metrics
    priority_df["Report_Count"] = 0
    priority_df["Severity_Score"] = 0
    priority_df["Days_Score"] = 0

    # Add community reports if available
    if "community_reports" in st.session_state and st.session_state.community_reports:

        reports_df = pd.DataFrame(st.session_state.community_reports)

        severity_mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Critical": 4
        }

        reports_df["Severity_Value"] = (
            reports_df["Severity"].map(severity_mapping).fillna(1)
        )

        report_summary = (
            reports_df.groupby("Locality")
            .agg(
                Report_Count=("Locality", "size"),
                Severity_Score=("Severity_Value", "mean"),
                Days_Score=("Days Without Water", "mean")
            )
            .reset_index()
        )

        priority_df = priority_df.merge(
            report_summary,
            on="Locality",
            how="left",
            suffixes=("", "_Report")
        )

        priority_df["Report_Count"] = (
            priority_df["Report_Count_Report"].fillna(0)
        )

        priority_df["Severity_Score"] = (
            priority_df["Severity_Score_Report"].fillna(0)
        )

        priority_df["Days_Score"] = (
            priority_df["Days_Score_Report"].fillna(0)
        )

        priority_df = priority_df.drop(
            columns=[
                "Report_Count_Report",
                "Severity_Score_Report",
                "Days_Score_Report"
            ]
        )

    # Normalize complaint volume
    max_reports = priority_df["Report_Count"].max()

    if max_reports > 0:
        priority_df["Complaint_Risk"] = (
            priority_df["Report_Count"] / max_reports * 100
        )
    else:
        priority_df["Complaint_Risk"] = 0

    # Normalize complaint severity (1-4 scale)
    priority_df["Severity_Risk"] = (
        priority_df["Severity_Score"] / 4 * 100
    )

    # Normalize reported days without proper water (max 30)
    priority_df["Days_Risk"] = (
        priority_df["Days_Score"].clip(upper=30) / 30 * 100
    )

    # Calculate combined priority score
    priority_df["Priority_Score"] = (
        priority_df["Supply_Risk"] * 0.45
        + priority_df["Complaint_Risk"] * 0.25
        + priority_df["Severity_Risk"] * 0.20
        + priority_df["Days_Risk"] * 0.10
    )

    # Sort highest priority first
    priority_df = priority_df.sort_values(
        "Priority_Score", ascending=False
    ).reset_index(drop=True)

    # Assign priority category
    def classify_priority(score):
        if score >= 70:
            return "Critical"
        elif score >= 45:
            return "High"
        elif score >= 25:
            return "Medium"
        else:
            return "Low"

    priority_df["Priority_Level"] = (
        priority_df["Priority_Score"].apply(classify_priority)
    )

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Highest Priority Score",
        f"{priority_df['Priority_Score'].max():.1f}/100"
    )

    col2.metric(
        "Localities Analyzed",
        len(priority_df)
    )

    col3.metric(
        "Community Reports",
        int(priority_df["Report_Count"].sum())
    )

    st.divider()

    # Priority chart
    st.subheader("📊 Locality Priority Ranking")

    fig = px.bar(
        priority_df,
        x="Locality",
        y="Priority_Score",
        color="Priority_Level",
        title="Water Shortage Priority Score by Locality",
        hover_data=[
            "Fulfillment",
            "Report_Count",
            "Shortage",
            "Avg_Delay"
        ]
    )

    fig.update_layout(
        yaxis_title="Priority Score (0-100)",
        xaxis_title="Locality",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Priority table
    st.subheader("🏆 Priority Action List")

    st.dataframe(
        priority_df[[
            "Locality",
            "Priority_Score",
            "Priority_Level",
            "Fulfillment",
            "Shortage",
            "Report_Count",
            "Avg_Delay"
        ]].round(2),
        use_container_width=True,
        hide_index=True
    )

    # Recommendations
    st.subheader("💡 Recommended Actions")

    for _, row in priority_df.iterrows():

        locality = row["Locality"]
        score = row["Priority_Score"]
        level = row["Priority_Level"]
        fulfillment = row["Fulfillment"]
        reports = row["Report_Count"]

        if level == "Critical":
            action = "Arrange urgent supply review, verify tanker delivery, and investigate the reported shortage."

        elif level == "High":
            action = "Prioritize the next water delivery and verify community reports."

        elif level == "Medium":
            action = "Monitor delivery schedules and check whether the shortage is recurring."

        else:
            action = "Continue routine monitoring and maintain the planned supply schedule."

        with st.expander(
            f"{level} Priority — {locality} | Score: {score:.1f}"
        ):

            st.write(f"**Fulfillment:** {fulfillment:.1f}%")
            st.write(f"**Community Reports:** {int(reports)}")
            st.info(action)

    st.caption(
        "Priority scores are transparent, rule-based decision-support estimates. "
        "Community reports and simulated records may be incomplete or unverified. "
        "Scores do not prove leakage, theft, or misconduct."
    )