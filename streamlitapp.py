# ============================================================
# PJME Energy Consumption Forecasting — Streamlit App
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PJME Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ PJME Hourly Energy Consumption Forecast")
st.markdown(
    "Upload your **PJME hourly CSV** file, explore the data, train an XGBoost model, "
    "and analyse forecast accuracy — all in one place."
)

# ─────────────────────────────────────────────
# Sidebar — File Upload & Settings
# ─────────────────────────────────────────────
st.sidebar.header("📂 Data & Model Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload PJME_hourly.csv",
    type=["csv"],
    help="CSV must contain 'Datetime' and 'PJME_MW' columns.",
)

split_date = st.sidebar.date_input(
    "Train / Test Split Date",
    value=pd.Timestamp("2015-01-01"),
    help="Data up to this date → training; after → test.",
)

n_estimators = st.sidebar.slider(
    "XGBoost — n_estimators (trees)", 100, 2000, 1000, step=100
)
early_stopping = st.sidebar.slider(
    "XGBoost — early_stopping_rounds", 10, 200, 50, step=10
)

run_model = st.sidebar.button("🚀 Train Model", use_container_width=True)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df


def create_features(df: pd.DataFrame, label: str = None):
    df = df.copy()
    df["date"] = df.index
    df["hour"]       = df["date"].dt.hour
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["quarter"]    = df["date"].dt.quarter
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    FEATURES = ["hour", "dayofweek", "quarter", "month", "year",
                "dayofyear", "dayofmonth", "weekofyear"]
    X = df[FEATURES]
    if label:
        y = df[label]
        return X, y
    return X


def mpl_fig_to_streamlit(fig):
    """Render a matplotlib figure via st.pyplot cleanly."""
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

if uploaded_file is None:
    st.info("👈  Upload a CSV file in the sidebar to get started.")
    st.stop()

# ── Load ──────────────────────────────────────
pjme = load_data(uploaded_file)

st.success(f"✅ Loaded **{len(pjme):,}** rows  |  "
           f"Date range: **{pjme.index.min().date()}** → **{pjme.index.max().date()}**")

# ─────────────────────────────────────────────
# Tab Layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Explorer", "🔀 Train / Test Split", "🤖 Model & Predictions", "📐 Error Analysis"]
)

# ══════════════════════════════════════════════
# TAB 1 — Data Explorer
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(pjme.head(50), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(pjme.describe(), use_container_width=True)

    st.subheader("Full Time Series")
    fig, ax = plt.subplots(figsize=(14, 4))
    pjme["PJME_MW"].plot(ax=ax, linewidth=0.6, color="#1f77b4")
    ax.set_title("PJME Hourly Energy Consumption (MW)")
    ax.set_ylabel("MW")
    ax.set_xlabel("")
    plt.tight_layout()
    mpl_fig_to_streamlit(fig)

# ══════════════════════════════════════════════
# TAB 2 — Train / Test Split
# ══════════════════════════════════════════════
with tab2:
    split_ts = pd.Timestamp(split_date)
    pjme_train = pjme.loc[pjme.index <= split_ts].copy()
    pjme_test  = pjme.loc[pjme.index >  split_ts].copy()

    col1, col2 = st.columns(2)
    col1.metric("Training rows", f"{len(pjme_train):,}")
    col2.metric("Test rows",     f"{len(pjme_test):,}")

    st.subheader("Training vs Test Split Visualisation")
    combined = pjme_test.rename(columns={"PJME_MW": "Test Set"}).join(
        pjme_train.rename(columns={"PJME_MW": "Training Set"}), how="outer"
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    combined["Training Set"].plot(ax=ax, label="Training Set", linewidth=0.5, color="#2196F3")
    combined["Test Set"].plot(ax=ax, label="Test Set", linewidth=0.5, color="#FF5722")
    ax.set_title("Hourly Energy Consumption — Train / Test Split")
    ax.set_ylabel("MW")
    ax.legend()
    plt.tight_layout()
    mpl_fig_to_streamlit(fig)

# ══════════════════════════════════════════════
# TAB 3 — Model & Predictions
# ══════════════════════════════════════════════
with tab3:
    if not run_model:
        st.info("Click **🚀 Train Model** in the sidebar to train XGBoost and see predictions.")
    else:
        split_ts = pd.Timestamp(split_date)
        pjme_train = pjme.loc[pjme.index <= split_ts].copy()
        pjme_test  = pjme.loc[pjme.index >  split_ts].copy()

        X_train, y_train = create_features(pjme_train, label="PJME_MW")
        X_test,  y_test  = create_features(pjme_test,  label="PJME_MW")

        with st.spinner("Training XGBoost model…"):
            reg = xgb.XGBRegressor(
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping,
                eval_metric="rmse",
                verbosity=0,
            )
            reg.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )
        st.success("Model trained!")

        # ── Metrics ──────────────────────────────
        preds = reg.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        mape  = np.mean(np.abs((y_test - preds) / y_test)) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE",  f"{rmse:,.0f} MW")
        c2.metric("MAE",   f"{mae:,.0f} MW")
        c3.metric("MAPE",  f"{mape:.2f} %")

        # ── Feature Importance ───────────────────
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_importance(reg, ax=ax, height=0.7, importance_type="gain")
        ax.set_title("XGBoost Feature Importance (Gain)")
        plt.tight_layout()
        mpl_fig_to_streamlit(fig)

        # ── Full Actual vs Predicted ─────────────
        pjme_test = pjme_test.copy()
        pjme_test["MW_Prediction"] = preds
        pjme_all = pd.concat([pjme_test, pjme_train]).sort_index()

        st.subheader("Actual vs Predicted — Full Period")
        fig, ax = plt.subplots(figsize=(14, 4))
        pjme_all["PJME_MW"].plot(ax=ax, label="Actual", linewidth=0.5, color="#2196F3")
        pjme_all["MW_Prediction"].plot(ax=ax, label="Predicted", linewidth=0.5,
                                       color="#FF5722", style=".")
        ax.set_ylabel("MW")
        ax.legend()
        ax.set_title("Full Period: Actual vs Predicted Energy Consumption")
        plt.tight_layout()
        mpl_fig_to_streamlit(fig)

        # ── Zoom Window ──────────────────────────
        st.subheader("Zoom into Specific Period")
        z_col1, z_col2 = st.columns(2)
        zoom_start = z_col1.date_input("From", value=pd.Timestamp("2015-01-01"))
        zoom_end   = z_col2.date_input("To",   value=pd.Timestamp("2015-12-31"))

        fig, ax = plt.subplots(figsize=(14, 4))
        pjme_all[["PJME_MW", "MW_Prediction"]].plot(ax=ax, style=["-", "."],
                                                      linewidth=0.8)
        ax.set_xbound(lower=str(zoom_start), upper=str(zoom_end))
        ax.set_ylim(0, 60000)
        ax.set_title(f"Forecast vs Actuals: {zoom_start} → {zoom_end}")
        ax.legend(["Actual", "Predicted"])
        plt.tight_layout()
        mpl_fig_to_streamlit(fig)

        # Store for Tab 4
        st.session_state["pjme_test_results"] = pjme_test

# ══════════════════════════════════════════════
# TAB 4 — Error Analysis
# ══════════════════════════════════════════════
with tab4:
    if "pjme_test_results" not in st.session_state:
        st.info("Train the model first (Tab 3) to see error analysis.")
    else:
        df_res = st.session_state["pjme_test_results"].copy()
        df_res["error"]     = df_res["PJME_MW"] - df_res["MW_Prediction"]
        df_res["abs_error"] = df_res["error"].abs()

        # Add time features for grouping
        df_res["year"]       = df_res.index.year
        df_res["month"]      = df_res.index.month
        df_res["dayofmonth"] = df_res.index.day

        error_by_day = (
            df_res.groupby(["year", "month", "dayofmonth"])
            .mean()[["PJME_MW", "MW_Prediction", "error", "abs_error"]]
        )

        # ── Error Distribution ───────────────────
        st.subheader("Prediction Error Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_res["error"], bins=80, color="#5C6BC0", edgecolor="none")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
        ax.set_xlabel("Error (MW)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Prediction Errors")
        ax.legend()
        plt.tight_layout()
        mpl_fig_to_streamlit(fig)

        # ── Worst / Best Days ────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("⬆️ Top 10 Overestimated Days")
            st.caption("Days where model predicted too HIGH (most negative error)")
            st.dataframe(
                error_by_day.sort_values("error", ascending=True).head(10).style.format("{:.1f}"),
                use_container_width=True,
            )

        with col_b:
            st.subheader("🎯 Top 10 Most Accurate Days")
            st.caption("Days with lowest absolute error")
            st.dataframe(
                error_by_day.sort_values("abs_error", ascending=True).head(10).style.format("{:.1f}"),
                use_container_width=True,
            )

        # ── Daily Error Over Time ────────────────
        st.subheader("Daily Absolute Error Over Time")
        fig, ax = plt.subplots(figsize=(14, 3))
        error_by_day["abs_error"].plot(ax=ax, linewidth=0.7, color="#EF5350")
        ax.set_ylabel("Absolute Error (MW)")
        ax.set_title("Daily Mean Absolute Error (Test Period)")
        plt.tight_layout()
        mpl_fig_to_streamlit(fig)

        # ── Download results ─────────────────────
        st.subheader("📥 Download Results")
        csv_bytes = df_res.to_csv().encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name="pjme_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
