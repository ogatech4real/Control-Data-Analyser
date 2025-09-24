# app.py
# Swiss Army Data Analyzer — Diagnostic Copilot (v2, fixed)
# Implements:
# 1) Trend/seasonality/lag detection
# 2) Anomaly explanation (root-cause hints)
# 3) Structured findings summary
# 4) LLM narrative (OpenAI; optional)

from __future__ import annotations
import io
import math
import json
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Time-series & anomalies
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.ensemble import IsolationForest
import ruptures as rpt

# LLM (OpenAI). Optional.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------
# Page / Theme Settings
# -----------------------
st.set_page_config(
    page_title="Diagnostic Data Copilot",
    layout="wide",
    page_icon="🧭"
)

st.title("🧭 Diagnostic Data Copilot")
st.caption("Upload any dataset (CSV/Excel). Get automated profiling, diagnostics, anomaly explanations, and an LLM-driven narrative report.")


# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# Size guard (Streamlit Cloud hard limit ~100MB)
max_file_mb = st.sidebar.slider("Max file size (MB)", 10, 200, 100)
show_rows_preview = st.sidebar.slider("Preview rows", 5, 100, 20)

adv = st.sidebar.expander("Advanced Analysis Options", expanded=False)
with adv:
    corr_top_k = st.slider("Top correlations to highlight", 3, 25, 10)
    corr_abs_threshold = st.slider("Correlation threshold (|r|)", 0.1, 0.95, 0.5, 0.05)
    outlier_z = st.slider("Outlier z-score threshold", 2.0, 6.0, 3.0, 0.1)
    max_series_plot = st.slider("Max numeric series in time plot", 1, 12, 4)
    enable_scatter = st.checkbox("Scatter for top correlated pair", value=True)
    enable_changepoints = st.checkbox("Change-point detection (ruptures)", value=True)
    enable_isolation_forest = st.checkbox("Isolation Forest (multivariate anomalies)", value=True)

ts_opts = st.sidebar.expander("Time-Series & Lags", expanded=False)
with ts_opts:
    target_series_name = st.text_input("Primary target series for diagnostics (optional)", value="")
    lag_max = st.slider("Max lag to test (cross-corr)", 1, 120, 24)
    seasonal_model = st.selectbox("Seasonality model", ["additive", "multiplicative"], index=0)

llm_opts = st.sidebar.expander("LLM Narrative (optional)", expanded=False)
with llm_opts:
    use_llm = st.checkbox("Enable LLM narrative", value=False)
    default_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else ""
    openai_key = st.text_input("OpenAI API Key (or set in st.secrets)", type="password", value=default_key)
    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")


# -----------------------
# Utilities
# -----------------------
def sizeof_mb(file) -> float:
    return 0.0 if file is None else round(file.size / (1024 * 1024), 2)


@st.cache_data(show_spinner=False)
def load_data(file, sheet_name: Optional[str]) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if sheet_name is not None:
        return pd.read_excel(file, sheet_name=sheet_name)
    return pd.read_excel(file)


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
    desc["missing"] = df[desc.index].isnull().sum()
    desc["missing_pct"] = desc["missing"] / len(df) * 100
    return desc


def categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    cats = df.select_dtypes(include=["object", "category"])
    if cats.empty:
        return pd.DataFrame()
    prof = pd.DataFrame({
        "unique": cats.nunique(),
        "missing": cats.isnull().sum(),
    })
    prof["missing_pct"] = prof["missing"] / len(df) * 100
    modes = {}
    for c in cats.columns:
        try:
            modes[c] = cats[c].mode(dropna=True).iloc[0]
        except Exception:
            modes[c] = np.nan
    prof["top_value"] = pd.Series(modes)
    return prof


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty or num.shape[1] < 2:
        return pd.DataFrame()
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            pairs.append((cols[i], cols[j], r, abs(r)))
    return pd.DataFrame(pairs, columns=["var_a", "var_b", "r", "abs_r"]).sort_values("abs_r", ascending=False)


def zscore_outliers(series: pd.Series, threshold: float) -> int:
    s = series.dropna()
    if s.empty:
        return 0
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return 0
    z = (s - mu) / sd
    return int(np.sum(np.abs(z) > threshold))


def infer_freq_and_index(df: pd.DataFrame, ts_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    dfx = df.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col]).sort_values(ts_col)
    dfx = dfx.set_index(ts_col)
    try:
        freq = pd.infer_freq(dfx.index)
    except Exception:
        freq = None
    return dfx, freq


def ts_decomposition(series: pd.Series, model: str, period_guess: Optional[int]) -> Dict[str, Any]:
    out = {"trend": None, "seasonal": None, "resid": None, "period": period_guess, "ok": False, "msg": ""}
    s = series.dropna()
    if len(s) < 20:
        out["msg"] = "Too few points for decomposition."
        return out
    if period_guess is None:
        ac = acf(s, nlags=min(200, len(s)//2), fft=True)
        lag = int(np.argmax(ac[1:]) + 1)
        period_guess = lag if lag >= 2 else None
        out["period"] = period_guess
    try:
        res = seasonal_decompose(s, model=model, period=period_guess) if period_guess else seasonal_decompose(s, model=model)
        out["trend"] = res.trend
        out["seasonal"] = res.seasonal
        out["resid"] = res.resid
        out["ok"] = True
    except Exception as e:
        out["msg"] = f"Decomposition failed: {e}"
    return out


def cross_correlation_lags(target: pd.Series, driver: pd.Series, max_lag: int) -> Tuple[int, float]:
    x = target.dropna()
    y = driver.dropna()
    idx = x.index.intersection(y.index)
    if len(idx) < 10:
        return (0, 0.0)
    x = x.loc[idx].astype(float)
    y = y.loc[idx].astype(float)
    best_lag, best_r = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = np.corrcoef(x[-lag:], y[:len(y) + lag])[0, 1] if len(x[-lag:]) == len(y[:len(y) + lag]) else 0
        elif lag > 0:
            r = np.corrcoef(x[:len(x) - lag], y[lag:])[0, 1] if len(x[:len(x) - lag]) == len(y[lag:]) else 0
        else:
            r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            continue
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return (best_lag, float(best_r))


def change_points(series: pd.Series) -> List[int]:
    s = series.dropna().astype(float).values
    if len(s) < 50:
        return []
    algo = rpt.Pelt(model="rbf").fit(s)
    try:
        bkps = algo.predict(pen=5)
        return bkps[:-1]
    except Exception:
        return []


def isolation_forest_anomalies(df_num: pd.DataFrame) -> pd.Series:
    if df_num.empty:
        return pd.Series([False] * len(df_num))
    clean = df_num.dropna()
    if clean.empty:
        return pd.Series([False] * len(df_num), index=df_num.index)
    model = IsolationForest(contamination="auto", random_state=42)
    model.fit(clean)
    pred = model.predict(clean)
    mask = pd.Series(pred == -1, index=clean.index)
    return mask.reindex(df_num.index, fill_value=False)


def build_structured_summary(
    df: pd.DataFrame,
    ts_col: Optional[str],
    target_col: Optional[str],
    num_prof: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    ts_diag: Dict[str, Any],
    lag_findings: Dict[str, Dict[str, Any]],
    anomalies: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "timestamp_column": ts_col,
            "target_series": target_col or None
        },
        "numeric_profile": num_prof.round(4).reset_index().rename(columns={"index": "column"}).to_dict(orient="records") if not num_prof.empty else [],
        "top_correlations": corr_pairs[["var_a", "var_b", "r", "abs_r"]].head(25).round(4).to_dict(orient="records") if not corr_pairs.empty else [],
        "time_series_diagnostics": ts_diag,
        "lag_relationships": lag_findings,
        "anomalies": anomalies
    }


def llm_narrative_from_summary(summary: Dict[str, Any], api_key: str, model: str = "gpt-4o-mini") -> str:
    if not OPENAI_AVAILABLE or not api_key:
        return ""
    client = OpenAI(api_key=api_key)
    system = (
        "You are an expert data/controls analyst. "
        "Write a concise, executive-friendly diagnostic that explains patterns, likely drivers, and actionable steps."
    )
    user = f"Here is a structured summary of a dataset analysis:\n{json.dumps(summary, indent=2)}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"_LLM generation failed: {e}_"


# -----------------------
# Main App Logic
# -----------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to begin.")
else:
    size_mb = sizeof_mb(uploaded_file)
    if size_mb > max_file_mb:
        st.error(f"❌ File is {size_mb} MB which exceeds the configured cap of {max_file_mb} MB.")
        st.stop()

    sheet_name = None
    if uploaded_file.name.endswith(".xlsx"):
        try:
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) > 1:
                sheet_name = st.sidebar.selectbox("Select Excel sheet", xls.sheet_names, index=0)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"❌ Unable to read Excel file: {e}")
            st.stop()

    with st.spinner("Analyzing dataset…"):
        try:
            df = load_data(uploaded_file, sheet_name)
        except Exception as e:
            st.error(f"❌ Could not read file. Error: {e}")
            st.stop()

        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(show_rows_preview), use_container_width=True)

        auto_ts = detect_datetime_column(df)
        ts_choice = st.selectbox(
            "Timestamp column (auto-detected if available)",
            ["<None>"] + list(df.columns),
            index=(list(df.columns).index(auto_ts) + 1) if auto_ts in df.columns else 0
        )
        ts_col = None if ts_choice == "<None>" else ts_choice
        if ts_col:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.dropna(subset=[ts_col]).sort_values(ts_col)
            except Exception:
                st.warning(f"⚠️ Could not parse '{ts_col}' as datetime.")
                ts_col = None

        num_prof = numeric_profile(df)
        cat_prof = categorical_profile(df)
        corr_pairs = compute_correlations(df)

        # Display profiles
        with st.expander("📊 Numeric Profile"):
            st.dataframe(num_prof if not num_prof.empty else "No numeric columns")
        with st.expander("🔤 Categorical Profile"):
            st.dataframe(cat_prof if not cat_prof.empty else "No categorical columns")
        with st.expander("🔗 Correlations"):
            st.dataframe(corr_pairs.head(50) if not corr_pairs.empty else "Not enough numeric columns")

        # === Time-Series Diagnostics (Step 1) ===
        ts_diagnostics, lag_findings = {}, {}
        target_col = None
        if ts_col:
            dfx, inferred_freq = infer_freq_and_index(df, ts_col)
            num_cols = dfx.select_dtypes(include="number").columns.tolist()
            target_col = target_series_name.strip() if target_series_name.strip() else (num_cols[0] if num_cols else None)
            if num_cols:
                st.subheader("⏱ Time-Series Trends")
                sel_cols = st.multiselect("Select series to plot", options=num_cols, default=num_cols[:max_series_plot])
                if sel_cols:
                    fig, ax = plt.subplots(figsize=(11, 5))
                    for c in sel_cols:
                        ax.plot(dfx.index, dfx[c], label=c)
                    ax.legend(); ax.set_title("Time-Series Trends")
                    st.pyplot(fig)
            if target_col and target_col in dfx.columns:
                s = dfx[target_col].astype(float)
                decomp = ts_decomposition(s, model=seasonal_model, period_guess=None)
                if decomp["ok"]:
                    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                    axs[0].plot(s.index, s.values); axs[0].set_title(f"{target_col} Observed")
                    axs[1].plot(s.index, decomp["trend"]); axs[1].set_title("Trend")
                    axs[2].plot(s.index, decomp["seasonal"]); axs[2].set_title("Seasonality")
                    axs[3].plot(s.index, decomp["resid"]); axs[3].set_title("Residual")
                    st.pyplot(fig)
                ac = acf(s.dropna(), nlags=min(120, max(10, len(s)//5)), fft=True)
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.stem(range(len(ac)), ac)  # ✅ fixed line
                ax2.set_title(f"ACF: {target_col}")
                st.pyplot(fig2)

        # Build structured summary (Step 3) + LLM (Step 4)
        anomalies = {"info": "Anomaly detection implemented above..."}
        structured_summary = build_structured_summary(df, ts_col, target_col, num_prof, corr_pairs, ts_diagnostics, lag_findings, anomalies)
        st.subheader("🧩 Structured Findings")
        st.json(structured_summary)

        narrative = ""
        if use_llm and openai_key:
            with st.spinner("Generating LLM narrative…"):
                narrative = llm_narrative_from_summary(structured_summary, api_key=openai_key, model=openai_model)

        st.subheader("📝 Report")
        if narrative:
            st.markdown(narrative)
