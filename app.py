# app.py
# Swiss Army Data Analyzer — Diagnostic Copilot (v2)
# Implements:
# 1) Trend/seasonality/lag detection
# 2) Anomaly explanation (root-cause hints)
# 3) Structured findings summary
# 4) LLM narrative (OpenAI; optional)

from __future__ import annotations
import io
import math
import json
import zipfile
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

# LLM (OpenAI). If you prefer another provider, wrap behind the same interface.
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
    # direct dtype
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # convertible
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
    """Set datetime index and attempt to infer frequency for decomposition."""
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
    """Decompose series into trend/seasonal/resid if possible."""
    out = {"trend": None, "seasonal": None, "resid": None, "period": period_guess, "ok": False, "msg": ""}
    s = series.dropna()
    if len(s) < 20:
        out["msg"] = "Too few points for decomposition."
        return out
    # If no explicit period, try a heuristic (e.g., 24, 48, 60) based on autocorr peak
    if period_guess is None:
        # Find first ACF peak as crude seasonality hint
        ac = acf(s, nlags=min(200, len(s)//2), fft=True)
        # Ignore lag 0; find max beyond lag 1
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
    """Return lag (driver leads +lag) with maximum absolute correlation and the correlation value."""
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
        # heuristic penalty; tune per domain
        bkps = algo.predict(pen=5)
        # bkps returns end indices of segments; drop final end
        return bkps[:-1]
    except Exception:
        return []


def isolation_forest_anomalies(df_num: pd.DataFrame) -> pd.Series:
    """Return boolean mask of anomalies using IsolationForest over numeric columns."""
    if df_num.empty:
        return pd.Series([False] * len(df_num))
    clean = df_num.dropna()
    if clean.empty:
        return pd.Series([False] * len(df_num), index=df_num.index)
    model = IsolationForest(contamination="auto", random_state=42)
    model.fit(clean)
    pred = model.predict(clean)  # -1 = anomaly
    mask = pd.Series(pred == -1, index=clean.index)
    # Reindex to original index
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
    """Step 3: Structured JSON-like object that the LLM can consume."""
    summary: Dict[str, Any] = {
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
    return summary


def llm_narrative_from_summary(
    summary: Dict[str, Any],
    api_key: str,
    model: str = "gpt-4o-mini"
) -> str:
    """Step 4: LLM narrative. If OpenAI unavailable or key missing, returns empty string."""
    if not OPENAI_AVAILABLE or not api_key:
        return ""
    client = OpenAI(api_key=api_key)

    system = (
        "You are an expert data/controls analyst. "
        "Write a concise, executive-friendly diagnostic that explains patterns, likely drivers, and actionable steps. "
        "Be concrete, avoid fluff, and quantify effects when possible."
    )
    user = (
        "Here is a structured summary of a dataset analysis. "
        "Please produce a diagnostic narrative with:\n"
        "1) Key behaviours (trend/seasonality/oscillation),\n"
        "2) Likely root causes using lag/correlation evidence,\n"
        "3) Actionable recommendations (stability, cost/CO2, data quality),\n"
        "4) Risks/assumptions to validate.\n\n"
        f"SUMMARY JSON:\n{json.dumps(summary, indent=2)}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"_LLM generation failed: {e}_"


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------
# Main App
# -----------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to begin.")
else:
    # Size guard
    size_mb = sizeof_mb(uploaded_file)
    if size_mb > max_file_mb:
        st.error(f"❌ File is {size_mb} MB which exceeds the configured cap of {max_file_mb} MB.")
        st.stop()

    # Excel sheet selection
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
        # Load
        try:
            df = load_data(uploaded_file, sheet_name)
        except Exception as e:
            st.error(f"❌ Could not read file. Error: {e}")
            st.stop()

        # Preview
        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(show_rows_preview), use_container_width=True)

        # Timestamp detect & choose
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
                st.warning(f"⚠️ Could not parse '{ts_col}' as datetime. Skipping time-series operations.")
                ts_col = None

        # Profiles & correlations
        num_prof = numeric_profile(df)
        cat_prof = categorical_profile(df)
        corr_pairs = compute_correlations(df)

        # Display profiles
        with st.expander("📊 Numeric Profile (describe + missing)"):
            if num_prof.empty:
                st.info("No numeric columns detected.")
            else:
                st.dataframe(num_prof, use_container_width=True)
        with st.expander("🔤 Categorical Profile"):
            if cat_prof.empty:
                st.info("No categorical columns detected.")
            else:
                st.dataframe(cat_prof, use_container_width=True)
        with st.expander("🔗 Correlation Pairs (sorted by |r|)"):
            if corr_pairs.empty:
                st.info("Not enough numeric columns for correlations.")
            else:
                st.dataframe(
                    corr_pairs.head(50).round({"r": 3, "abs_r": 3}),
                    use_container_width=True
                )

        # === Step 1: Trend / Seasonality / Lag Detection ===
        ts_diagnostics: Dict[str, Any] = {"summary": "no time-series column selected"}
        lag_findings: Dict[str, Dict[str, Any]] = {}

        if ts_col:
            dfx, inferred_freq = infer_freq_and_index(df, ts_col)
            ts_diagnostics = {"inferred_freq": inferred_freq, "targets": {}}

            # Pick target for deep diagnostics
            target_col = target_series_name.strip() if target_series_name.strip() else None
            num_cols = dfx.select_dtypes(include="number").columns.tolist()

            # If user didn't pick, default to the first numeric column
            if not target_col:
                target_col = num_cols[0] if num_cols else None

            # Time-series plots (user-select subset)
            if num_cols:
                st.subheader(f"⏱ Time-Series Trends ({ts_col})")
                sel_cols = st.multiselect(
                    "Select series to plot",
                    options=num_cols,
                    default=num_cols[:max_series_plot]
                )
                if sel_cols:
                    fig, ax = plt.subplots(figsize=(11, 5))
                    for c in sel_cols:
                        ax.plot(dfx.index, dfx[c], label=c)
                    ax.legend(loc="upper right", ncol=2)
                    ax.set_title("Time-Series Trends")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Value")
                    st.pyplot(fig)

            # Decompose target series
            if target_col and target_col in dfx.columns:
                s = dfx[target_col].astype(float)
                decomp = ts_decomposition(s, model=seasonal_model, period_guess=None)
                ts_diagnostics["targets"][target_col] = {
                    "decomposition_ok": decomp["ok"],
                    "period": decomp["period"],
                    "msg": decomp.get("msg", "")
                }

                if decomp["ok"]:
                    # Plot decomposition
                    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                    axs[0].plot(s.index, s.values); axs[0].set_title(f"{target_col} - Observed")
                    axs[1].plot(s.index, decomp["trend"]); axs[1].set_title("Trend")
                    axs[2].plot(s.index, decomp["seasonal"]); axs[2].set_title("Seasonality")
                    axs[3].plot(s.index, decomp["resid"]); axs[3].set_title("Residual")
                    plt.tight_layout()
                    st.pyplot(fig)

                # ACF/PACF
                ac = acf(s.dropna(), nlags=min(120, max(10, len(s)//5)), fft=True)
                pc = pacf(s.dropna(), nlags=min(40, max(10, len(s)//10)))
                ts_diagnostics["targets"][target_col]["acf_peaks"] = int(np.argmax(ac[1:]) + 1) if len(ac) > 1 else None

                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.stem(range(len(ac)), ac, use_line_collection=True)
                ax2.set_title(f"ACF: {target_col}")
                ax2.set_xlabel("Lag")
                ax2.set_ylabel("Autocorr")
                st.pyplot(fig2)

                # Cross-correlation vs other numeric drivers
                lag_findings[target_col] = {}
                for c in [c for c in num_cols if c != target_col]:
                    lag, r = cross_correlation_lags(dfx[target_col], dfx[c], max_lag=lag_max)
                    lag_findings[target_col][c] = {"lag": lag, "r": r}

        else:
            target_col = None

        # === Step 2: Anomaly Explanation (root-cause hints) ===
        anomalies: Dict[str, Any] = {"zscore_outliers": {}, "changepoints": {}, "isolation_forest": {}}
        # Column-level outliers (z-score)
        for col in df.select_dtypes(include="number").columns:
            anomalies["zscore_outliers"][col] = zscore_outliers(df[col], outlier_z)

        # Change points for target series (if any)
        if ts_col and target_col and target_col in df.columns:
            cp_idx = change_points(df.set_index(ts_col)[target_col].astype(float))
            anomalies["changepoints"][target_col] = cp_idx

        # Isolation Forest multivariate anomalies
        if enable_isolation_forest:
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                mask = isolation_forest_anomalies(num_df)
                anomalies["isolation_forest"]["anomaly_count"] = int(mask.sum())
                anomalies["isolation_forest"]["anomaly_ratio"] = float(mask.mean())
            else:
                anomalies["isolation_forest"]["info"] = "No numeric data."

        # === Step 3: Structured Summary ===
        structured_summary = build_structured_summary(
            df=df,
            ts_col=ts_col,
            target_col=target_col,
            num_prof=num_prof,
            corr_pairs=corr_pairs,
            ts_diag=ts_diagnostics,
            lag_findings=lag_findings,
            anomalies=anomalies
        )

        # Present machine findings (human-readable)
        st.subheader("🧩 Structured Findings (Machine Summary)")
        st.json(structured_summary)

        # === Step 4: LLM Narrative Layer ===
        narrative = ""
        if use_llm:
            if not OPENAI_AVAILABLE:
                st.warning("OpenAI package not available in this environment.")
            elif not openai_key:
                st.warning("Please provide an OpenAI API key (or add OPENAI_API_KEY in st.secrets).")
            else:
                with st.spinner("Generating LLM narrative…"):
                    narrative = llm_narrative_from_summary(structured_summary, api_key=openai_key, model=openai_model)

        # Assemble final Markdown report
        st.subheader("📝 Auto-Generated Report")
        report_sections: List[str] = []

        # Overview
        report_sections.append("## Dataset Overview")
        report_sections.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        report_sections.append(f"- Columns: {list(df.columns)}")
        if ts_col:
            report_sections.append(f"- Time axis: **{ts_col}**")
        if target_col:
            report_sections.append(f"- Target series: **{target_col}**")

        # Key stats
        report_sections.append("\n## Key Statistics (Numeric)")
        report_sections.append(num_prof.round(3).to_string() if not num_prof.empty else "No numeric columns.")

        # Top correlations
        report_sections.append("\n## Top Correlations")
        report_sections.append(
            corr_pairs.head(corr_top_k)[["var_a", "var_b", "r"]].round(3).to_string(index=False)
            if not corr_pairs.empty else "Not enough numeric columns for correlation analysis."
        )

        # Diagnostics summary
        report_sections.append("\n## Diagnostics Summary")
        if ts_col and target_col:
            diag = structured_summary["time_series_diagnostics"]["targets"].get(target_col, {})
            period = diag.get("period", None)
            report_sections.append(f"- Decomposition: {'OK' if diag.get('decomposition_ok') else 'Failed'}; "
                                   f"period guess: {period}")
            # Lag highlights
            lags = structured_summary["lag_relationships"].get(target_col, {})
            if lags:
                top_lags = sorted(lags.items(), key=lambda kv: abs(kv[1]['r']), reverse=True)[:5]
                bullets = [f"**{k}** → lag={v['lag']}, r={v['r']:.2f}" for k, v in top_lags]
                report_sections.append(f"- Strongest lag relationships: {', '.join(bullets)}")
        else:
            report_sections.append("- No time-series diagnostics (timestamp or target unavailable).")

        # Anomalies summary
        report_sections.append("\n## Anomalies Summary")
        if anomalies["zscore_outliers"]:
            zs = [f"{k}: {v} points" for k, v in anomalies["zscore_outliers"].items() if v > 0]
            report_sections.append("- Z-score outliers → " + (", ".join(zs) if zs else "none flagged"))
        if "changepoints" in anomalies and target_col in anomalies["changepoints"]:
            cps = anomalies["changepoints"][target_col]
            report_sections.append(f"- Change points in {target_col} → {len(cps)} segments flagged" if cps else f"- No change points flagged in {target_col}")
        if "isolation_forest" in anomalies and "anomaly_count" in anomalies["isolation_forest"]:
            ac = anomalies["isolation_forest"]["anomaly_count"]
            ar = anomalies["isolation_forest"]["anomaly_ratio"]
            report_sections.append(f"- Isolation Forest (multivariate) → {ac} anomalies ({ar:.2%})")

        # LLM narrative
        if use_llm and narrative:
            report_sections.append("\n## AI Diagnostic Narrative")
            report_sections.append(narrative)
        elif use_llm and not narrative:
            report_sections.append("\n## AI Diagnostic Narrative")
            report_sections.append("_Narrative generation unavailable._")

        final_report_md = "\n".join(report_sections)
        st.markdown(final_report_md)

        # Downloads
        st.markdown("### Downloads")
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=final_report_md.encode("utf-8"),
            file_name="diagnostic_report.md",
            mime="text/markdown"
        )

        if not num_prof.empty:
            st.download_button(
                label="📥 Download Numeric Profile (CSV)",
                data=num_prof.to_csv().encode("utf-8"),
                file_name="numeric_profile.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("Built by Adewale • Diagnostic Copilot v2")
