# app.py
# Swiss Army Data Analyzer — Streamlit MVP (production-ready)
# Author: Adewale
# Purpose: Upload any CSV/Excel, auto-profile, plot time series, and generate a shareable report.

from __future__ import annotations
import io
import math
import zipfile
from typing import Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Page / Theme Settings
# -----------------------
st.set_page_config(
    page_title="Swiss Army Data Analyzer",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Swiss Army Data Analyzer")
st.caption("Upload any dataset (CSV/Excel). Get instant insights, visualizations, and a downloadable report.")


# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Upload & Options")

uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
max_file_mb = st.sidebar.slider("Max file size (MB)", 10, 200, 50)
show_rows_preview = st.sidebar.slider("Preview rows", 5, 100, 20)

advanced_opts = st.sidebar.expander("Advanced Options", expanded=False)
with advanced_opts:
    corr_top_k = st.slider("Top correlations to highlight", 3, 20, 8)
    corr_abs_threshold = st.slider("Correlation threshold (abs)", 0.1, 0.95, 0.5, 0.05)
    outlier_z = st.slider("Outlier z-score threshold", 2.0, 5.0, 3.0, 0.1)
    plot_series_limit = st.slider("Max numeric series in time plot", 1, 10, 3)
    enable_scatter = st.checkbox("Generate scatter for top correlated pair", value=True)


# -----------------------
# Utilities
# -----------------------
def sizeof_mb(file) -> float:
    return 0.0 if file is None else round(file.size / (1024 * 1024), 2)


@st.cache_data(show_spinner=False)
def load_data(file, sheet_name: Optional[str]) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    # Excel
    if sheet_name is not None:
        return pd.read_excel(file, sheet_name=sheet_name)
    # Default first sheet
    return pd.read_excel(file)


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    # Try exact dtype first
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # Try convertible columns
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    # Robust describe for numeric with percentiles
    if df.select_dtypes(include="number").empty:
        return pd.DataFrame()
    desc = df.select_dtypes(include="number").describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
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
    # Most frequent value
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
    # Flatten upper triangle
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            pairs.append((cols[i], cols[j], r, abs(r)))
    pairs_df = pd.DataFrame(pairs, columns=["var_a", "var_b", "r", "abs_r"]).sort_values("abs_r", ascending=False)
    return pairs_df


def zscore_outliers(series: pd.Series, threshold: float) -> int:
    if series.dropna().empty:
        return 0
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return 0
    z = (series - mu) / sd
    return int(np.sum(np.abs(z) > threshold))


def build_narrative_insights(
    df: pd.DataFrame,
    num_prof: pd.DataFrame,
    cat_prof: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    ts_col: Optional[str],
    corr_top_k: int,
    corr_abs_threshold: float,
    outlier_z: float
) -> List[str]:
    insights: List[str] = []
    insights.append("### Quick Insights")

    # Missingness
    missing = df.isnull().sum()
    high_missing = missing[missing > 0].sort_values(ascending=False)
    if not high_missing.empty:
        top_missing = [f"**{c}** ({int(v)}; {v/len(df)*100:.1f}%)" for c, v in high_missing.head(5).items()]
        insights.append(f"- Columns with notable missing values: {', '.join(top_missing)}.")
    else:
        insights.append("- No missing values detected.")

    # Correlations
    if not corr_pairs.empty:
        strong = corr_pairs[corr_pairs["abs_r"] >= corr_abs_threshold].head(corr_top_k)
        if not strong.empty:
            bullets = [f"**{a} ↔ {b}** (r={r:.2f})" for a, b, r in strong[["var_a", "var_b", "r"]].values]
            insights.append(f"- Strongest correlations (|r| ≥ {corr_abs_threshold:.2f}): {', '.join(bullets)}.")
        else:
            insights.append(f"- No correlations above the |r| ≥ {corr_abs_threshold:.2f} threshold.")
    else:
        insights.append("- Not enough numeric columns for correlation analysis.")

    # Outliers (numeric)
    if not num_prof.empty:
        outlier_flags = []
        for col in num_prof.index:
            n_out = zscore_outliers(df[col], outlier_z)
            if n_out > 0:
                outlier_flags.append(f"**{col}** (~{n_out} outliers @ z>{outlier_z:.1f})")
        if outlier_flags:
            insights.append(f"- Potential outliers: {', '.join(outlier_flags)}.")
        else:
            insights.append(f"- No strong outlier signals at z>{outlier_z:.1f}.")

    # Time series
    if ts_col:
        insights.append(f"- Time axis detected: **{ts_col}**. Trends and oscillations visualized for leading numeric fields.")
    else:
        insights.append("- No clear timestamp column detected; time-series plots skipped.")

    insights.append("- Use correlation pairs and outlier flags as starting points for root-cause or stability analysis.")
    return insights


def generate_markdown_report(
    df: pd.DataFrame,
    num_prof: pd.DataFrame,
    cat_prof: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    ts_col: Optional[str],
    insights: List[str],
    corr_top_k: int
) -> str:
    lines: List[str] = []
    lines.append("## Dataset Overview")
    lines.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"- Columns: {list(df.columns)}")

    lines.append("\n## Column Profiles (Numeric)")
    if num_prof.empty:
        lines.append("- No numeric columns detected.")
    else:
        lines.append(num_prof.round(3).to_string())

    lines.append("\n## Column Profiles (Categorical)")
    if cat_prof.empty:
        lines.append("- No categorical columns detected.")
    else:
        lines.append(cat_prof.to_string())

    lines.append("\n## Correlations (Top Pairs)")
    if corr_pairs.empty:
        lines.append("- Not enough numeric columns for correlation analysis.")
    else:
        lines.append(corr_pairs.head(corr_top_k)[["var_a", "var_b", "r"]].round(3).to_string(index=False))

    if ts_col:
        lines.append(f"\n## Time Axis Detected\n- Timestamp column: **{ts_col}**")

    lines.append("\n## Insights")
    lines.extend(insights)
    return "\n".join(lines)


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------
# Main App Logic
# -----------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to begin.")
else:
    # File size guard
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
            uploaded_file.seek(0)  # reset pointer
        except Exception as e:
            st.error(f"❌ Unable to read Excel file: {e}")
            st.stop()

    with st.spinner("Analyzing dataset…"):
        try:
            df = load_data(uploaded_file, sheet_name)
        except Exception as e:
            st.error(f"❌ Could not read file. Error: {e}")
            st.stop()

        # Preview
        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(show_rows_preview), use_container_width=True)

        # Detect/choose timestamp column
        auto_ts = detect_datetime_column(df)
        ts_choice = st.selectbox(
            "Timestamp column (auto-detected if available)",
            ["<None>"] + list(df.columns),
            index=(list(df.columns).index(auto_ts) + 1) if auto_ts in df.columns else 0
        )
        ts_col = None if ts_choice == "<None>" else ts_choice

        # Coerce ts if selected
        if ts_col:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                # If too many NaT, warn and unset
                nat_ratio = df[ts_col].isna().mean()
                if nat_ratio > 0.5:
                    st.warning(f"⚠️ Over 50% of values in '{ts_col}' are not valid datetimes. Time-series plots may be unreliable.")
                df = df.sort_values(ts_col)
            except Exception:
                st.warning(f"⚠️ Could not parse '{ts_col}' as datetime. Time-series will be skipped.")
                ts_col = None

        # Profiles
        num_prof = numeric_profile(df)
        cat_prof = categorical_profile(df)
        corr_pairs = compute_correlations(df)

        # Panels
        colA, colB = st.columns([1, 1])

        with colA:
            with st.expander("📌 Missing Values", expanded=False):
                st.write(df.isnull().sum())

            with st.expander("📊 Numeric Profile (describe + missing)", expanded=True):
                if num_prof.empty:
                    st.info("No numeric columns detected.")
                else:
                    st.dataframe(num_prof, use_container_width=True)

            with st.expander("🔤 Categorical Profile", expanded=False):
                if cat_prof.empty:
                    st.info("No categorical columns detected.")
                else:
                    st.dataframe(cat_prof, use_container_width=True)

        with colB:
            with st.expander("🔗 Correlation Pairs (sorted by |r|)", expanded=True):
                if corr_pairs.empty:
                    st.info("Not enough numeric columns for correlations.")
                else:
                    st.dataframe(
                        corr_pairs.head(50).round({"r": 3, "abs_r": 3}),
                        use_container_width=True
                    )

            # Time-series plot
            png_series: Optional[bytes] = None
            if ts_col:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    sel_cols = st.multiselect(
                        "Select series to plot over time",
                        options=num_cols,
                        default=num_cols[:plot_series_limit]
                    )
                    if sel_cols:
                        fig, ax = plt.subplots(figsize=(11, 5))
                        for c in sel_cols:
                            ax.plot(df[ts_col], df[c], label=c)
                        ax.set_title("Time-Series Trends")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value")
                        ax.legend(loc="upper right", ncol=2)
                        st.pyplot(fig)
                        png_series = fig_to_png_bytes(fig)

            # Scatter for top correlated pair
            png_scatter: Optional[bytes] = None
            if enable_scatter and not corr_pairs.empty:
                top_pair = corr_pairs.iloc[0]
                a, b = top_pair["var_a"], top_pair["var_b"]
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.scatter(df[a], df[b], alpha=0.6)
                ax2.set_title(f"Scatter: {a} vs {b} (r={top_pair['r']:.2f})")
                ax2.set_xlabel(a)
                ax2.set_ylabel(b)
                st.pyplot(fig2)
                png_scatter = fig_to_png_bytes(fig2)

        # Build insights + report
        insights = build_narrative_insights(
            df=df,
            num_prof=num_prof,
            cat_prof=cat_prof,
            corr_pairs=corr_pairs,
            ts_col=ts_col,
            insights=None,
            corr_top_k=corr_top_k,
            corr_abs_threshold=corr_abs_threshold,
            outlier_z=outlier_z
        )

        report_md = generate_markdown_report(
            df=df,
            num_prof=num_prof,
            cat_prof=cat_prof,
            corr_pairs=corr_pairs,
            ts_col=ts_col,
            insights=insights,
            corr_top_k=corr_top_k
        )

        st.subheader("📝 Auto-Generated Report")
        st.markdown(report_md)

        # Downloads
        st.markdown("### Downloads")

        st.download_button(
            label="📥 Download Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name="analysis_report.md",
            mime="text/markdown"
        )

        # Stats CSV
        if not num_prof.empty:
            stats_csv = num_prof.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download Numeric Profile (CSV)",
                data=stats_csv,
                file_name="numeric_profile.csv",
                mime="text/csv"
            )

        # ZIP of plots (if any)
        if ts_col or enable_scatter:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                if ts_col and png_series:
                    zf.writestr("time_series.png", png_series)
                if enable_scatter and png_scatter:
                    zf.writestr("top_correlation_scatter.png", png_scatter)
            zip_buf.seek(0)
            st.download_button(
                label="📦 Download Plots (ZIP)",
                data=zip_buf,
                file_name="plots.zip",
                mime="application/zip"
            )

st.markdown("---")
st.markdown("Made with ❤️ by Adewale • Streamlit MVP")
