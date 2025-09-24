# 🧭 Diagnostic Data Copilot

**Diagnostic Data Copilot** is a **Swiss-Army Knife for dataset diagnostics**.
Upload any dataset (CSV or Excel), and the app automatically generates:

* ✅ **Data profiling** (numeric, categorical, missing values, duplicates)
* ✅ **Correlation analysis** (top drivers, strongest relationships)
* ✅ **Time-series diagnostics** (trend, seasonality, autocorrelation, cross-correlation lags)
* ✅ **Anomaly detection** (z-score outliers, change-points, Isolation Forest multivariate anomalies)
* ✅ **Structured JSON summary** (machine-readable insights for integration into other apps/APIs)
* ✅ **LLM narrative report** (executive-friendly diagnostics with causes + recommended actions, via OpenAI API)
* ✅ **One-click downloads** (Markdown report, CSV stats, ZIP of diagnostic plots)

This project is designed as both a **self-contained Streamlit app** and a **blueprint for scaling into SaaS** (with React/Next.js or API backends).

---

## 🚀 Features

### 1. Data Profiling

* Numeric stats: mean, std, min, max, percentiles (1%, 5%, 95%, 99%).
* Missing value counts and percentages.
* Categorical profile: unique counts, missing, top values.

### 2. Correlation Explorer

* Pairwise correlations across numeric variables.
* Sorted by absolute value of `r`.
* Optional scatterplot for the strongest pair.

### 3. Time-Series Diagnostics

* Auto-detects timestamp column (or manual override).
* Decomposition into **Trend, Seasonality, Residuals** using `statsmodels`.
* Autocorrelation Function (ACF) plots for periodicity.
* Cross-correlation lag analysis between a **target series** and other drivers.

### 4. Anomaly Detection

* **Z-score method**: highlights univariate outliers.
* **Change-point detection** (via `ruptures`): detects shifts in mean/variance.
* **Isolation Forest**: multivariate anomaly detection across all numeric columns.

### 5. Structured Findings

* Generates a **JSON object** capturing:

  * Dataset shape, columns, timestamp info
  * Profiles & correlations
  * Time-series diagnostics
  * Lag relationships
  * Anomalies summary

This JSON is consumable by APIs, dashboards, or other machine pipelines.

### 6. LLM-Powered Narrative

* Optional integration with **OpenAI API** (e.g. `gpt-4o-mini`).
* Converts structured findings into an **executive-friendly diagnostic report**.
* Explains:

  * Observed behaviours (trends, oscillations, cycles)
  * Likely drivers (lag/correlation evidence)
  * Recommended actions (process stability, cost/CO₂, data quality)
  * Risks and assumptions

### 7. Outputs & Downloads

* **Interactive dashboard** (Streamlit)
* **Markdown Report** (ready for sharing)
* **CSV of numeric profile**
* **ZIP archive of plots** (time-series, decomposition, ACF)

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/diagnostic-data-copilot.git
cd diagnostic-data-copilot
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

---

## 🔑 OpenAI API (Optional)

To enable **AI-driven diagnostic narratives**:

1. Get an API key from [OpenAI](https://platform.openai.com/).

2. Add it to your Streamlit secrets (recommended for Streamlit Cloud):

   `.streamlit/secrets.toml`

   ```toml
   OPENAI_API_KEY = "sk-xxxxx"
   ```

   Or paste it into the **sidebar field** when running locally.

3. Toggle **Enable LLM narrative** in the sidebar.

---

## 🌐 Deployment (Streamlit Cloud)

1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Select your repo → `main` branch → `app.py`.
4. Deploy and get a public URL like:

   ```
   https://<your-username>-diagnostic-data-copilot.streamlit.app
   ```

---

## 📊 Example Workflow

1. Upload a **cement plant dataset** with columns:

   * `timestamp`, `kiln_temp`, `fuel_rate`, `feed_rate`, `co2_emissions`.
2. The app:

   * Detects timestamp & target series.
   * Decomposes kiln temperature into **trend + cycles**.
   * Flags strong correlation (`r=0.73`) between fuel rate and kiln temperature.
   * Identifies lag relationship (fuel rate changes lead kiln temp by \~3 mins).
   * Detects change-points during kiln instability events.
   * Isolation Forest highlights 2.5% anomalous points.
3. Narrative output:

   > *“Kiln temperature oscillates with a 2-hour cycle, likely driven by fluctuations in fuel rate. Change-points align with shifts in feed rate. Recommendation: apply damping logic on fuel feed adjustments to reduce oscillations, stabilising clinker quality and cutting CO₂ emissions.”*

---

## 📂 Project Structure

```
diagnostic-data-copilot/
 ├── app.py              # Main Streamlit app
 ├── requirements.txt    # Dependencies
 └── README.md           # Documentation
```

---

## 🔧 Tech Stack

* **Streamlit** → interactive dashboard
* **Pandas / NumPy** → data wrangling
* **Matplotlib** → charts & plots
* **Statsmodels** → time-series decomposition & ACF
* **Scikit-learn** → Isolation Forest anomaly detection
* **Ruptures** → change-point detection
* **OpenAI API** → (optional) AI-generated narrative insights

---

## 📈 Roadmap

* [ ] Add sampling mode for very large datasets (>100 MB).
* [ ] Build FastAPI backend for API-first deployment.
* [ ] Expose structured findings as a REST endpoint.
* [ ] Develop React/Next.js frontend.
* [ ] Integrate domain-specific AI fine-tuning (cement/steel/energy datasets).

---

## 🛡️ License

This project is distributed under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## 👤 Author

Developed by **Adewale**
Control Engineer & AI/Tech Developer
Focused on **sustainable product development, energy, IoT, and AI solutions**.
