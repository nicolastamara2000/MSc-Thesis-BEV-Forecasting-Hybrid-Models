# Forecasting Battery Electric Vehicle (BEV) Adoption Using Hybrid Diffusion-Neural Models

**Master's Thesis — Nicolas Tamara**
**Data Science, University of Padua (Universita degli Studi di Padova)**
**April 2026**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Description](#2-data-description)
3. [Project Structure](#3-project-structure)
4. [Methodology Pipeline](#4-methodology-pipeline)
   - 4.1 [Exploratory Data Analysis (EDA)](#41-exploratory-data-analysis-eda)
   - 4.2 [Diffusion Models (Mean Trend Component)](#42-diffusion-models-mean-trend-component)
   - 4.3 [Residual Correction Models](#43-residual-correction-models)
   - 4.4 [Evaluation Framework](#44-evaluation-framework)
5. [Key Design Decisions and Rationale](#5-key-design-decisions-and-rationale)
6. [Results Summary](#6-results-summary)
7. [Reproducing the Experiments](#7-reproducing-the-experiments)

---

## 1. Overview

This thesis investigates the problem of **forecasting monthly Battery Electric Vehicle (BEV) new registrations** in two European countries — **Italy** and **Norway** — over a 24-month horizon. The two countries were chosen to represent contrasting stages of EV adoption: Italy as an **early-stage, rapidly growing market**, and Norway as a **mature, near-saturation market**.

The core contribution is a **hybrid modeling framework** that decomposes the forecasting task into two stages:

1. **A diffusion model** captures the long-run S-curve adoption trend (the "mean" component).
2. **A neural or statistical residual model** captures short-term deviations, seasonality, and shocks left unexplained by the diffusion curve.

The final forecast is the sum: **y_hat(t) = diffusion_mean(t) + residual_forecast(t)**.

This decomposition is motivated by the observation that technology adoption follows well-known diffusion dynamics (Bass, 1969), but real-world monthly data also exhibits seasonal patterns, policy-driven shocks, and other short-term fluctuations that a smooth diffusion curve cannot capture. By separating these two components, each sub-model can be tailored to what it does best.

---

## 2. Data Description

| Attribute | Italy | Norway |
|---|---|---|
| **Source file** | `BEV Registrations.xlsx` | `BEV Registrations Norway.xlsx` |
| **Columns** | `Ano` (Year), `Mes` (Month), `BEV registrations` | Same structure |
| **Time range** | January 2014 – December 2025 | January 2014 – December 2025 |
| **Number of observations** | 144 monthly records | 144 monthly records |
| **Missing values** | None | None |
| **Data type** | All integer (int64) | All integer (int64) |
| **Mean registrations** | ~2,766 per month | Higher (Norway is a more mature BEV market) |
| **Range** | 33 – 15,304 | Broader range reflecting higher penetration |

**Why these two countries?**
- **Italy** represents an early-adoption market where BEV registrations exhibit strong exponential growth with pronounced seasonal patterns, making it an ideal candidate for diffusion modeling in the "takeoff" phase.
- **Norway** is the world leader in EV adoption (over 80% market share by 2025), representing a market approaching saturation. This tests whether the models generalize across different adoption stages.

---

## 3. Project Structure

```
ThesisUnipd/
├── BEV Registrations.xlsx            # Italy monthly BEV registration data
├── BEV Registrations Norway.xlsx     # Norway monthly BEV registration data
├── EDA copy - Copy.ipynb             # Main analysis notebook (all code)
├── EDA copy.ipynb                    # Additional/backup notebook
├── Thesis_NicolasTamara_DataScience_April2026.pdf  # Written thesis document
└── README.md                         # This file
```

---

## 4. Methodology Pipeline

### 4.1 Exploratory Data Analysis (EDA)

The EDA phase establishes the statistical properties of the time series that inform all subsequent modeling decisions.

#### 4.1.1 Visual Inspection
- **Raw time series plots**: Reveal a clear upward trend in both countries, with Italy showing more explosive recent growth and Norway showing a more gradual, saturating trajectory.
- **Log-transformed series** (`log1p(BEV)`): The `log1p` transform is applied to stabilize variance. On the log scale, the Italian series becomes more linear, confirming exponential-type growth. Norway's series is also smoother.

#### 4.1.2 Seasonal Analysis
- **Box plots by month**: Reveal clear seasonality in both countries. Certain months (e.g., March, June, September, December in Italy) show systematically higher or lower registrations, likely tied to fiscal incentives, model-year cycles, and end-of-quarter effects.
- **Additive seasonal decomposition** (period = 12): Extracts trend, seasonal, and residual components. Applied on both raw and log scales.

#### 4.1.3 Stationarity Tests

| Test | Italy (raw) | Italy (log1p) | Norway (raw) | Norway (log1p) |
|---|---|---|---|---|
| **ADF** (H0: unit root) | stat=0.10, p=0.97 (non-stationary) | stat=-0.58, p=0.87 (non-stationary) | stat=0.56, p=0.99 (non-stationary) | stat=-0.59, p=0.87 (non-stationary) |
| **KPSS** (H0: stationary) | stat=1.57, p=0.01 (non-stationary) | stat=1.63, p=0.01 (non-stationary) | stat=1.57, p=0.01 (non-stationary) | stat=1.65, p=0.01 (non-stationary) |

**Conclusion**: Both series are strongly non-stationary in levels and in log, which is expected for a technology adoption process. This confirms the need for a trend model (diffusion curve) rather than a direct ARIMA on levels.

#### 4.1.4 Differencing Analysis

After first differencing of the log1p series:
- **Italy**: ADF stat=-3.56, p=0.006 → **stationary after diff(1)**
- **Norway**: ADF stat=-4.06, p=0.001 → **stationary after diff(1)**

After both diff(1) + diff(12):
- Both countries achieve even stronger stationarity (p < 0.001), confirming the presence of both a stochastic trend and seasonal unit root.

**Why this matters**: The residuals from the diffusion model should behave like a (near-)stationary process, which justifies fitting SARIMAX or neural models on them.

#### 4.1.5 Outlier Detection
- Rolling z-score method (window = 12, threshold |z| > 2.5) identifies **August 2025 for Italy** as a potential shock month (z = -2.53). This could reflect supply-chain disruptions, policy changes, or seasonal anomalies.

---

### 4.2 Diffusion Models (Mean Trend Component)

Three diffusion models are implemented, each with increasing flexibility:

#### 4.2.1 Standard Bass Model

The Bass diffusion model (Bass, 1969) describes technology adoption through the differential equation:

```
s(t) = m * (p + q * F(t-1)) * (1 - F(t-1))
```

Where:
- `m` = total market potential (ultimate number of adopters)
- `p` = coefficient of innovation (external influence)
- `q` = coefficient of imitation (internal influence / word-of-mouth)
- `F(t) = Y_cum(t) / m` = cumulative fraction adopted

**Fitting approach**: Two variants are used:
1. **Log1p fitting**: Minimizes MSE between `log1p(observed_flow)` and `log1p(predicted_flow)` — this effectively gives more weight to early periods when counts are small, producing a more balanced fit across the entire adoption curve.
2. **Raw weighted MSE fitting**: Minimizes weighted MSE on raw counts with weights `w = 1/(y + eps)` where `eps=50` — an alternative way to handle heteroscedasticity.

**Optimization**: L-BFGS-B with bounds:
- `p ∈ [1e-6, 1.0]`
- `q ∈ [1e-6, 5.0]`
- `m ∈ [cum_obs * 1.01, cum_obs * 200 + 1e6]`

**Fitted parameters** (train end = Dec 2022):

| Country | p | q | m |
|---|---|---|---|
| Italy (log1p) | 0.000118 | 0.0569 | 256,123 |
| Italy (raw weighted) | 0.000029 | 0.0782 | 256,122 |
| Norway (log1p) | 0.00121 | 0.0324 | 801,552 |
| Norway (raw weighted) | 0.000815 | 0.0370 | 801,551 |

**Interpretation**:
- Italy's very low `p` and moderate `q` indicate adoption is primarily driven by imitation (word-of-mouth, social influence) rather than external innovation — consistent with a market still in early growth.
- Norway's higher `p` reflects a more established market with broader external awareness.
- Market potentials (~256K for Italy, ~802K for Norway) reflect the model's estimate of total cumulative registrations at the training cutoff.

#### 4.2.2 Dynamic Bass Model with Logistic Market Potential

**Motivation**: The standard Bass model assumes a **constant** market potential `m`. In reality, the addressable market for EVs evolves over time as infrastructure improves, new models launch, and policies change.

The Dynamic Bass extension replaces the constant `m` with a time-varying logistic function:

```
M(t) = M_min + (M_max - M_min) / (1 + exp(-k * (t - t0)))
```

Where `M_min` is the initial market floor, `M_max` is the ultimate ceiling, `k` controls the growth rate of the market potential itself, and `t0` is the inflection point.

The flow equation becomes:

```
s(t) = (p + q * F(t-1)) * (M(t) - Y_cum(t-1))
```

**Optimization**: L-BFGS-B with 6 parameters `[p, q, M_min, M_max, k, t0]` and light regularization on `k` and `t0` to prevent extreme values.

**Fitted parameters** (Italy):
- `p = 5.2e-5`, `q = 0.080`, `M_min = 153,644`, `M_max = 305,697`, `k = 0.0037`, `t0 = 27.2`
- Market potential grows from ~226K to ~241K over the training period.

**Why logistic M(t)?** The logistic function ensures the market potential grows smoothly from a lower bound to an upper bound, which is physically interpretable: the EV market won't grow infinitely, but it does expand as conditions improve.

#### 4.2.3 Logistic Growth Model

A standard logistic S-curve is fitted on **cumulative** registrations:

```
C(t) = K / (1 + exp(-r * (t - t0)))
```

Monthly flows are obtained by differencing: `s(t) = C(t) - C(t-1)`.

**Fitted parameters**: Italy `K = 226,102`, `r = 0.095`, `t0 = 94.8`; Norway `K = 1,256,958`, `r = 0.035`, `t0 = 117.6`.

**Comparison**: The logistic model is simpler (3 parameters vs. Bass's 3 or Dynamic's 6) but lacks the innovation/imitation decomposition, making it less interpretable for adoption dynamics.

---

### 4.3 Residual Correction Models

Once a diffusion model provides the mean trend, the residuals `e(t) = y(t) - y_hat_diffusion(t)` are modeled by a second-stage model.

#### 4.3.1 SARIMAX(1,0,1)(1,0,0,12)

**Configuration**:
- Non-seasonal order: `(1, 0, 1)` — AR(1) + MA(1), no differencing (residuals are already approximately stationary)
- Seasonal order: `(1, 0, 0, 12)` — seasonal AR(1) with period 12
- No trend term (`trend="n"`)
- `enforce_stationarity=False`, `enforce_invertibility=False`

**Why (1,0,1)(1,0,0,12)?**
- The ACF/PACF analysis of the diffusion residuals showed significant autocorrelation at lag 1 and at seasonal lag 12.
- A parsimonious ARMA(1,1) captures first-order serial dependence, while SAR(1) captures the yearly seasonal pattern.
- No differencing is needed because the diffusion model has already removed the trend, making the residuals near-stationary.

#### 4.3.2 LSTM Residual Model

**Architecture**:
- Input: lookback window of 24 past residuals, each as a 1-dimensional feature
- Month embedding: `nn.Embedding(12, 8)` — maps month index (0–11) to an 8-dimensional learned vector
- LSTM: `hidden_size=64`, single layer, `batch_first=True`
- Head: `Linear(64+8, 64) → ReLU → Linear(64, 1)`

**Why LSTM?** LSTMs are well-suited for sequential data with long-range dependencies. The month embedding provides explicit calendar information, allowing the network to learn month-specific seasonal adjustments.

**Training**: MSE loss, AdamW optimizer, gradient clipping at 1.0, 300 epochs, batch size 32, learning rate 1e-3.

**Forecast mode**: Autoregressive — each predicted residual is appended to the history for the next step.

#### 4.3.3 TCN (Temporal Convolutional Network) Residual Model

**Architecture**:
- **4 causal convolutional blocks** with dilations `[1, 2, 4, 8]`, kernel size 3, hidden channels 32
- Effective receptive field: `(3-1) * (1+2+4+8) = 30` time steps — nearly the full lookback window of 36
- Each block: causal padding → Conv1d → ReLU → Dropout(0.15) → residual connection
- Additional features: lag-12 residual + month sin/cos encoding (3 extra features)
- Head: `Linear(32+3, 64) → ReLU → Linear(64, 1)`

**Why TCN over LSTM?**
- TCNs can process the entire input window in parallel (no sequential bottleneck), making training more efficient.
- Dilated causal convolutions provide exponentially growing receptive fields, capturing both short-term and seasonal patterns.
- TCNs have been shown to match or outperform LSTMs on many sequence modeling benchmarks while being simpler to train.

**Training**: MSE loss, AdamW with weight decay 1e-4, gradient clipping at 1.0, 500–600 epochs, batch size 16, learning rate 1e-3.

**Forecast mode**: Autoregressive (same as LSTM).

#### 4.3.4 Multi-Horizon TCN (MH-TCN)

**Architecture** — the key innovation of this work:
- **Encoder**: Same TCN backbone (4 levels, dilation 2^i, kernel 3, hidden 32) processes the lookback window of 36 residuals
- **Decoder**: The final TCN hidden state (context vector) is replicated H=24 times and concatenated with future month features (sin/cos) for each horizon step
- **Per-step head**: `Linear(32+2, 64) → ReLU → Linear(64, 1)` applied independently to each of the 24 horizon steps
- Output: **all 24 future residuals predicted simultaneously** (direct multi-step forecasting)

**Why Multi-Horizon (direct) instead of autoregressive?**
- **Error accumulation**: Autoregressive models feed predictions back as inputs, so errors compound over long horizons (24 months). The Multi-Horizon approach avoids this entirely.
- **Coherent forecasting**: The model sees all future time steps during training, learning to produce a coherent trajectory rather than myopically optimizing one step at a time.
- **Calendar-aware**: Future month sin/cos features allow the model to explicitly account for seasonality at each forecast step without needing intermediate predictions.

**Training**: MSE loss, AdamW with weight decay 1e-4, 800 epochs, batch size 16, learning rate 1e-3, seed fixed at 42 for reproducibility.

**Dataset construction**: A sliding window produces training samples where each input is a window of L=36 past residuals and each target is the next H=24 residuals. This means fewer training samples than the autoregressive setup (since we need L+H consecutive points per sample), but each sample teaches the model to predict the full horizon.

---

### 4.4 Evaluation Framework

#### Train/Test Split
- **Training**: January 2014 – December 2022 (108 months)
- **Test**: January 2023 – December 2024 (24 months)
- This is a strict temporal split with no leakage.

#### Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **MAE** | `mean(|y - y_hat|)` | Average absolute error; interpretable in the same units as the data |
| **RMSE** | `sqrt(mean((y - y_hat)^2))` | Penalizes large errors more heavily than MAE |
| **sMAPE** | `100 * mean(2*|y - y_hat| / (|y| + |y_hat| + eps))` | Scale-independent percentage error; symmetric to avoid bias |

#### Scale Variants
All models are evaluated in **two scales**:
1. **Log1p scale**: Diffusion fitted on `log1p(flow)`, residuals and evaluation on `log1p` — emphasizes relative accuracy across all magnitudes.
2. **Raw scale**: Diffusion fitted on raw counts, residuals and evaluation on raw counts — directly interpretable but dominated by recent large values.

---

## 5. Key Design Decisions and Rationale

### 5.1 Why Hybrid Diffusion + Residual (Not Pure ML)?

A purely data-driven model (e.g., LSTM on raw series) would need to learn both the long-run S-curve trend and short-term patterns from data alone. With only ~108 training observations, this is extremely challenging. The hybrid approach:
- **Injects domain knowledge** via the diffusion model (adoption follows an S-curve) — this requires only 3–6 parameters.
- **Lets the neural model focus on what it's good at**: learning complex, nonlinear short-term patterns from the residuals.
- **Reduces the risk of overfitting**: the neural model sees near-stationary residuals rather than a strongly trended series.

### 5.2 Why log1p Transform?

- BEV registrations span from 33 to 34,346 (Italy), creating severe heteroscedasticity.
- The `log1p` transform stabilizes variance and makes the diffusion model fit more balanced across early (small) and late (large) values.
- The results confirm that **log-scale models consistently outperform raw-scale models** across almost all configurations.

### 5.3 Why Bass Over Other Diffusion Models?

- The Bass model is the **gold standard** for technology adoption forecasting, with extensive empirical validation across industries.
- It explicitly separates innovation (`p`) and imitation (`q`) effects, providing interpretable parameters.
- The Dynamic Bass extension addresses the main limitation (constant market potential) while keeping the model parsimonious.

### 5.4 Why SARIMAX(1,0,1)(1,0,0,12)?

- The ACF/PACF of diffusion residuals suggested AR(1) + MA(1) non-seasonal structure and AR(1) seasonal structure.
- No differencing is applied because residuals from the diffusion model are already approximately stationary.
- This is intentionally **parsimonious**: a simple baseline that captures first-order dynamics and seasonality without overfitting on ~108 points.

### 5.5 Why TCN Architecture Choices?

- **4 levels with kernel 3**: Gives a receptive field of 30, covering nearly the full lookback of 36 months.
- **Hidden channels = 32**: Small enough to avoid overfitting on ~70–80 training windows, large enough to capture patterns.
- **Dropout = 0.15**: Moderate regularization for the small dataset size.
- **Month sin/cos encoding**: A continuous representation of cyclical month information, superior to one-hot encoding for small networks.

### 5.6 Why Multi-Horizon (Direct) Forecasting?

Over a 24-month horizon, autoregressive error accumulation is severe. The Multi-Horizon approach:
- Eliminates compounding prediction errors.
- Produces the entire 24-month trajectory in a single forward pass.
- Explicitly conditions on future calendar features, giving the model access to information about which months it's predicting.

### 5.7 Why Two Countries?

Comparing Italy (early growth) and Norway (near saturation) tests whether the methodology generalizes across adoption stages. A model that only works for one stage would have limited practical value.

### 5.8 Train/Test Split at December 2022

- Provides **24 months of out-of-sample test data** (2023–2024), a substantial and practically relevant horizon.
- Leaves **108 months of training data** (2014–2022), sufficient for diffusion model fitting and neural model training.
- Avoids COVID-era data (2020–2021) being at the train/test boundary, reducing the impact of pandemic disruptions on evaluation.

---

## 6. Results Summary

### 6.1 Standard Bass + Residual Models (Italy, log1p scale)

| Model | MAE | RMSE | sMAPE |
|---|---|---|---|
| Bass mean only | 0.476 | 0.583 | 5.65% |
| Bass + SARIMAX | 0.444 | 0.542 | 5.26% |
| Bass + LSTM | 0.964 | 1.107 | 11.99% |
| Bass + TCN | 0.595 | 0.679 | 7.17% |
| **Bass + Multi-Horizon TCN** | **0.375** | **0.444** | **4.45%** |

### 6.2 Standard Bass + Residual Models (Norway, log1p scale)

| Model | MAE | RMSE | sMAPE |
|---|---|---|---|
| Bass mean only | 0.485 | 0.611 | 5.50% |
| Bass + SARIMAX | 0.352 | 0.577 | 3.96% |
| Bass + LSTM | 2.268 | 2.348 | 22.36% |
| Bass + TCN | 0.328 | 0.553 | 3.68% |
| Bass + Multi-Horizon TCN | 0.443 | 0.540 | 5.05% |

### 6.3 Dynamic Bass + Residual Models (Norway, log1p scale)

| Model | MAE | RMSE | sMAPE |
|---|---|---|---|
| DynamicBass mean only | 0.411 | 0.612 | 4.60% |
| DynamicBass + SARIMAX | 0.405 | 0.636 | 4.52% |
| DynamicBass + LSTM | 0.669 | 0.874 | 7.57% |
| DynamicBass + TCN | 0.315 | 0.528 | 3.54% |
| **DynamicBass + MH-TCN** | **0.312** | **0.479** | **3.51%** |

### 6.4 Key Findings

1. **Multi-Horizon TCN is the best residual model for Italy (log scale)**: It achieves the lowest MAE (0.375), RMSE (0.444), and sMAPE (4.45%) — a **21% improvement in MAE** over the Bass-only baseline and a **16% improvement** over Bass + SARIMAX.

2. **For Norway, the DynamicBass + MH-TCN combination is the overall winner** (sMAPE 3.51%), slightly outperforming even the autoregressive TCN variant. The Dynamic Bass model is more beneficial for Norway because its time-varying market potential better captures the saturation dynamics.

3. **Log-scale models vastly outperform raw-scale models**: Raw-scale sMAPE values often exceed 40–90%, while log-scale values stay below 5–12%. This confirms the importance of variance stabilization for this type of data.

4. **LSTM generally underperforms** compared to both SARIMAX and TCN, likely due to the small training set (~70–80 samples) and the difficulty of training recurrent networks with so few examples.

5. **Autoregressive TCN performs well for Norway but not as well for Italy**: Norway's smoother, more predictable series is more amenable to step-by-step forecasting. Italy's more volatile series suffers more from error accumulation.

6. **The Logistic growth model consistently underperforms Bass-family models**, confirming that the innovation/imitation decomposition provides meaningful structure beyond a simple S-curve.

---

## 7. Reproducing the Experiments

### Prerequisites

```
Python >= 3.9
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
torch (PyTorch)
openpyxl (for reading .xlsx files)
```

### Running

1. Place `BEV Registrations.xlsx` and `BEV Registrations Norway.xlsx` in the same directory as the notebook.
2. Open `BEV_Forecasting_Thesis.ipynb` in Jupyter Lab / VS Code.
3. Run all cells sequentially. Neural model cells may take several minutes depending on hardware.
4. For reproducibility, random seeds are set to 42 where applicable (`torch.manual_seed(42)`, `np.random.seed(42)`).

