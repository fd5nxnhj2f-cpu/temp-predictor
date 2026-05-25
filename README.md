# 🌡️ Next-Day High Temp Prediction Engine

A Gradient Boosting Model (MOS technique) that predicts next-day high temperatures for major US cities, backtested against real historical data from Open-Meteo.

## Deploy to Vercel (Step-by-Step)

### Step 1 — Push to GitHub

1. Create a new repository on github.com (call it `temp-predictor`)
2. Upload all these files, keeping the folder structure:
   ```
   temp-predictor/
   ├── package.json
   ├── public/
   │   └── index.html
   └── src/
       ├── index.js
       └── App.js
   ```

### Step 2 — Deploy on Vercel

1. Go to [vercel.com](https://vercel.com) and sign up with your GitHub account
2. Click **"New Project"**
3. Select your `temp-predictor` repository
4. Leave all settings as default — Vercel auto-detects Create React App
5. Click **"Deploy"**
6. In ~2 minutes you'll get a live URL like `temp-predictor.vercel.app`

### Step 3 — Use the App

1. Open your Vercel URL in any browser (Safari, Chrome, etc.)
2. Select a city and training window
3. Click **"Run Backtest"** — it fetches real data from Open-Meteo and trains the model live

## How It Works

- **Data**: Open-Meteo ERA5 reanalysis archive (free, no API key needed)
- **Model**: Gradient Boosting Machine (GBM) built from scratch in JavaScript
- **Features**: Day-of-year seasonality, lag temperatures, rolling averages, wind, precipitation
- **Backtest**: 75% train / 25% chronological holdout, compared vs persistence baseline
- **Metrics**: MAE, RMSE, Bias, % improvement over baseline
