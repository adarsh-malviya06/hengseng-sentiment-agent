# HengSeng Sentiment Price Prediction Agent

An internship assessment project for monitoring ten HKEX and KRX equities with historical market data, recent investor discussion, relevant news, multilingual sentiment, and reproducible next-session price-only estimates.

## Problem and scope

The project examines whether recent public discussion and market context can help explain short-term stock movements. It currently has about two years of daily price history and a small, recent social sample. Because there is no historical social archive aligned to the price history, model evaluation is **price-only**. Recent social sentiment is reported separately; this repository does not claim a historical sentiment-enhanced backtest or guaranteed forecasts.

## Features

- yfinance ingestion and OHLCV feature generation for ten HKEX/KRX assets.
- Apify ingestion integrations for LIHKG and Naver Finance; existing Telegram exports are normalized.
- Complete normalized social history plus a separate configurable ten-day view.
- Explicit alias matching with ambiguous multi-asset mentions left unassigned.
- Optional multilingual transformer sentiment for English, Chinese, and Korean.
- Engagement-weighted daily source sentiment.
- Exa news collection with URL-based deduplication.
- Naive, linear regression, and random forest price-only estimates, chronological holdout, and TimeSeriesSplit validation.
- Streamlit dashboard backed by generated project data.

## Architecture

`config/assets.py` holds the monitored universe. `ingestion/` downloads prices, company news, and social source samples. `processing/` normalizes social data and creates price features. `sentiment/` scores texts and aggregates daily social sentiment. `prediction/` performs chronological evaluation and creates estimates. `dashboard/app.py` reads those generated files. Raw inputs remain under `data/raw/`; derived files go under `data/processed/` and `data/predictions/`.

## Supported assets

| Exchange | Ticker | Company |
|---|---|---|
| HKEX | 0700.HK | Tencent |
| HKEX | 9988.HK | Alibaba |
| HKEX | 3690.HK | Meituan |
| HKEX | 1810.HK | Xiaomi |
| HKEX | 0005.HK | HSBC |
| KRX | 005930.KS | Samsung Electronics |
| KRX | 000660.KS | SK Hynix |
| KRX | 005380.KS | Hyundai Motor |
| KRX | 066570.KS | LG Electronics |
| KRX | 035420.KS | NAVER |

## Data sources and limitations

- **Prices:** yfinance daily history. Availability and adjusted-price behavior depend on the upstream service.
- **HKEX social:** LIHKG actor `claude_code_reviewer/lihkg-scraper-en`.
- **KRX social:** Naver actor `hgservices/naver-stock-discussion-scraper` and Telegram actor/export `k-data-lab/telegram-channel-scraper`.
- **HKEX source coverage is incomplete:** only LIHKG is currently validated. An Apify Store AAStocks scraper found during audit collects financial-news listings/headlines, not investor posts; it does not satisfy the second-social-source requirement. No AAStocks/ETNet discussion or Futu/Moomoo source has been integrated because a reliable social actor/API and permitted access have not been validated. Do not describe this as meeting the two-source HKEX requirement. See the [AASTOCKS news actor description](https://apify.com/bovi/aastocks-scraper).
- **News:** Exa SDK search results, with returned headline, highlights/summary, URL, publication date, source metadata, and search timestamp. Results depend on the Exa account and search index.
- **Social coverage:** present raw samples include 6 LIHKG, 10 Naver, and 57 Telegram records. Counts change as new records are collected. Channel subscriber count is kept distinct from author follower count.
- Collecting public site content may be subject to source terms, robots rules, privacy requirements, and retention/licensing restrictions. Review those before submission or redistribution.

## Environment setup (Windows PowerShell)

Python 3.11 is recommended. The existing `.venv` can be reused:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `APIFY_API_TOKEN` and `EXA_API_KEY` to `.env` locally. Keep `.env` private; `.gitignore` excludes it and raw/processed datasets. `.env.example` contains blank placeholders only. The sentiment model package requires PyTorch and Transformers; on first inference, Hugging Face downloads the model weights. The selected model card states support for English, Chinese, and Korean and maps five ordered classes from very negative through very positive. Its authors report synthetic multilingual training data, so review outputs critically. The model is licensed CC BY-NC 4.0; confirm that license permits the intended submission/use. See the [model card](https://huggingface.co/tabularisai/multilingual-sentiment-analysis).

## Run the pipeline

```powershell
# Existing end-to-end API ingestion (downloads prices and performs Exa news searches)
python main.py

# Process downloaded prices
python process_data.py

# Normalize every JSON file under data/raw/social; full history and separate 10-day view
python -m processing.normalize_social

# Score relevant posts (requires model dependencies and initial model download)
python -m sentiment.analyze

# Score and aggregate Exa news separately
python -c "from sentiment.analyze import run_news_sentiment; run_news_sentiment()"
python -c "from sentiment.aggregate import run_news_aggregation; run_news_aggregation()"

# Aggregate social sentiment by exchange-local calendar date and source
python -m sentiment.aggregate

# Train/evaluate price-only models and write latest estimates
python -m prediction.models

# Launch dashboard
streamlit run dashboard/app.py

# Run tests
python -m pytest -q
```

Social date window defaults to ten days and can be changed with the `SOCIAL_WINDOW_DAYS` environment variable. Cutoff uses collection time in UTC; original timestamps remain UTC. Daily grouping converts to Asia/Hong_Kong or Asia/Seoul before deriving the calendar date. Weekends/holidays are not shifted to the next trading session; daily social rows represent local calendar dates.

For a small Apify collection run, use `python test_social.py`; it consumes Apify credits and writes raw samples. Exa news ingestion also performs live searches and may incur account usage.

## Methodology

### Relevance and sentiment

Ticker and known company aliases are matched using token boundaries for ASCII names/codes and exact alias substrings for Chinese/Korean names. Multiple matches are marked ambiguous. Company-name matches are review candidates; a stock-board post carries investor-forum context but sentiment is still a language-model estimate, not a verified investor opinion. Empty and unassigned texts receive no sentiment.

The selected five-class multilingual model produces class probabilities. Positive score is `P(positive)+P(very positive)`, negative score is `P(negative)+P(very negative)`, neutral score is `P(neutral)`, and signed score is `(-2P(vn)-P(n)+P(p)+2P(vp))/2`, bounded to [-1,+1]. Confidence is the maximum class probability.

Engagement weight is `min(10, sqrt(1 + likes + dislikes + views + reposts))`, using each available count once and treating missing counts as zero. It is a volume weight only; it does not measure author credibility. Weighted sentiment is the weighted mean score. Social and news remain separate datasets.

### Price predictions and evaluation

Target is the next available price row's close. Features use current/past OHLCV, returns, rolling momentum/averages/volatility, and range/volume changes; target and next-session columns are excluded. Each ticker is sorted chronologically, the first 80% trains and final 20% is the holdout. TimeSeriesSplit is applied only on training rows. Comparisons include today's close as a naive forecast, linear regression, and random forest. Metrics are MAE, RMSE, safe MAPE (zero actuals excluded), and directional accuracy. Direction compares the signs of `predicted_close / prior_close - 1` and `actual_next_close / prior_close - 1` on identical rows; a zero-return naive forecast is only directionally correct on a flat next session. The latest model is selected by lowest mean TimeSeriesSplit MAE on training data, including the naive baseline; the holdout MAE winner is separately reported and never used to select the latest model. Latest estimates are refit on all labeled rows and marked price-only; future actual is null until observed. Forecasts outside the asset's observed one-day training return range are flagged but not clipped.

Support and resistance are observed swing lows/highs over the latest 60 sessions (three-point local extrema); levels are not arbitrary price targets. HKD and KRW prices remain in their original exchange currencies.

## Outputs

- `data/processed/social/all_social_posts.csv` — all normalized unique source records.
- `data/processed/social/recent_social_posts.csv` — only the configured UTC collection window.
- `data/processed/social/recent_social_sentiment.csv` and `daily_social_sentiment.csv` — scored recent social data and local-date aggregates.
- `data/processed/news/news_sentiment.csv` and `daily_news_sentiment.csv` — separately scored and aggregated Exa articles.
- `data/processed/all_assets_processed.csv` — combined market feature table. The latest row is kept with null next-session target.
- `data/predictions/latest_predictions.csv`, `prediction_history.csv`, `model_metrics.csv`, and `backtest_predictions.csv` — latest estimates, carried-forward forecasts with subsequently observed actuals, and chronological evaluation.
- `data/raw/company_news/` — per-asset Exa result files. No news result is fabricated if Exa is unavailable.

On the current local datasets, normalization produced 61 unique timestamp-valid social rows and the price-only prediction run produced 20 current estimates (two ML models for each of ten assets), 30 model/asset evaluation rows (naive plus both ML models), and historical holdout prediction rows. These are run outputs, not fixed benchmark claims; results change with input data and package versions.

## Quality and known limitations

- Existing raw social files are immutable inputs to normalization; old posts are not deleted when the recent window changes.
- The existing API connection script remains a manual smoke check; its helpers are not pytest-discovered, so routine tests do not consume Apify/Exa credits. Focused unit tests are under `tests/` and use mocked model inference.
- Current normalization check produced 61 unique social posts with 61 valid timestamps (6 LIHKG, 10 Naver, 45 Telegram after duplicate removal); 58 fall in the ten-day cutoff at the time of collection.
- Multilingual model weights are not bundled; first sentiment run needs a network connection and sufficient disk/RAM.
- In the existing `.venv` used for this audit, `torch`, `transformers`, `streamlit`, and `plotly` were absent. Their dependencies are listed in `requirements.txt`, but the dashboard/model runtime packages were not installed in this session; therefore real transformer inference and Streamlit startup remain unverified. The sentiment unit test uses injected probabilities and does not claim model-runtime validation.
- HKEX has only one validated social source. Existing data contains too little recent discussion for robust exchange-wide conclusions.
- Social weekend/holiday dates remain local calendar dates and are not forward-filled to trading days.
- Exa, Apify, yfinance and the sentiment model can change behavior; inspect generated timestamps, source coverage, duplicate counts, and nulls before presenting results.
- Outputs are research estimates, not investment advice.

## Submission checklist

1. Keep `.env` and all raw/private data out of Git; check `git status` and `git diff` before staging.
2. Install requirements and run tests.
3. Run price processing, social normalization, sentiment, aggregation, and prediction in order.
4. Confirm actual source coverage and timestamp counts in the generated files.
5. Launch Streamlit and verify that unavailable data appears as a warning rather than sample metrics.
6. Explain the missing second HKEX source and the price-only backtest limitation in the assessment.

Example Git commands (review the staging list before commit):

```powershell
git status --short
git add README.md .gitignore .env.example requirements.txt config ingestion processing sentiment prediction dashboard tests main.py process_data.py test_connections.py test_social.py
git diff --cached --stat
git commit -m "Complete sentiment price prediction pipeline"
```
