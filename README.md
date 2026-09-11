# Trading Bot

A collection of experiments in ML/LLM-driven trading strategies for forex and crypto. This repo is a workbench, not a single shipped product — it holds several parallel attempts at the same problem (predict price direction / generate trade signals) built at different times with different approaches. Nothing here is production-ready or currently functional end-to-end.

## Status

⚠️ Experimental / research code. Models are not currently profitable per the criteria in [`NOTES_dataset_and_model_criteria.md`](NOTES_dataset_and_model_criteria.md), and folders overlap heavily (there are three near-duplicate copies of the same LSTM pipeline). Treat this as a set of prototypes to mine ideas from, not a working bot to point at a live account.

## Repo layout

| Folder | Approach | Status |
|---|---|---|
| [`Complex model - not functional/`](Complex%20model%20-%20not%20functional/) | LSTM + attention layer, multi-feature pipeline with news sentiment folded into the model input | Not functional (kept for reference/parts) |
| [`Simplified Trading bot/`](Simplified%20Trading%20bot/) | Simplified LSTM classifier predicting price **direction** (up/down) rather than raw price, with a separate news-sentiment scraper used post-prediction | Most actively developed |
| [`Trading_bot_copy/`](Trading_bot_copy/) | Working copy / fork of `Simplified Trading bot` used for parameter experiments (different `epochs`, `lookback_years`, `window_size`, etc.) | Experimental branch of the Simplified bot |
| [`Tradegpt/`](Tradegpt/) | Different architecture entirely: instead of training a model on time series, it prompts an LLM (OpenAI / Gemini / DeepSeek) with recent price data, a hand-written strategy (RSI + EMA + sentiment), and asks it to reason about the trade | Prototype |
| `NOTES_dataset_and_model_criteria.md` | Root-level notes: dataset column definitions and what counts as a "good" model (accuracy/precision/recall thresholds) | Reference notes |
| `ideas.md` | Scratchpad of TODOs and open questions for the modeling approach | Reference notes |
| `Algorithmic_Trading_Machine_Learning_Quant_Strategies.ipynb` | Notebook following an external quant-strategies tutorial (see git history) | Learning material |
| `requirements.txt` | Combined Python dependencies across all sub-projects | — |

Each subfolder that had its own `Readme.md` has that file renamed to `NOTES_*.md` (dataset column glossaries, architecture notes) so it doesn't collide with this top-level README — see the table above for links.

## How the pieces fit together

Most sub-projects (`Simplified Trading bot`, `Trading_bot_copy`, `Tradegpt`, and the complex model) share the same general pipeline shape:

1. **`data_fetcher.py`** — pulls OHLCV market data (via `yfinance`/Alpha Vantage/Twelve Data) for one or more symbols (forex pairs or crypto pairs).
2. **`data_pipeline.py` / `feature_Engine.py`** — engineers technical-indicator features (RSI, MACD, ATR, Bollinger Bands, OBV, CMF, KST, volatility, etc.) and builds classification targets (trend direction, reversal, volatility band).
3. **`news_scraper.py` / `news_processor.py` / `News_analysis.py`** — fetches recent news for the asset and scores sentiment (used as an input feature in the LSTM approach, or as a post-hoc adjustment in the GPT approach — see `ideas.md` for why sentiment was moved *out* of the trained features).
4. **`model_builder*.py`** — builds and trains the model:
   - LSTM variants (`Complex model`, `Simplified Trading bot`, `Trading_bot_copy`) predict direction as a classification problem, using Keras/TensorFlow, with trained weights checkpointed under each project's `models/` folder.
   - `Tradegpt/GPT.py` skips training entirely and instead feeds recent data + strategy rules + sentiment into an LLM prompt to get a reasoned buy/sell/hold call.
5. **Backtesting** — `Tradegpt/Backtest.py` runs a simple RSI+EMA+sentiment signal backtest with `matplotlib` visualization; the LSTM projects evaluate via validation accuracy/precision/recall (thresholds documented in `NOTES_dataset_and_model_criteria.md`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with the API keys the sub-projects expect (see each `config.py`):

```
ALPHA_VANTAGE_KEY=
TWELVEDATA_KEY=
NEWSAPI_KEY=
ALPACA_KEY_ID=
ALPACA_SECRET_KEY=
OPENAI_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
```

Not every sub-project needs every key — the LSTM projects need the market-data and news keys; `Tradegpt` additionally needs at least one of the LLM keys (OpenAI, Gemini, or DeepSeek).

## Running a sub-project

Each folder is self-contained (own `config.py`, own `data/` and `models/` output directories). `cd` into the one you want and run its entry point, e.g.:

```bash
cd "Simplified Trading bot"
python "Simplified Trading bot.py"
```

```bash
cd Tradegpt
python GPT.py
```

## Known issues / cleanup TODO

- `Complex model - not functional`, `Simplified Trading bot`, and `Trading_bot_copy` duplicate most of their pipeline code (`data_fetcher.py`, `feature_Engine.py`, `news_processor.py`, `news_scraper.py`) — these should eventually be merged into one shared package instead of copy-pasted per experiment.
- The `news_scraper.py` in each folder uses Selenium with ChromeDriver. Selenium 4.6+ includes Selenium Manager which downloads the driver automatically, but Chrome must be installed on the machine.
- See `ideas.md` for the running list of modeling TODOs (e.g. removing sentiment from training features, keeping only the best checkpoint by validation loss).
