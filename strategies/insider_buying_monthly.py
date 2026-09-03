"""
Tutorial: Testing Five Trading Strategies Built on Insider Buying
Built with the Quiver Quantitative API and Polygon.io price data. See the
accompanying tutorial for a full walkthrough of each section:
https://www.quiverquant.com/tutorial/insiderbuyingstrategy/

Runs five variants of an insider-buying strategy through one shared
pipeline and compares them head-to-head against buy-and-hold. All five
share the same rules — monthly rebalance, up to 10 equal-weighted S&P 500
holdings, a rolling 3-month ranking window, one-month execution lag,
point-in-time index membership — so the comparison isolates the selection
rule alone:

  1. Dollar Value (baseline)     — total $ insiders spent buying
  2. Cluster Buying              — number of DISTINCT insiders who bought
                                    (Lakonishok & Lee 2001)
  3. Executives Only             — dollar value, officers' purchases only
                                    (Seyhun)
  4. Stake Increase (conviction) — relative growth of insiders' existing
                                    positions; strips out size bias
  5. Dollar Value + Uptrend      — baseline, but only names above their
                                    200-day moving average; targets the
                                    "falling knife" losers seen in earlier runs

Round 1 runs the five pre-registered contestants above. Round 2 then runs
two post-hoc refinements of the Round 1 winner (the uptrend filter), which
in a live run produced the best return AND drawdown but the HIGHEST turnover:
  A. Uptrend at Entry Only    — must be above 200-day MA to be ADDED; an
                                existing holding stays while still ranked
  B. Entry-only + Sticky 20   — plus an existing holding survives while in
                                the insider top 20, not just the top 10
Round 2 was designed after seeing Round 1, so it carries more overfitting
risk — say so wherever it's discussed.

PORTFOLIO_START = "auto" starts the portfolio AND the benchmark on the first
day any variant actually trades. The insider feed's coverage begins years
after BACKTEST_START; a real run showed all variants flat at $100k from 2018
to Aug 2021 while the benchmark compounded, overstating the benchmark ~3x.

freq="1D" (plus year_freq="252 days") is required for Sharpe/Sortino/Calmar:
real trading-day data has no inferable frequency, and without it vectorbt
silently drops those metrics — confirmed by an empty Sharpe column in the
first live scorecard.

Outputs per round: a scorecard CSV, per-variant holdings-history CSVs, and
an overlaid equity-curve chart (HTML) with buy-and-hold as a dashed line.

With five contestants, the best one is partly the winner of a coin-flipping
contest — treat the winner as a hypothesis to re-test on another period or
universe, not a finished strategy.

IMPORTANT DIFFERENCES FROM OTHER QUIVER TUTORIALS IN THIS PROJECT:
- Insider Trading has NO historical or bulk endpoint, only /beta/live/insiders.
  Backfilling means looping over DATES, not tickers — ~2,000 calls for a
  multi-year backtest, meaningfully more than a ticker-based backfill.
- The date parameter for this endpoint is YYYYMMDD, not YYYY-MM-DD.
- Insider Trading is a Tier 2 dataset — requires the Trader plan ($75/mo),
  not Hobbyist.
- This endpoint is documented as "recent" data — historical depth is
  uncertain. The coverage check below is not optional; run it before
  committing to a multi-year backfill.

Requires: requests, pandas, vectorbt
  pip install requests pandas vectorbt

Set QUIVER_API_KEY and POLYGON_API_KEY in your environment before running.
Quiver: Trader plan required — see https://api.quiverquant.com/pricing/.
Code 50YEAR gets 50% off your first year.
Polygon: Stocks Starter plan (~$29/mo) recommended — see
https://massive.com/pricing (Polygon.io rebranded to Massive in 2026;
same API and keys).
"""

import os
import time
import datetime
import requests
import pandas as pd
import vectorbt as vbt

# --- Image uploads ---
# All tutorial images live in one flat S3 folder, so each tutorial gets a
# unique ID that prefixes its image filenames — this is what keeps
# insider-buying's round1-equity-curves.png from colliding with some other
# tutorial's round1-equity-curves.png in that shared folder.
TUTORIAL_ID = "insider_sep2_2026"
IMG_BASE = "https://quiver-logos.s3.us-east-2.amazonaws.com/tutorials"

# --- 1. Quiver auth & setup ---
API_KEY = os.environ["QUIVER_API_KEY"]
BASE_URL = "https://api.quiverquant.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# --- Polygon auth & setup ---
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
POLYGON_BASE_URL = "https://api.polygon.io"


def get_with_retry(url, params=None, max_retries=5):
    """Returns None once retries are exhausted, not a raised exception — a
    single persistently-failing date shouldn't crash a multi-hour backfill.
    Retries on 429 (rate limit) AND any 5xx."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  {resp.status_code} error on attempt {attempt + 1}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        print(f"Error {resp.status_code} on {url}: {resp.text[:500]}")
        resp.raise_for_status()
    print(f"  Giving up on {url} after {max_retries} retries")
    return None


def paginate(url, params=None, page_size=250):
    params = dict(params or {})
    params["page_size"] = page_size
    page, all_rows = 1, []
    while True:
        params["page"] = page
        rows = get_with_retry(url, params)
        if rows is None or not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return all_rows


def backfill_insiders(start, end, pause=0.25, checkpoint_every=200, checkpoint_path="insiders_backfill_checkpoint.csv"):
    """Loops over business days, not tickers — Insider Trading only exposes
    a live/current endpoint with a date parameter, no per-ticker history."""
    trading_days = pd.bdate_range(start, end)
    all_rows, failed_dates = [], []
    for i, day in enumerate(trading_days):
        date_str = day.strftime("%Y%m%d")  # YYYYMMDD, not YYYY-MM-DD, for this endpoint
        url = f"{BASE_URL}/beta/live/insiders"
        rows = paginate(url, {"date": date_str})
        if not rows:
            failed_dates.append(date_str)
        all_rows.extend(rows)
        time.sleep(pause)
        if i % 50 == 0:
            print(f"  {i}/{len(trading_days)} days pulled ({len(all_rows)} filings so far)")
        if i > 0 and i % checkpoint_every == 0:
            pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved to {checkpoint_path} ({len(all_rows)} rows)")
    if failed_dates:
        print(f"\nNo data returned for {len(failed_dates)} dates (holidays, gaps, or failures)")
    return all_rows


def get_polygon_with_retry(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  {resp.status_code} error on attempt {attempt + 1}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        print(f"Error {resp.status_code} on {url}: {resp.text[:500]}")
        resp.raise_for_status()
    print(f"  Giving up on {url} after {max_retries} retries")
    return None


def download_price_polygon(tickers, start, end, pause=0.25):
    series = {}
    for i, ticker in enumerate(tickers):
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
        data = get_polygon_with_retry(url, params)
        results = (data or {}).get("results") or []
        if results:
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            series[ticker] = df.set_index("date")["c"]
        time.sleep(pause)
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} tickers pulled")
    return pd.DataFrame(series)


# --- 2. Point-in-time S&P 500 universe (avoids survivorship / look-ahead bias) ---
SP500_HIST_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
sp500_hist = pd.read_csv(SP500_HIST_URL, parse_dates=["date"]).sort_values("date")


def sp500_members_asof(date, hist=sp500_hist):
    """S&P 500 tickers as of `date`, using the most recent snapshot on or
    before it — only ever reflects membership that was public knowledge
    on that date, which is what avoids look-ahead/survivorship bias."""
    row = hist[hist["date"] <= pd.Timestamp(date)].iloc[-1]
    return set(row["tickers"].split(","))


BACKTEST_START = "2018-01-01"  # shorten this if the coverage check below finds gaps
BACKTEST_END = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

# --- 3. Confirm historical depth before committing to a multi-year backfill ---
print("Checking historical coverage before backfilling...")
for test_date in ["20180102", "20200102", "20220103"]:
    rows = get_with_retry(f"{BASE_URL}/beta/live/insiders", {"date": test_date, "page_size": 5})
    print(f"  {test_date} -> {len(rows) if rows else 0} filings returned")

# --- 4. Backfill insider filings by looping over dates ---
raw_rows = backfill_insiders(BACKTEST_START, BACKTEST_END)
insiders_df = pd.DataFrame(raw_rows)
print(insiders_df.columns.tolist())  # confirm actual field names before referencing them

# --- 5. Clean ticker strings before using them anywhere downstream ---
# Quiver's Insider Trading feed occasionally has exchange-prefixed or
# parenthesized ticker strings (e.g. "(NYSE:FBC)") instead of a plain symbol
# — confirmed directly, this crashes a naive Polygon price pull with a 404
# ("Ticker prefix does not exist") since it's not a valid ticker at all.
import re

def clean_ticker(raw_ticker):
    """Strip exchange prefixes/parens down to a plain ticker, or return None
    if what's left still doesn't look like a real one."""
    if not isinstance(raw_ticker, str):
        return None
    t = raw_ticker.strip().strip("()")
    if ":" in t:
        t = t.split(":")[-1]
    t = t.strip().upper()
    return t if re.fullmatch(r"[A-Z]{1,6}(\.[A-Z])?", t) else None

insiders_df["Ticker"] = insiders_df["Ticker"].apply(clean_ticker)
insiders_df = insiders_df.dropna(subset=["Ticker"])

# --- 6. Filter to genuine open-market purchases only ---
# TransactionCode "P" is an open-market or private purchase — the SEC Form 4
# code for an insider actually buying shares with their own money. Grants
# ("A"), option exercises ("M"), tax withholding ("F"), and gifts ("G") all
# also show up as "acquisitions" in this feed but aren't a discretionary bet
# on the stock, so they're excluded here.
insiders_df["fileDate"] = pd.to_datetime(insiders_df["fileDate"])
purchases = insiders_df[insiders_df["TransactionCode"] == "P"].copy()
print(f"Genuine open-market purchases: {len(purchases)} of {len(insiders_df)} total filings")

# --- 7. Pull price data (Polygon), scoped to the S&P 500 universe FIRST ---
# The insider feed covers the whole market (thousands of tickers), not just
# the S&P 500 — pulling price data for every ticker that ever had a purchase
# means pulling for ~6,000 names instead of a few hundred, almost all of
# which this strategy will never actually trade. Intersect with the
# point-in-time S&P 500 window before downloading anything.
window = sp500_hist[(sp500_hist["date"] >= BACKTEST_START) & (sp500_hist["date"] <= BACKTEST_END)]
sp500_universe = set().union(*window["tickers"].str.split(","))

universe = sorted(set(purchases["Ticker"].dropna().unique()) & sp500_universe)
price = download_price_polygon(universe, BACKTEST_START, BACKTEST_END)
price = price.dropna(axis=1, how="all").astype(float)
print(f"Price data returned for {price.shape[1]} of {len(universe)} tickers")

# =============================================================================
# 8. Five ways to trade the same insider-buying data
# =============================================================================
# The data pipeline above is identical for every variant. What differs is
# how each one turns "insiders bought shares" into "which 10 companies to
# hold this month." Each variant tests a specific, falsifiable hypothesis
# about what makes an insider purchase informative — several drawn from the
# academic literature on insider trading, others from failure modes observed
# directly in earlier real runs of this strategy.
#
# All variants share: monthly rebalance, up to MAX_POSITIONS equal-weighted
# holdings, a rolling RANKING_LOOKBACK_MONTHS window for the ranking, a
# one-month execution lag, point-in-time S&P 500 membership, and the same
# portfolio start date. That way the comparison isolates the selection rule.

MAX_POSITIONS = 10
RANKING_LOOKBACK_MONTHS = 3
INIT_CASH = 100_000

# Portfolio start. "auto" = the first rebalance on which ANY variant actually
# takes a position, shared by all variants so the comparison is fair. The
# insider feed's coverage starts years after BACKTEST_START, and every variant
# sits in cash until real filings appear — a real run showed 2018-2021 flat at
# $100k for all five while buy-and-hold compounded the whole time. Measuring
# the benchmark from 2018 against strategies that couldn't trade until 2021
# overstated the benchmark by roughly 3x. Aligning both to the first trade
# fixes that. Override with a date string ("2023-01-01") to force a start —
# but note that starting AFTER the first trade skips real history (the 2022
# bear market, for instance), which flatters every result.
PORTFOLIO_START = "auto"

monthly_index = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME")

# Coerce the boolean role flags robustly — the API returns real booleans,
# but a CSV round-trip through the checkpoint file turns them into strings.
for col in ["isOfficer", "isDirector", "isTenPercentOwner"]:
    if col in purchases.columns:
        purchases[col] = purchases[col].astype(str).str.lower().eq("true")

purchases["Value"] = purchases["Shares"] * purchases["PricePerShare"]

# Per-purchase "stake increase": how much did this purchase grow the
# insider's EXISTING position? Capped at 1.0 (a 100% increase) so an insider
# going from ~0 shares to a real position doesn't produce an absurd ratio.
prior_holdings = (purchases["SharesOwnedFollowing"] - purchases["Shares"]).clip(lower=1)
purchases["StakeIncrease"] = (purchases["Shares"] / prior_holdings).clip(upper=1.0)

# 200-day moving average from past prices only — no look-ahead. Computed on
# the FULL price history so it's already warmed up by the portfolio start.
ma200 = price.rolling(200, min_periods=200).mean()

membership_by_month = {m_end: sp500_members_asof(m_end) for m_end in monthly_index}

# Annualize ratio metrics on trading days. Real price data has no fixed
# frequency (weekends + holidays are missing, so pandas can't infer one),
# and without an explicit freq vectorbt silently omits Sharpe, Sortino and
# Calmar from stats() — confirmed directly: the first real scorecard came
# back with an empty Sharpe column. freq="1D" below fixes that.
vbt.settings.returns["year_freq"] = "252 days"


def rank_by_dollar_value(window_purchases):
    """Baseline: total dollars insiders spent buying, per company."""
    return window_purchases.groupby("Ticker")["Value"].sum()


def rank_by_cluster_buying(window_purchases):
    """Cluster buying: how many DIFFERENT insiders bought, per company
    (Lakonishok & Lee 2001). Ties broken by dollar value."""
    distinct_buyers = window_purchases.groupby("Ticker")["Name"].nunique()
    dollar_value = window_purchases.groupby("Ticker")["Value"].sum()
    return distinct_buyers + (dollar_value / dollar_value.max()).fillna(0) * 0.5


def rank_by_executives_only(window_purchases):
    """Executives only: dollar value, counting only officers' purchases
    (Seyhun found executives' trades more informative than directors')."""
    officers = window_purchases[window_purchases["isOfficer"]]
    return officers.groupby("Ticker")["Value"].sum()


def rank_by_stake_increase(window_purchases):
    """Conviction: total relative increase in insiders' existing stakes."""
    return window_purchases.groupby("Ticker")["StakeIncrease"].sum()


# uptrend_mode: None      -> no trend filter
#               "always"  -> a company must be above its 200-day MA to be held at all
#               "entry"   -> must be above its 200-day MA to be ADDED; an existing
#                            holding stays as long as it's still ranked (no re-test)
# sticky_rank:  None      -> an existing holding must stay in the top MAX_POSITIONS
#               N         -> an existing holding stays if it's anywhere in the top N
VARIANTS = {
    "Dollar Value (baseline)":       {"rank": rank_by_dollar_value,    "uptrend_mode": None},
    "Cluster Buying":                {"rank": rank_by_cluster_buying,  "uptrend_mode": None},
    "Executives Only":               {"rank": rank_by_executives_only, "uptrend_mode": None},
    "Stake Increase (conviction)":   {"rank": rank_by_stake_increase,  "uptrend_mode": None},
    "Dollar Value + Uptrend Filter": {"rank": rank_by_dollar_value,    "uptrend_mode": "always"},
}


def build_qualifies(rank_fn, uptrend_mode=None, sticky_rank=None):
    """Monthly top-MAX_POSITIONS selection grid for one variant."""
    qualifies = pd.DataFrame(False, index=monthly_index, columns=price.columns)
    prev_held = set()
    for m_end in monthly_index:
        m_start = m_end - pd.DateOffset(months=RANKING_LOOKBACK_MONTHS) + pd.Timedelta(days=1)
        window_purchases = purchases[
            (purchases["fileDate"] >= m_start) & (purchases["fileDate"] <= m_end)
        ]
        scores = rank_fn(window_purchases).sort_values(ascending=False)
        members = membership_by_month[m_end]
        candidates = [t for t in scores.index if t in price.columns and t in members]

        in_uptrend = None
        if uptrend_mode is not None:
            pos = price.index.searchsorted(m_end, side="right") - 1
            if pos >= 0:
                asof = price.index[pos]
                in_uptrend = price.loc[asof] > ma200.loc[asof]

        if uptrend_mode == "always":
            if in_uptrend is not None:
                candidates = [t for t in candidates if bool(in_uptrend.get(t, False))]
            selected = candidates[:MAX_POSITIONS]
        elif uptrend_mode == "entry":
            keep_pool = candidates[:sticky_rank] if sticky_rank else candidates[:MAX_POSITIONS]
            kept = [t for t in keep_pool if t in prev_held]
            new = [t for t in candidates
                   if t not in kept and (in_uptrend is None or bool(in_uptrend.get(t, False)))]
            selected = (kept + new)[:MAX_POSITIONS]
        else:
            selected = candidates[:MAX_POSITIONS]

        qualifies.loc[m_end, selected] = True
        prev_held = set(selected)
    return qualifies


def first_trade_day(qualifies):
    """First trading day on which this variant actually holds something
    (after the one-month execution lag)."""
    has_holdings = qualifies.sum(axis=1) > 0
    if not has_holdings.any():
        return None
    first_signal_month = has_holdings[has_holdings].index[0]
    lag_month_pos = monthly_index.get_loc(first_signal_month) + 1  # shift(1) in run_backtest
    if lag_month_pos >= len(monthly_index):
        return None
    pos = price.index.searchsorted(monthly_index[lag_month_pos])
    return price.index[pos] if pos < len(price.index) else None


def run_backtest(qualifies, start):
    """Turn a monthly selection grid into target weights and run the backtest
    from `start`, so the strategy and the buy-and-hold benchmark begin on the
    same day. Identical for every variant."""
    target_weights = pd.DataFrame(0.0, index=monthly_index, columns=qualifies.columns)
    for m_end in monthly_index:
        selected = qualifies.columns[qualifies.loc[m_end]]
        if len(selected) > 0:
            target_weights.loc[m_end, selected] = 1.0 / len(selected)
    target_weights = target_weights.shift(1).fillna(0.0)  # one month of execution lag

    size = pd.DataFrame(float("nan"), index=price.index, columns=price.columns)
    for m_end, row in target_weights.iterrows():
        pos = price.index.searchsorted(m_end)
        if pos < len(price.index):
            size.loc[price.index[pos]] = row.values

    # Slice BOTH price and size to the common start so the benchmark
    # (buy-and-hold of this universe) is measured over the same window as
    # the strategy — not from years before the strategy could trade.
    price_bt = price.loc[start:]
    size_bt = size.loc[start:]

    # from_orders + targetpercent: each rebalance sizes off CURRENT portfolio
    # value, so gains compound. NOT size_type="percent" (sizes off remaining
    # cash ticker by ticker -> lopsided allocation; confirmed by testing).
    # fees=0.0: $0 commission is standard at every major US retail broker.
    # slippage=0.0002: 2bps, a modest stand-in for spread + impact on a
    # $10-25k trade in a liquid S&P 500 name.
    portfolio = vbt.Portfolio.from_orders(
        price_bt, size=size_bt, size_type="targetpercent",
        init_cash=INIT_CASH, fees=0.0, slippage=0.0002,
        group_by=True, cash_sharing=True, freq="1D",
    )

    # Monthly holdings snapshot — more useful than the raw trade log, which
    # fragments a continuously-held position into many partial-rebalance trades.
    rebalance_days = size_bt.dropna(how="all").index.tolist()
    rows = []
    for i, day in enumerate(rebalance_days):
        held = size_bt.loc[day][size_bt.loc[day] > 0]
        next_day = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else price_bt.index[-1]
        for ticker, w in held.items():
            rows.append({
                "Rebalance Date": day.date(), "Ticker": ticker, "Weight": round(w, 4),
                "Return": round(price_bt.loc[next_day, ticker] / price_bt.loc[day, ticker] - 1, 4),
            })
    holdings_history = pd.DataFrame(rows)

    # Month-over-month persistence of the selection, over months actually traded
    q_live = qualifies.loc[qualifies.index >= pd.Timestamp(start) - pd.DateOffset(months=1)]
    months = q_live.index.tolist()
    overlaps = []
    for i in range(1, len(months)):
        prev = set(q_live.columns[q_live.loc[months[i - 1]]])
        curr = set(q_live.columns[q_live.loc[months[i]]])
        if prev:
            overlaps.append(len(prev & curr) / len(prev))
    avg_turnover = 1 - (sum(overlaps) / len(overlaps)) if overlaps else float("nan")
    avg_holdings = q_live.sum(axis=1)[q_live.sum(axis=1) > 0].mean()

    return portfolio, holdings_history, avg_turnover, avg_holdings


def run_round(variants, start, label):
    """Run a set of variants from a common start and return results + scorecard."""
    results = {}
    for name, cfg in variants.items():
        print(f"\n=== {name} ===")
        qualifies = build_qualifies(cfg["rank"], cfg.get("uptrend_mode"), cfg.get("sticky_rank"))
        portfolio, holdings_history, avg_turnover, avg_holdings = run_backtest(qualifies, start)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace(",", "")
        holdings_history.to_csv(f"holdings_history_{slug}.csv", index=False)
        stats = portfolio.stats()
        results[name] = {
            "portfolio": portfolio,
            "Total Return [%]": stats["Total Return [%]"],
            "Benchmark Return [%]": stats["Benchmark Return [%]"],
            "Max Drawdown [%]": stats["Max Drawdown [%]"],
            "Sharpe Ratio": stats.get("Sharpe Ratio", float("nan")),
            "Win Rate [%]": stats["Win Rate [%]"],
            "Avg Holdings": avg_holdings,
            "Monthly Turnover [%]": avg_turnover * 100,
        }
        print(f"  Total return: {stats['Total Return [%]']:.1f}%  |  Benchmark: {stats['Benchmark Return [%]']:.1f}%  "
              f"|  Max DD: {stats['Max Drawdown [%]']:.1f}%  |  Sharpe: {results[name]['Sharpe Ratio']:.2f}  "
              f"|  Turnover: {avg_turnover*100:.0f}%/mo")
    scorecard = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "portfolio"} for k, v in results.items()}).T
    pd.set_option("display.width", 200)
    print(f"\n=== SCORECARD: {label} ===")
    print(scorecard.round(2).to_string())
    scorecard.round(4).to_csv(f"variant_scorecard_{label.lower().replace(' ', '_')}.csv")
    return results, scorecard


def plot_round(results, title, image_id):
    """One chart, every variant in the round, plus buy-and-hold as a dashed line.

    Writes two files named after `image_id`, which matches the tutorial's
    <img> placeholder id so uploading them is the only step left:
      {image_id}.png   — static image for the tutorial (needs `pip install kaleido`)
      {image_id}.html  — the interactive version, if you'd rather embed that
    """
    import plotly.graph_objects as go
    palette = ["#57D7BA", "#999cde", "#f5a623", "#e05a7a", "#4fa3f7", "#c084fc"]
    fig = go.Figure()
    for (name, res), color in zip(results.items(), palette):
        value = res["portfolio"].value()
        fig.add_trace(go.Scatter(x=value.index, y=value.values, name=name, line=dict(color=color, width=1.6)))
    benchmark = next(iter(results.values()))["portfolio"].benchmark_value()
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Buy & Hold (benchmark)",
                             line=dict(color="rgb(151,153,154)", width=1.2, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#121212",
        font=dict(family="Figtree, sans-serif", color="rgb(241,243,244)"),
        xaxis=dict(gridcolor="#2F3F4D", linecolor="#2F3F4D"),
        yaxis=dict(gridcolor="#2F3F4D", linecolor="#2F3F4D", title="Portfolio Value ($)"),
        title=dict(text=title, font=dict(color="rgb(251,253,254)")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.write_html(f"{image_id}.html")
    try:
        # Consistent size + 2x scale gives a crisp image that matches the
        # 750px tutorial column when displayed at half resolution.
        fig.write_image(f"{image_id}.png", width=1400, height=700, scale=2)
        print(f"  Saved {image_id}.png and {image_id}.html")
    except Exception as e:
        # kaleido (plotly's image exporter) isn't installed or can't find a
        # browser. The interactive HTML still saved; screenshot it manually
        # or run: pip install kaleido
        print(f"  Saved {image_id}.html (PNG export skipped: {e})")
    fig.show()


# =============================================================================
# 9. Determine the common portfolio start
# =============================================================================
if PORTFOLIO_START == "auto":
    starts = []
    for cfg in VARIANTS.values():
        q = build_qualifies(cfg["rank"], cfg.get("uptrend_mode"), cfg.get("sticky_rank"))
        d = first_trade_day(q)
        if d is not None:
            starts.append(d)
    portfolio_start = min(starts)
else:
    portfolio_start = pd.Timestamp(PORTFOLIO_START)
print(f"\nPortfolio start (shared by all variants and the benchmark): {portfolio_start.date()}")

# =============================================================================
# 10. Round 1 — the five pre-registered contestants
# =============================================================================
round1_results, round1_scorecard = run_round(VARIANTS, portfolio_start, "Round 1")
plot_round(round1_results, "Five Ways to Trade Insider Buying", f"{TUTORIAL_ID}_round1-equity-curves")

# =============================================================================
# 11. Round 2 — post-hoc refinements of the Round 1 winner
# =============================================================================
# These were designed AFTER seeing Round 1's results, so they carry more
# overfitting risk than the pre-registered contestants above — say so
# plainly whenever they're discussed. Both target the one weakness the
# winning variant showed in a real run: the always-on trend filter produced
# the HIGHEST turnover of any contestant (companies flip in and out as they
# cross their 200-day average), which is the same compounding-drag problem
# that hurt earlier versions of this strategy.
#
#   - "Uptrend at Entry Only": a company must be above its 200-day MA to be
#     ADDED, but an existing holding is not re-tested each month — it stays as
#     long as insiders still rank it in the top 10. Stops the filter from
#     selling a good position on a routine dip through the average.
#   - "Uptrend at Entry + Sticky Top-20": the same, plus an existing holding
#     is retained if it's anywhere in the insider top 20, not just the top 10.
#     Stops churn from marginal ranking changes right at the cutoff.
ROUND2_VARIANTS = {
    "Dollar Value + Uptrend Filter":      {"rank": rank_by_dollar_value, "uptrend_mode": "always"},
    "Uptrend at Entry Only":              {"rank": rank_by_dollar_value, "uptrend_mode": "entry"},
    "Uptrend at Entry + Sticky Top-20":   {"rank": rank_by_dollar_value, "uptrend_mode": "entry", "sticky_rank": 20},
}
round2_results, round2_scorecard = run_round(ROUND2_VARIANTS, portfolio_start, "Round 2")
plot_round(round2_results, "Round 2: Refining the Winner (post-hoc)", f"{TUTORIAL_ID}_round2-equity-curves")
