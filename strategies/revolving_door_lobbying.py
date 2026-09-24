"""
Tutorial: Building a Trading Strategy on the Change in a Company's
Revolving-Door Lobbyist Hires
Built entirely on the Quiver Quantitative API, including daily prices. See
the accompanying tutorial for a full walkthrough of each section:
https://www.quiverquant.com/tutorial/revolving-door-lobbying/

THESIS
  Companies that employ more former Hill and agency staff as lobbyists
  (Lobbying Disclosure Act "covered position" disclosures -- the revolving
  door) should get more out of each lobbying dollar. If so, their stocks
  should outperform equally heavy lobbyists who lack those connections.

Six contestants, one shared pipeline, one round:

  1. Most Connected (level)          -- distinct revolving-door lobbyists on
                                        the roster. The static baseline.
  2. Rising Connections              -- roster count now minus the prior
                                        window. The change idea on its own.
  3. Rising Connections x Rising Spend -- avg percentile rank of the change
                                        in connections and the change in
                                        lobbying $. Both accelerating at once.
  4. Rising Connections x Spend Level  -- avg percentile rank of the change in
                                        connections and the LEVEL of lobbying
                                        $. New hires at an already-heavy lobbyist.
  5. Rising Connections, Contractors Only -- 2, restricted to companies with
                                        federal contract revenue. Connections
                                        should matter most where government
                                        is the customer.
  6. Heavy Spenders, Unconnected (control) -- top lobbying $ among companies
                                        with ZERO revolving-door hires. 3-5
                                        have to beat this, not just the market.

CHANGE IS A COUNT DELTA, NOT A PERCENTAGE. A percentage from a zero base is
undefined, and going from one lobbyist to three is "200% growth" that should
not outrank going from twenty to thirty.

Shared rules: quarterly rebalance, up to MAX_POSITIONS equal-weighted S&P 500
holdings, point-in-time index membership, a rolling LOOKBACK_QUARTERS window
compared against an equal-length prior window, and ONE QUARTER of execution
lag. Every filing is dated by when it was POSTED, and the lag comfortably
covers the LDA's ~20-day posting window.

DATA SOURCES (all Quiver API, all Tier 1 / Hobbyist)
  Lobbying        /beta/historical/lobbying/{ticker}
  Gov contracts   /beta/historical/govcontracts/{ticker}
  Revolving door  /beta/live/revolvingdoor   (date_from/date_to on the posted
                  date, 1,000 rows a page; one pass over the whole feed, then
                  scoped to the universe)
  Prices          /beta/historical/dailyprices/{ticker}   (AdjClose; one
                  5,000-row page covers the whole window per ticker)

A TICKER IS NOT A COMPANY. A symbol's price history contains every security
that ever traded under it, plus placeholder prices for some names from before
they listed. clean_prices() masks each symbol to its point-in-time S&P 500
membership window before anything is backtested, and the benchmark buys only
the members as of its first day. Skip either step and the benchmark is wrong.

READ THIS BEFORE BELIEVING ANY NUMBER BELOW
  Revolving-door postings in the API begin in January 2016. A CHANGE metric
  needs two non-overlapping windows, so the first change signal fires at the
  end of Q4 2016 and the first trade is Q1 2017. The round is scored from the
  LATEST first trade among its contestants so nobody is charged for quarters
  spent in cash. Six contestants is still a multiple-comparisons problem --
  treat the winner as a hypothesis, not a strategy.

Re-running: the four API pulls take ~40 minutes across ~770 tickers and each
is checkpointed to output/_pull_cache/<name>.pkl as it completes.
  REUSE_CACHE=1 python <this file>   skip every pull, re-run the contestants
  RESUME=1      python <this file>   load what is cached, pull the rest
  UNIVERSE_LIMIT=40 ...              debug run on 40 tickers, separate cache

Requires: requests, pandas, vectorbt, plotly<6, kaleido==0.2.1, django-environ
  pip install -r requirements.txt
Put your key in a .env file at the repo root (quiver_api_key=...) or export
QUIVER_API_KEY in your shell. Nothing else is needed -- no separate price
data subscription. Hobbyist plan is enough: https://api.quiverquant.com/pricing/
Code 50YEAR gets 50% off your first year.
"""

import os
import time
import pickle
import datetime
from pathlib import Path

import environ
import requests
import numpy as np
import pandas as pd
import vectorbt as vbt

# The key comes from a .env at the repo root (one level up from strategies/),
# or from QUIVER_API_KEY already exported in the shell. Never hardcode it.
env = environ.Env()
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    environ.Env.read_env(str(_ENV_FILE))

# --- Every file this script writes lands here; the folder is git-ignored ---
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Pull cache & checkpoints ---
CACHE_DIR = OUTPUT_DIR / "_pull_cache"
CACHE_DIR.mkdir(exist_ok=True)
REUSE_CACHE = os.environ.get("REUSE_CACHE") == "1"
RESUME = REUSE_CACHE or os.environ.get("RESUME") == "1"
if REUSE_CACHE:
    print(f"REUSE_CACHE=1: loading pulled data from {CACHE_DIR.name}/")


def load_or_pull(name, pull_fn):
    """Pickle each dataset the moment its pull finishes, so a crash forty
    minutes in loses one dataset, not four."""
    path = CACHE_DIR / f"{name}.pkl"
    if RESUME and path.exists():
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        print(f"  [{name}] loaded from cache ({path.name})")
        return obj
    if REUSE_CACHE:
        raise SystemExit(f"REUSE_CACHE=1 but {path} is missing -- run with RESUME=1 to pull it")
    obj = pull_fn()
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  [{name}] checkpointed -> output/{CACHE_DIR.name}/{path.name}")
    return obj


def out(name):
    return str(OUTPUT_DIR / name)


# =============================================================================
# 1. Quiver auth & setup
# =============================================================================
API_KEY = env("quiver_api_key", default="") or env("QUIVER_API_KEY", default="") or None
if not API_KEY:
    raise SystemExit("No Quiver API key found. Put quiver_api_key=... in .env at the repo root "
                     "or export QUIVER_API_KEY. Get one at https://api.quiverquant.com/pricing/")
BASE_URL = "https://api.quiverquant.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def get_with_retry(url, params=None, max_retries=5):
    """Returns None once retries are exhausted, not a raised exception -- a
    single persistently-failing page shouldn't crash a multi-hour backfill.
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
        if isinstance(rows, dict):
            rows = rows.get("data") or []
        if rows is None or not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return all_rows


def parse_dollar_string(series):
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def yyyymmdd(ts):
    """The dailyprices and revolvingdoor endpoints take YYYYMMDD, no dashes."""
    return pd.Timestamp(ts).strftime("%Y%m%d")


# =============================================================================
# 2. Point-in-time S&P 500 universe (avoids survivorship / look-ahead bias)
# =============================================================================
SP500_HIST_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
sp500_hist = pd.read_csv(SP500_HIST_URL, parse_dates=["date"]).sort_values("date")


def sp500_members_asof(date, hist=sp500_hist):
    row = hist[hist["date"] <= pd.Timestamp(date)].iloc[-1]
    return set(row["tickers"].split(","))


# Revolving-door postings begin in January 2016. Lobbying and prices start a
# year earlier so the lobbying baselines are already warm when the first
# connection signal can fire.
BACKTEST_START = "2015-01-01"
BACKTEST_END = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

window = sp500_hist[(sp500_hist["date"] >= BACKTEST_START) & (sp500_hist["date"] <= BACKTEST_END)]
sp500_universe = sorted(set().union(*window["tickers"].str.split(",")))
print(f"Point-in-time S&P 500 universe over the backtest window: {len(sp500_universe)} tickers")

# Debug knob: UNIVERSE_LIMIT=40 runs the whole pipeline on 40 tickers in a few
# minutes, with its own cache folder so RESUME=1 never mistakes it for real data.
_LIMIT = int(os.environ.get("UNIVERSE_LIMIT", "0") or 0)
if _LIMIT:
    sp500_universe = sp500_universe[:_LIMIT]
    CACHE_DIR = OUTPUT_DIR / f"_pull_cache_debug{_LIMIT}"
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR = OUTPUT_DIR / f"_debug{_LIMIT}"
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"UNIVERSE_LIMIT={_LIMIT}: debug run; outputs -> {OUTPUT_DIR}")


def _progress(i, n, rows, label="tickers pulled"):
    if i % 50 == 0:
        print(f"  {i}/{n} {label} ({len(rows)} records so far)")


# =============================================================================
# 3. Lobbying -- per-ticker historical, looped over the universe
# =============================================================================
def pull_lobbying():
    print("\nPulling lobbying history (per-ticker loop)...")
    rows = []
    for i, ticker in enumerate(sp500_universe):
        rows.extend(paginate(f"{BASE_URL}/beta/historical/lobbying/{ticker}"))
        time.sleep(0.1)
        _progress(i, len(sp500_universe), rows)
    df = pd.DataFrame(rows)
    print(f"  {len(df)} lobbying records pulled; columns: {df.columns.tolist()}")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Amount"] = parse_dollar_string(df["Amount"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["Amount", "Ticker"])
    return df[(df["Date"] >= BACKTEST_START) & (df["Date"] <= BACKTEST_END)]


lobbying_df = load_or_pull("lobbying", pull_lobbying)
print(f"  Lobbying: {len(lobbying_df)} rows, {lobbying_df['Date'].min().date()} .. {lobbying_df['Date'].max().date()}")


# =============================================================================
# 4. Gov contracts -- pre-aggregated QUARTERLY totals, one call per ticker
# =============================================================================
def pull_contracts():
    print("\nPulling gov contracts history (per-ticker loop)...")
    rows = []
    for i, ticker in enumerate(sp500_universe):
        got = get_with_retry(f"{BASE_URL}/beta/historical/govcontracts/{ticker}")
        if got:                                    # None (gave up) or [] (no contracts) both skip
            rows.extend(got)
        time.sleep(0.1)
        _progress(i, len(sp500_universe), rows)
    df = pd.DataFrame(rows)
    df["Amount"] = parse_dollar_string(df["Amount"])
    df["QtrEnd"] = df.apply(
        lambda r: pd.Period(year=int(r["Year"]), quarter=int(r["Qtr"]), freq="Q").end_time.normalize(), axis=1)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["Amount", "Ticker"])


contracts_df = load_or_pull("contracts", pull_contracts)
print(f"  Contracts: {len(contracts_df)} quarter-records; {contracts_df['Ticker'].nunique()} tickers with federal contract $")


# =============================================================================
# 5. Revolving door -- /beta/live/revolvingdoor, paged across the whole feed
# =============================================================================
# The endpoint is "live" in name but takes date_from/date_to on the POSTED
# date and pages newest-first at up to 1,000 rows a page. One pass over the
# whole feed (~135 pages back to 2016) is far cheaper than a per-ticker loop.
RD_PULL_FROM = pd.Timestamp(BACKTEST_START) - pd.DateOffset(months=6 * 2)    # 2 x LOOKBACK window
RD_FEED_BEGINS = "2016-01-01"   # earliest posting the endpoint serves, confirmed 2026-09-23


def pull_revolving_door():
    print(f"\nPulling revolving-door disclosures (whole feed, posted since {RD_PULL_FROM.date()})...")
    rows = paginate(f"{BASE_URL}/beta/live/revolvingdoor", {"date_from": yyyymmdd(RD_PULL_FROM)}, page_size=1000)
    df = pd.DataFrame(rows)
    print(f"  {len(df)} rows pulled; columns: {df.columns.tolist()}")
    df = df.rename(columns={"DatePosted": "dt_posted", "LobbyistID": "lobbyist_id",
                            "BioGuideID": "member_bioguide_id", "NewLobbyist": "is_new"})
    df["ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["dt_posted"] = pd.to_datetime(df["dt_posted"], errors="coerce")
    df["member_bioguide_id"] = df["member_bioguide_id"].replace("", None)
    df["has_member"] = df["member_bioguide_id"].notna()
    df["is_new"] = df["is_new"].fillna(False).astype(bool)
    df = df.dropna(subset=["dt_posted", "lobbyist_id", "ticker"])
    expected_oldest = max(RD_PULL_FROM, pd.Timestamp(RD_FEED_BEGINS))
    if df["dt_posted"].min() > expected_oldest + pd.DateOffset(months=3):
        print(f"  WARNING: oldest posting is {df['dt_posted'].min().date()}, expected ~{expected_oldest.date()} -- "
              "a page may have failed; delete output/_pull_cache/revolving_door.pkl and re-run with RESUME=1")
    return df[["ticker", "lobbyist_id", "dt_posted", "member_bioguide_id", "has_member", "is_new",
               "TickerMatchType", "TickerMatchConfidence"]]


rd_all = load_or_pull("revolving_door", pull_revolving_door)
rd_df = rd_all[rd_all["ticker"].isin(set(sp500_universe))].copy()
print(f"  Revolving door: {len(rd_all)} rows market-wide; {len(rd_df)} rows across {rd_df['ticker'].nunique()} "
      f"S&P 500 tickers, posted {rd_df['dt_posted'].min().date()} .. {rd_df['dt_posted'].max().date()}")
# Earliest posting, less a month of grace. compute_scores() refuses to call
# anything a "change" whose baseline window reaches back before this.
RD_DATA_START = rd_df["dt_posted"].min().normalize() - pd.Timedelta(days=31)


# =============================================================================
# 6. Prices -- /beta/historical/dailyprices/{ticker}, adjusted closes
# =============================================================================
def download_price_quiver(tickers, start, end, pause=0.1):
    series, empty = {}, []
    params = {"date_from": yyyymmdd(start), "date_to": yyyymmdd(end)}
    for i, ticker in enumerate(tickers):
        rows = paginate(f"{BASE_URL}/beta/historical/dailyprices/{ticker}", params, page_size=5000)
        if not rows:
            empty.append(ticker)
            continue
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        s = pd.to_numeric(df.set_index("Date")["AdjClose"], errors="coerce").dropna().sort_index()
        series[ticker] = s[~s.index.duplicated(keep="last")]
        time.sleep(pause)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} tickers pulled")
    if empty:
        print(f"  No price data for {len(empty)} tickers: {empty[:15]}")
    return pd.DataFrame(series)


def pull_prices():
    print("\nPulling daily prices (per-ticker loop)...")
    px = download_price_quiver(sp500_universe, BACKTEST_START, BACKTEST_END)
    return px.dropna(axis=1, how="all").astype(float)


price_raw = load_or_pull("prices", pull_prices)
print(f"  Prices: {price_raw.shape[1]} of {len(sp500_universe)} tickers, "
      f"{price_raw.index.min().date()} .. {price_raw.index.max().date()}")


# =============================================================================
# 6b. Clean the price panel with point-in-time membership
# =============================================================================
# A ticker symbol is not a company. Requesting a symbol's history returns
# every security that ever traded under it, and for some names placeholder
# prices from before the company listed at all (MRNA carries nine-cent prices
# in 2017). Keep a symbol's prices only while it was an S&P 500 member plus
# about two quarters after, so a position taken at the last eligible rebalance
# can still be sold at the next one; forward-fill for at most a quarter so a
# stock acquired mid-quarter is carried at its last print until it is sold.
HOLD_AFTER_EXIT_DAYS = 130    # trading days, ~two quarters
FFILL_LIMIT_DAYS = 70         # trading days, ~one quarter plus slack


def membership_panel(index, columns):
    """Boolean frame: was `column` an S&P 500 member on `index` day?"""
    hist = sp500_hist[sp500_hist["date"] <= index[-1]]
    rows = pd.concat([hist[hist["date"] < index[0]].tail(1), hist[hist["date"] >= index[0]]])
    mask = pd.DataFrame(False, index=index, columns=columns)
    boundaries = rows["date"].tolist()[1:] + [index[-1] + pd.Timedelta(days=1)]
    for (_, row), nxt in zip(rows.iterrows(), boundaries):
        members = [t for t in row["tickers"].split(",") if t in mask.columns]
        mask.loc[max(row["date"], index[0]):nxt - pd.Timedelta(days=1), members] = True
    return mask


def clean_prices(raw):
    member = membership_panel(raw.index, raw.columns)
    holdable = member.rolling(HOLD_AFTER_EXIT_DAYS, min_periods=1).max().astype(bool)   # looks back = extends forward
    cleaned = raw.where(holdable).ffill(limit=FFILL_LIMIT_DAYS)
    removed = int((raw.notna() & cleaned.isna()).sum().sum())
    print(f"  Cleaned: {removed:,} price rows fell outside any membership window and were dropped")
    jumps = np.log(cleaned).diff().abs().max()
    wild = jumps[jumps > 1.0].sort_values(ascending=False)
    if len(wild):
        print(f"  {len(wild)} tickers still show a >170% one-day move inside their membership window: "
              f"{wild.round(2).head(12).to_dict()}")
    return cleaned.dropna(axis=1, how="all")


price = clean_prices(price_raw)
print(f"  Tradable panel: {price.shape[1]} tickers x {price.shape[0]} days")


# =============================================================================
# 7. Six ways to trade connections, lobbying, and the CHANGE in both
# =============================================================================
MAX_POSITIONS = 10
LOOKBACK_QUARTERS = 2
INIT_CASH = 100_000

quarterly_index = pd.date_range(BACKTEST_START, BACKTEST_END, freq="QE")
membership_by_quarter = {q_end: sp500_members_asof(q_end) for q_end in quarterly_index}

# Annualize ratio metrics on trading days -- without this vectorbt silently
# drops Sharpe/Sortino/Calmar from stats().
vbt.settings.returns["year_freq"] = "252 days"


def _window_totals(q_end, window_start):
    """Raw per-ticker totals for one window, using only filings POSTED inside it."""
    lw = lobbying_df[(lobbying_df["Date"] >= window_start) & (lobbying_df["Date"] <= q_end)]
    rw = rd_df[(rd_df["dt_posted"] >= window_start) & (rd_df["dt_posted"] <= q_end)]
    cw = contracts_df[(contracts_df["QtrEnd"] >= window_start) & (contracts_df["QtrEnd"] <= q_end)]
    roster = rw.groupby("ticker")["lobbyist_id"].agg(set)          # who is on the payroll, de-duplicated
    return {
        "lobbying": lw.groupby("Ticker")["Amount"].sum(),
        "rd_count": roster.map(len),
        "contracts": cw.groupby("Ticker")["Amount"].sum(),
        "roster": roster,
    }


_SCORES_CACHE = {}


def compute_scores(q_end, window_start):
    """Memoized: the scores for a quarter are identical for every contestant."""
    key = (q_end, window_start)
    if key not in _SCORES_CACHE:
        _SCORES_CACHE[key] = _compute_scores(q_end, window_start)
    return _SCORES_CACHE[key]


def _compute_scores(q_end, window_start):
    """Levels for the current window, plus CHANGE vs the immediately preceding,
    equal-length, non-overlapping window. Changes are count/dollar deltas."""
    baseline_end = window_start - pd.Timedelta(days=1)
    baseline_start = window_start - pd.DateOffset(months=3 * LOOKBACK_QUARTERS)
    cur = _window_totals(q_end, window_start)
    base = _window_totals(baseline_end, baseline_start)

    # A change needs a real baseline. If the baseline window starts before the
    # revolving-door data does, "change" is just the level measured against
    # emptiness and nearly every company "rises" (302 of 321 on our first
    # run). Blank the change series for that quarter rather than trade on it.
    baseline_ok = baseline_start >= RD_DATA_START

    def delta(a, b):
        idx = a.index.union(b.index)
        return a.reindex(idx, fill_value=0.0).astype(float) - b.reindex(idx, fill_value=0.0).astype(float)

    empty = pd.Series(dtype=float)
    return {
        "levels": {"lobbying": cur["lobbying"], "rd_count": cur["rd_count"], "contracts": cur["contracts"]},
        "change": {
            "lobbying": delta(cur["lobbying"], base["lobbying"]),
            "rd_count": delta(cur["rd_count"], base["rd_count"]) if baseline_ok else empty,
        },
    }


def _pct_blend(*series):
    parts = [s.rank(pct=True) for s in series if len(s) > 0]
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1) if parts else pd.Series(dtype=float)


def rank_most_connected(s):
    return s["levels"]["rd_count"]


def rank_rising_connections(s):
    return s["change"]["rd_count"]


def rank_rising_x_rising_spend(s):
    if len(s["change"]["rd_count"]) == 0:
        return s["change"]["rd_count"]          # no connection change -> no signal, not a lobbying-only pick
    return _pct_blend(s["change"]["rd_count"], s["change"]["lobbying"])


def rank_rising_x_spend_level(s):
    if len(s["change"]["rd_count"]) == 0:
        return s["change"]["rd_count"]
    return _pct_blend(s["change"]["rd_count"], s["levels"]["lobbying"])


def rank_rising_contractors_only(s):
    d = s["change"]["rd_count"]
    contractors = set(s["levels"]["contracts"].index[s["levels"]["contracts"] > 0])
    return d[d.index.isin(contractors)]


def rank_unconnected_heavy(s):
    """CONTROL: heavy lobbying spenders with no revolving-door hires at all in
    the current window. The thesis says these should lag contestants 3-5."""
    lvl = s["levels"]["lobbying"]
    connected = set(s["levels"]["rd_count"].index[s["levels"]["rd_count"] > 0])
    return lvl[~lvl.index.isin(connected)]


VARIANTS = {
    "Most Connected (level)":                 {"rank": rank_most_connected},
    "Rising Connections":                     {"rank": rank_rising_connections},
    "Rising Connections x Rising Spend":      {"rank": rank_rising_x_rising_spend},
    "Rising Connections x Spend Level":       {"rank": rank_rising_x_spend_level},
    "Rising Connections, Contractors Only":   {"rank": rank_rising_contractors_only},
    "Heavy Spenders, Unconnected (control)":  {"rank": rank_unconnected_heavy},
}
CONTROL = "Heavy Spenders, Unconnected (control)"


# =============================================================================
# 8. Backtest: quarterly top-10, equal weight, one quarter of lag, shared cash
# =============================================================================
def build_qualifies(rank_fn):
    """Quarterly top-MAX_POSITIONS selection grid for one contestant. A
    contestant with no positive signal in a quarter holds nothing."""
    qualifies = pd.DataFrame(False, index=quarterly_index, columns=price.columns)
    for q_end in quarterly_index:
        window_start = q_end - pd.DateOffset(months=3 * LOOKBACK_QUARTERS) + pd.Timedelta(days=1)
        ranked = rank_fn(compute_scores(q_end, window_start))
        ranked = ranked[ranked > 0].sort_values(ascending=False)
        members = membership_by_quarter[q_end]
        selected = [t for t in ranked.index if t in price.columns and t in members][:MAX_POSITIONS]
        qualifies.loc[q_end, selected] = True
    return qualifies


def first_trade_day(qualifies):
    has_holdings = qualifies.sum(axis=1) > 0
    if not has_holdings.any():
        return None
    lag_pos = quarterly_index.get_loc(has_holdings[has_holdings].index[0]) + 1  # the shift(1) in run_backtest
    if lag_pos >= len(quarterly_index):
        return None
    pos = price.index.searchsorted(quarterly_index[lag_pos])
    return price.index[pos] if pos < len(price.index) else None


def compute_manual_benchmark(price_bt, init_cash=INIT_CASH):
    """Equal-weight buy-and-hold of the S&P 500 AS IT STOOD on the round's
    first day, each name carried at its exit price if it later leaves the
    index. Buying every column priced on day one instead would include
    companies that were not members yet -- look-ahead in the benchmark."""
    start = price_bt.index[0]
    members = [t for t in sp500_members_asof(start) if t in price_bt.columns and pd.notna(price_bt[t].iloc[0])]
    px = price_bt[members].ffill()
    shares = (init_cash / len(members)) / px.iloc[0]
    return (px * shares).sum(axis=1)


def run_backtest(qualifies, start):
    target_weights = pd.DataFrame(0.0, index=quarterly_index, columns=qualifies.columns)
    for q_end in quarterly_index:
        selected = qualifies.columns[qualifies.loc[q_end]]
        if len(selected) > 0:
            target_weights.loc[q_end, selected] = 1.0 / len(selected)
    target_weights = target_weights.shift(1).fillna(0.0)      # one quarter of execution lag

    size = pd.DataFrame(float("nan"), index=price.index, columns=price.columns)
    for q_end, row in target_weights.iterrows():
        pos = price.index.searchsorted(q_end)
        if pos < len(price.index):
            size.loc[price.index[pos]] = row.values

    price_bt, size_bt = price.loc[start:], size.loc[start:]
    portfolio = vbt.Portfolio.from_orders(
        price_bt, size=size_bt, size_type="targetpercent",
        init_cash=INIT_CASH, fees=0.0, slippage=0.0002,
        group_by=True, cash_sharing=True, freq="1D",
    )
    benchmark = compute_manual_benchmark(price_bt)

    # Holdings log: what was held each rebalance and what it returned by the next one
    rebalance_days = size_bt.dropna(how="all").index.tolist()
    n_live = int((size_bt.dropna(how="all") > 0).any(axis=1).sum())
    rows = []
    for i, day in enumerate(rebalance_days):
        held = size_bt.loc[day][size_bt.loc[day] > 0]
        next_day = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else price_bt.index[-1]
        for ticker, w in held.items():
            rows.append({"Rebalance Date": day.date(), "Ticker": ticker, "Weight": round(w, 4),
                         "Return": round(price_bt.loc[next_day, ticker] / price_bt.loc[day, ticker] - 1, 4)})
    holdings_history = pd.DataFrame(rows)

    q_live = qualifies.loc[qualifies.index >= pd.Timestamp(start) - pd.DateOffset(months=3)]
    quarters = q_live.index.tolist()
    overlaps = []
    for i in range(1, len(quarters)):
        prev = set(q_live.columns[q_live.loc[quarters[i - 1]]])
        curr = set(q_live.columns[q_live.loc[quarters[i]]])
        if prev:
            overlaps.append(len(prev & curr) / len(prev))
    avg_turnover = 1 - (sum(overlaps) / len(overlaps)) if overlaps else float("nan")
    held_counts = q_live.sum(axis=1)
    avg_holdings = held_counts[held_counts > 0].mean()
    return portfolio, holdings_history, avg_turnover, avg_holdings, benchmark, n_live


# =============================================================================
# 9. Run the round
# =============================================================================
# The round begins on the LATEST first trade among its contestants, so nobody
# is scored over quarters spent in cash waiting for its data to exist.
quals = {name: build_qualifies(cfg["rank"]) for name, cfg in VARIANTS.items()}
firsts = {name: first_trade_day(q) for name, q in quals.items()}
start = max(d for d in firsts.values() if d is not None)
months = (pd.Timestamp(BACKTEST_END) - start).days / 30.4
print("\n" + "!" * 78)
print(f"!! Shared start {start.date()} = latest first trade; ~{months:.0f} months / ~{months/3:.0f} quarterly rebalances.")
for n, d in firsts.items():
    print(f"!!   {n:<40} first trade {d.date() if d is not None else 'never'}")
print("!" * 78)

results, benchmark = {}, None
for name, cfg in VARIANTS.items():
    print(f"\n=== {name} ===")
    portfolio, holdings_history, avg_turnover, avg_holdings, bench, n_rebal = run_backtest(quals[name], start)
    benchmark = bench if benchmark is None else benchmark
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    holdings_history.to_csv(out(f"holdings_{slug}.csv"), index=False)
    stats = portfolio.stats()
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    results[name] = {
        "portfolio": portfolio,
        "Total Return [%]": stats["Total Return [%]"],
        "Benchmark Return [%]": benchmark_return,
        "Max Drawdown [%]": stats["Max Drawdown [%]"],
        "Sharpe Ratio": stats.get("Sharpe Ratio", float("nan")),
        "Win Rate [%]": stats["Win Rate [%]"],
        "Avg Holdings": avg_holdings,
        "Quarterly Turnover [%]": avg_turnover * 100,
        "Live Rebalances": n_rebal,
    }
    print(f"  Total return: {stats['Total Return [%]']:.1f}%  |  Benchmark: {benchmark_return:.1f}%  "
          f"|  Max DD: {stats['Max Drawdown [%]']:.1f}%  |  Sharpe: {results[name]['Sharpe Ratio']:.2f}  "
          f"|  Turnover: {avg_turnover*100:.0f}%/qtr  |  Live rebalances: {n_rebal}")

scorecard = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "portfolio"} for k, v in results.items()}).T
pd.set_option("display.width", 200)
print("\n=== SCORECARD ===")
print(scorecard.round(2).to_string())
scorecard.round(4).to_csv(out("scorecard.csv"))

# Calendar-year returns: the shape of an edge matters as much as its size
yearly = {}
for name, res in results.items():
    v = res["portfolio"].value()
    yr = v.groupby(v.index.year).last()
    prev = pd.concat([pd.Series([v.iloc[0]], index=[yr.index[0] - 1]), yr]).shift(1).loc[yr.index]
    yearly[name] = ((yr / prev - 1) * 100).round(1)
yb = benchmark.groupby(benchmark.index.year).last()
prev = pd.concat([pd.Series([benchmark.iloc[0]], index=[yb.index[0] - 1]), yb]).shift(1).loc[yb.index]
yearly["Buy & Hold (benchmark)"] = ((yb / prev - 1) * 100).round(1)
yearly = pd.DataFrame(yearly).T
print("\n=== CALENDAR-YEAR RETURNS [%] (first and last years partial) ===")
print(yearly.to_string())
yearly.to_csv(out("yearly_returns.csv"))

# The thesis, as one number: did adding connections beat lacking them?
ctrl = scorecard.loc[CONTROL, "Total Return [%]"]
print("\nTHESIS CHECK -- connected contestants vs the unconnected-heavy-spender control:")
for name in VARIANTS:
    if name != CONTROL:
        print(f"  {name:<40} {scorecard.loc[name, 'Total Return [%]'] - ctrl:+6.1f} pts vs control")
print(f"  (control returned {ctrl:.1f}%; benchmark {scorecard['Benchmark Return [%]'].iloc[0]:.1f}%)")


# =============================================================================
# 10. Chart
# =============================================================================
def plot_round(results, title, image_id, benchmark):
    """Writes {image_id}.png and {image_id}.html into OUTPUT_DIR."""
    import plotly.graph_objects as go
    palette = ["#57D7BA", "#999cde", "#f5a623", "#e05a7a", "#4fa3f7", "#c084fc"]
    fig = go.Figure()
    for (name, res), color in zip(results.items(), palette):
        value = res["portfolio"].value()
        fig.add_trace(go.Scatter(x=value.index, y=value.values, name=name, line=dict(color=color, width=1.6)))
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
    fig.write_html(out(f"{image_id}.html"))
    try:
        fig.write_image(out(f"{image_id}.png"), width=1400, height=700, scale=2)
        print(f"  Saved {image_id}.png and {image_id}.html -> output/")
    except Exception as e:
        print(f"  Saved {image_id}.html (PNG export skipped: {e})")


plot_round(results, "Six Ways to Trade Revolving-Door Connections vs. Buy & Hold",
           "equity-curves", benchmark)
print(f"\nAll outputs in {OUTPUT_DIR}")
