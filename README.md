# Quiver Strategies

Open-source trading strategy scripts built on the [Quiver Quantitative API](https://api.quiverquant.com/docs/) — congressional stock trading, SEC insider filings, government contracts, corporate lobbying, institutional holdings, and more.

Every script here is the complete, runnable code behind a tutorial on [quiverquant.com/tutorials](https://www.quiverquant.com/tutorials/). Each one pulls real data, runs a real backtest, and reports what actually happened — including the strategies that lost.

> **Not investment advice.** These are research scripts. Backtested results are historical and are not a prediction of future returns. See [Disclaimers](https://www.quiverquant.com/disclaimers/).

## Strategies

| Script | Dataset | Tutorial | Headline result |
| --- | --- | --- | --- |
| [`strategies/insider_buying_monthly.py`](strategies/insider_buying_monthly.py) | [Insider Trading](https://www.quiverquant.com/insiders/) (SEC Form 4) | [Testing Five Trading Strategies Built on Insider Buying](https://www.quiverquant.com/tutorial/insiderbuyingstrategy/) | Insider dollar value + a 200-day uptrend filter returned **+78.5%** vs **+57.3%** for buy-and-hold (Aug 2021 – Aug 2026), with a shallower drawdown |
| [`strategies/revolving_door_lobbying.py`](strategies/revolving_door_lobbying.py) | [Corporate Lobbying](https://www.quiverquant.com/lobbying/), [Government Contracts](https://www.quiverquant.com/govcontracts/), Revolving Door and Historical Daily Prices (all Hobbyist tier) | [Building a Trading Strategy on the Change in a Company's Revolving-Door Lobbyist Hires](https://www.quiverquant.com/tutorials/revolving-door-lobbying/) | Rising revolving-door hires, restricted to federal contractors, returned **+410%** vs **+192%** for buy-and-hold (Mar 2017 – Sep 2026), Sharpe 1.13; the static "most connected" ranking and the unconnected heavy-spender control both lost to the market |

## Getting started

You'll need a [Quiver API key](https://api.quiverquant.com/pricing/). Dataset access varies by plan tier, and some scripts also need a separate price data source — each script's docstring says exactly what it needs. The revolving-door script runs on a Hobbyist key alone, using Quiver's Historical Daily Prices endpoint.

```bash
git clone https://github.com/QuiverQuant/quiver-api-tutorials.git
cd quiver-api-tutorials

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export QUIVER_API_KEY="your_key_here"
export POLYGON_API_KEY="your_key_here"     # price data, only for scripts that say they need it

python strategies/insider_buying_monthly.py
python strategies/revolving_door_lobbying.py
```

Scripts that use `django-environ` will also pick the key up from a `.env` file at the repo root (`quiver_api_key=your_key_here`), which is git-ignored, so you can skip the `export` if you prefer.

New to the API? Start with the [Quiver API setup guide](https://www.quiverquant.com/api-setup/), which walks through getting a key and making your first request.

## What each script does

Scripts are self-contained and run top to bottom. They share a common shape:

1. **Backfill** the dataset from the Quiver API, with retries and on-disk checkpointing so a long pull can be resumed.
2. **Clean and scope** the data — filter to the transactions that actually carry signal, and intersect with a point-in-time index membership list to avoid survivorship bias.
3. **Rank and select** holdings on a fixed schedule, using only information that was public at the time.
4. **Backtest** with [vectorbt](https://vectorbt.dev/), applying an execution lag so the results aren't look-ahead biased.
5. **Report** a scorecard CSV, per-rebalance holdings history, and equity-curve charts against a buy-and-hold benchmark.

## Conventions we hold to

These matter more than any individual strategy, and they're the reason the numbers in the tutorials are worth reading:

- **Point-in-time universes.** Index membership is looked up as of each rebalance date, never today's constituent list applied retroactively.
- **Public-at-the-time data only.** Rankings use filing dates, not transaction dates — insiders get up to two business days to file.
- **Execution lag.** Signals are acted on a period after they're generated, never at the close of the bar that produced them.
- **An honest benchmark.** The strategy and its buy-and-hold benchmark start on the same day. Sliding the start date is the easiest way to fake outperformance.
- **Pre-registered hypotheses.** Where a script tests several variants, each one states what it expects before results are in, and anything designed after seeing results is labeled post-hoc.
- **Losing results published.** Strategies that underperformed stay in the repo and in the tutorial.

## Contributing

Issues and pull requests are welcome — a bug in a backtest is worth reporting, and a variant we didn't think to test is worth adding. If you build something interesting on Quiver data, we'd like to hear about it: [info@quiverquant.com](mailto:info@quiverquant.com).

## Links

- [Quiver Tutorials](https://www.quiverquant.com/tutorials/) — all guides, for the site, the API, and MCP
- [API documentation](https://api.quiverquant.com/docs/) — endpoint reference
- [`quiverquant` on PyPI](https://pypi.org/project/quiverquant/) — official Python client
- [Quiver MCP server](https://www.quiverquant.com/mcp-setup/) — query Quiver data from Claude and other AI tools

## License

[MIT](LICENSE)
