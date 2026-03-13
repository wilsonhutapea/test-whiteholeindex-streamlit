"""
export_whi.py — Standalone WHI Calculator & CSV Exporter
=========================================================
Calculates the Whitehole Index (WHI) for the date range 2025-01-20 to 2026-01-29
and writes the results to 'whi_output.csv'.

Columns: Date, WHI_value (raw ratio), WHI_X800 (scaled), IHSG_Close

Usage:
    python export_whi.py
"""

import datetime
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE = datetime.date(2025, 1, 20)
END_DATE   = datetime.date(2026, 1, 29)
OUTPUT_CSV = "whi_output.csv"
SCALE      = 800.0

IHSG_TICKER = "^JKSE"

STOCK_TICKERS: list[str] = [
    # Banking & Financial Services
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
    "BBTN.JK", "BNGA.JK", "BDMN.JK", "BNII.JK", "BTPS.JK",
    "NISP.JK", "MEGA.JK", "PNBN.JK", "BJTM.JK", "BJBR.JK",
    # Tobacco & Consumer
    "HMSP.JK", "GGRM.JK", "UNVR.JK", "ICBP.JK", "INDF.JK",
    "KLBF.JK", "SIDO.JK", "MYOR.JK", "CPIN.JK", "JPFA.JK",
    # Telco & Tech
    "TLKM.JK", "EXCL.JK", "ISAT.JK", "TOWR.JK", "TBIG.JK",
    "MTEL.JK", "GOTO.JK", "BUKA.JK", "EMTK.JK", "SCMA.JK",
    # Mining & Energy
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "INDY.JK", "BUMI.JK",
    "ANTM.JK", "INCO.JK", "TINS.JK", "MDKA.JK", "BRMS.JK",
    "MEDC.JK", "PGAS.JK", "AKRA.JK", "ELSA.JK", "ESSA.JK",
    # Automotive & Industrial
    "ASII.JK", "UNTR.JK", "AUTO.JK", "SMSM.JK", "IMAS.JK",
    "GJTL.JK", "INDS.JK",
    # Property & Construction
    "BSDE.JK", "CTRA.JK", "SMRA.JK", "PWON.JK", "DILD.JK",
    "LPKR.JK", "WIKA.JK", "WSKT.JK", "PTPP.JK", "ADHI.JK",
    "JSMR.JK",
    # Cement & Basic Materials
    "SMGR.JK", "INTP.JK", "SMCB.JK", "WTON.JK", "TPIA.JK",
    "BRPT.JK", "INKP.JK", "TKIM.JK",
    # Plantation & Agriculture
    "AALI.JK", "LSIP.JK", "DSNG.JK", "SIMP.JK", "SGRO.JK",
    # Retail & Trade
    "ACES.JK", "MAPI.JK", "LPPF.JK", "RALS.JK", "ERAA.JK",
    "AMRT.JK",
    # Healthcare & Pharma
    "HEAL.JK", "MIKA.JK", "SILO.JK", "DVLA.JK", "KAEF.JK",
    # Logistics & Transport
    "BIRD.JK", "ASSA.JK", "TMAS.JK", "SMDR.JK",
    # Others / Diversified
    "ARTO.JK", "BBYB.JK", "HRUM.JK", "DSSA.JK", "MPMX.JK",
    "SRTG.JK", "AMMN.JK", "MBMA.JK", "TAPG.JK", "CUAN.JK",
    "PGEO.JK",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_ihsg(start: datetime.date, end: datetime.date) -> pd.Series:
    """Download IHSG (^JKSE) Close prices."""
    # Extend start slightly to ensure enough warm-up data
    df = yf.download(IHSG_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Failed to download IHSG data.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"]


def fetch_stock_closes(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Download Close prices for all stock tickers."""
    # Download extra warm-up data (3 MA + 5 rolling = 8 extra trading days; use 20 to be safe)
    warmup_start = start - datetime.timedelta(days=30)
    df = yf.download(tickers, start=warmup_start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Failed to download stock data.")
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"] if "Close" in df.columns.get_level_values(0) else df.xs("Close", axis=1, level=0)
    return df.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------
# WHI calculation
# ---------------------------------------------------------------------------
def calculate_whi(closes: pd.DataFrame, scale: float = SCALE) -> pd.DataFrame:
    """
    Compute WHI for each date in the closes DataFrame.

    Formula (from researcher):
        WHI = Σ 1(P_t > max_{j=1..5} MA_3(t-j))
            / Σ 1(P_t < min_{j=1..5} MA_3(t-j))

    Steps:
        A: MA3  = Close.rolling(3).mean()
        B: Max_MA3_5d = max of MA3 shifted by 1..5 days
        C: Min_MA3_5d = min of MA3 shifted by 1..5 days
        D: numerator   = count(P_t > Max_MA3_5d)   [bullish / breakout]
        E: denominator = count(P_t < Min_MA3_5d)   [bearish / panic]
        F: raw_whi  = numerator / denominator
        G: whi_x800 = raw_whi * scale

    Returns a DataFrame with columns: ['WHI_value', 'WHI_X800']
    """
    ma3          = closes.rolling(window=3).mean()
    max_ma3_5d   = ma3.shift(1).rolling(window=5).max()
    min_ma3_5d   = ma3.shift(1).rolling(window=5).min()

    # Compare Close prices (P_t) against max/min of MA3
    # WHI spikes during crashes: count stocks BELOW floor / count stocks ABOVE ceiling
    numerator    = (closes < min_ma3_5d).sum(axis=1)
    denominator  = (closes > max_ma3_5d).sum(axis=1)

    epsilon      = 1e-9
    raw_whi      = numerator / (denominator + epsilon)
    raw_whi[denominator == 0] = 0.0

    whi_x800     = raw_whi * scale

    return pd.DataFrame({"WHI_value": raw_whi, "WHI_X800": whi_x800, "Num": numerator, "Denom": denominator})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Fetching IHSG data ({START_DATE} → {END_DATE})…")
    ihsg_close = fetch_ihsg(START_DATE, END_DATE)

    print(f"Fetching Close prices for {len(STOCK_TICKERS)} stocks…")
    closes = fetch_stock_closes(STOCK_TICKERS, START_DATE, END_DATE)
    print(f"  → {closes.shape[1]} tickers successfully downloaded.")

    print("Calculating WHI…")
    whi_df = calculate_whi(closes)

    # Trim to the requested date range (warm-up rows are excluded)
    mask   = (whi_df.index >= pd.Timestamp(START_DATE)) & (whi_df.index <= pd.Timestamp(END_DATE))
    whi_df = whi_df.loc[mask]

    # Align IHSG to the same dates
    ihsg_aligned = ihsg_close.reindex(whi_df.index)

    # Build output DataFrame
    output = pd.DataFrame({
        "Date":       whi_df.index.strftime("%Y-%m-%d"),
        "WHI_value":  whi_df["WHI_value"].round(6),
        "WHI_X800":   whi_df["WHI_X800"].round(4),
        "IHSG_Close": ihsg_aligned.round(2).values,
    })

    # Drop rows where WHI could not be calculated (early warm-up NaNs)
    output = output.dropna(subset=["WHI_value", "IHSG_Close"])

    output.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! {len(output)} rows written to '{OUTPUT_CSV}'.")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
