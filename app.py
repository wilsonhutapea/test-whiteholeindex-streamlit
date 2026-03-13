"""
Whitehole Index (WHI) — Market Timing Indicator for IHSG
=========================================================
A Streamlit application that calculates and visualizes the WHI,
a custom market-timing indicator based on extreme price movements
across the top 100 Indonesian stocks.

A spike in WHI above 4,000 indicates a retail flush phase, signaling
a potential price reversal or dead cat bounce.
"""

import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Whitehole Index (WHI)",
    page_icon="🕳️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Top 100 Indonesian stocks by market cap (KOMPAS100-style constituents)
# Each ticker is appended with '.JK' for Yahoo Finance.
# ---------------------------------------------------------------------------
STOCK_TICKERS: list[str] = [
    "AADI.JK", "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK",
    "AMMN.JK", "AMRT.JK", "ANTM.JK", "ARCI.JK", "ARTO.JK",
    "ASII.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK",
    "BBYB.JK", "BKSL.JK", "BMRI.JK", "BREN.JK", "BRIS.JK",
    "BRMS.JK", "BRPT.JK", "BSDE.JK", "BTPS.JK", "BUKA.JK",
    "BULL.JK", "BUMI.JK", "BUVA.JK", "CBDK.JK", "CMRY.JK",
    "CPIN.JK", "CTRA.JK", "CUAN.JK", "DEWA.JK", "DSNG.JK",
    "DSSA.JK", "ELSA.JK", "EMTK.JK", "ENRG.JK", "ERAA.JK",
    "ESSA.JK", "EXCL.JK", "FILM.JK", "GOTO.JK", "HEAL.JK",
    "HMSP.JK", "HRTA.JK", "HRUM.JK", "ICBP.JK", "IMPC.JK",
    "INCO.JK", "INDF.JK", "INDY.JK", "INET.JK", "INKP.JK",
    "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK",
    "KIJA.JK", "KLBF.JK", "KPIG.JK", "MAPA.JK", "MAPI.JK",
    "MBMA.JK", "MDKA.JK", "MEDC.JK", "MIKA.JK", "MTEL.JK",
    "MYOR.JK", "NCKL.JK", "PANI.JK", "PGAS.JK", "PGEO.JK",
    "PNLF.JK", "PSAB.JK", "PTBA.JK", "PTRO.JK", "PWON.JK",
    "RAJA.JK", "RATU.JK", "SCMA.JK", "SGER.JK", "SIDO.JK",
    "SMGR.JK", "SMIL.JK", "SMRA.JK", "SSIA.JK", "TAPG.JK",
    "TCPI.JK", "TINS.JK", "TLKM.JK", "TOBA.JK", "TOWR.JK",
    "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIFI.JK", "WIRG.JK"
]

IHSG_TICKER = "^JKSE"

# Default date range: 1 Jan 2025 → today
DEFAULT_START = datetime.date(2025, 1, 1)
DEFAULT_END = datetime.date.today()


# ---------------------------------------------------------------------------
# Data fetching (cached to avoid repeated downloads)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ihsg(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Download IHSG (^JKSE) daily OHLC data."""
    df = yf.download(IHSG_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_closes(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Download Close prices for a list of stock tickers.

    Returns a DataFrame with dates as the index and tickers as columns.
    """
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    # yfinance returns multi-level columns (Price, Ticker) for multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        # Extract only Close prices
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df = df.xs("Close", axis=1, level=0)
    return df


# ---------------------------------------------------------------------------
# WHI calculation engine
# ---------------------------------------------------------------------------
def calculate_whi(closes: pd.DataFrame, scale: float = 800.0) -> pd.Series:
    """Calculate the Whitehole Index for each trading day.

    The WHI spikes during market crashes: it counts stocks breaking
    *below* their short-term MA floor (bearish) divided by those
    breaking *above* their MA ceiling (bullish), then scales by 800.

    Parameters
    ----------
    closes : pd.DataFrame
        DataFrame of daily Close prices (index = date, columns = tickers).
    scale : float
        Multiplier applied to the raw ratio (default 800).

    Returns
    -------
    pd.Series
        Scaled WHI values indexed by date.
    """
    # Step A: 3-period moving average of Close for every stock
    ma3 = closes.rolling(window=3).mean()

    # Step B: 5-period rolling max of MA3, EXCLUDING the current day (shift 1)
    max_ma3_5d = ma3.shift(1).rolling(window=5).max()

    # Step C: 5-period rolling min of MA3, EXCLUDING the current day (shift 1)
    min_ma3_5d = ma3.shift(1).rolling(window=5).min()

    # Step D (Numerator - Bearish): Count stocks where Close < Min_MA3_5d
    numerator = (closes < min_ma3_5d).sum(axis=1)

    # Step E (Denominator - Bullish): Count stocks where Close > Max_MA3_5d
    denominator = (closes > max_ma3_5d).sum(axis=1)

    # Step F: Raw WHI = Numerator / Denominator (handle division by zero)
    epsilon = 1e-9
    raw_whi = numerator / (denominator + epsilon)
    raw_whi[denominator == 0] = 0.0

    # Step G: Scale the raw ratio to match the paper's threshold bands
    whi = raw_whi * scale

    return whi


# ---------------------------------------------------------------------------
# Plotly visualization
# ---------------------------------------------------------------------------
def build_chart(ihsg_df: pd.DataFrame, whi: pd.Series) -> go.Figure:
    """Create a dual-axis chart: IHSG line + WHI bars with threshold lines."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=("IHSG (Jakarta Composite Index)", "Whitehole Index (WHI)"),
    )

    # --- Primary subplot: IHSG Close price ---
    fig.add_trace(
        go.Scatter(
            x=ihsg_df.index,
            y=ihsg_df["Close"],
            mode="lines",
            name="IHSG Close",
            line=dict(color="black", width=1.8),
        ),
        row=1, col=1,
    )

    # --- Secondary subplot: WHI bars ---
    # Colour bars based on threshold
    colors = [
        "#D32F2F" if v >= 4000 else "#FF9800" if v >= 2000 else "#9E9E9E"
        for v in whi.values
    ]

    fig.add_trace(
        go.Bar(
            x=whi.index,
            y=whi.values,
            name="WHI",
            marker_color=colors,
            opacity=0.85,
        ),
        row=2, col=1,
    )

    # --- Threshold lines on WHI subplot ---
    fig.add_hline(
        y=4000, line_dash="dash", line_color="red", line_width=1.5,
        annotation_text="Band 4000 (Extreme Panic)",
        annotation_position="top left",
        annotation_font_color="red",
        row=2, col=1,
    )
    fig.add_hline(
        y=2000, line_dash="dash", line_color="orange", line_width=1.5,
        annotation_text="Band 2000",
        annotation_position="top left",
        annotation_font_color="orange",
        row=2, col=1,
    )

    # --- Layout polish ---
    fig.update_layout(
        height=720,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
        bargap=0.15,
    )
    fig.update_yaxes(title_text="IHSG Close", row=1, col=1)
    fig.update_yaxes(title_text="WHI Value", row=2, col=1)
    fig.update_xaxes(
        title_text="Date",
        row=2, col=1,
        tickformat="%d %b %Y",
        rangeslider=dict(visible=False),
    )

    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Whitehole Index (WHI) — Market Timing Indicator for IHSG")

    st.markdown(
        """
        > **A spike in WHI above 4,000 indicates a *retail flush* phase,
        > signaling a potential price reversal.**
        >
        > The WHI measures the ratio of stocks breaking *below* their
        > short-term moving-average floor versus those breaking *above*
        > their ceiling, scaled by x800. A high value means far more
        > stocks are capitulating than advancing — a classic panic signal.
        """
    )

    # ---- Sidebar: date range picker ----
    st.sidebar.header("📅 Date Range")
    start_date = st.sidebar.date_input("Start date", value=DEFAULT_START)
    end_date = st.sidebar.date_input("End date", value=DEFAULT_END)

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Tracking **{len(STOCK_TICKERS)}** Indonesian stocks "
        f"against **IHSG** (`{IHSG_TICKER}`)."
    )

    # ---- Fetch data ----
    with st.spinner("Fetching IHSG data from Yahoo Finance…"):
        ihsg_df = fetch_ihsg(start_date, end_date)

    if ihsg_df.empty:
        st.error("Could not fetch IHSG data. Please check your internet connection or try again later.")
        st.stop()

    with st.spinner("Fetching stock data for 100 tickers (this may take a moment)…"):
        closes_df = fetch_stock_closes(STOCK_TICKERS, start_date, end_date)

    if closes_df.empty:
        st.error("Could not fetch stock data. Please try again later.")
        st.stop()

    # Drop tickers with all NaN values (delisted / unavailable)
    closes_df = closes_df.dropna(axis=1, how="all")
    available_count = closes_df.shape[1]

    # ---- Calculate WHI ----
    whi = calculate_whi(closes_df)

    # Align WHI and IHSG to common dates
    common_idx = ihsg_df.index.intersection(whi.index)
    ihsg_df = ihsg_df.loc[common_idx]
    whi = whi.loc[common_idx]

    # Drop initial NaN rows (warm-up period)
    valid_mask = whi.notna()
    ihsg_df = ihsg_df.loc[valid_mask]
    whi = whi.loc[valid_mask]

    # ---- Metrics ----
    latest_whi = whi.iloc[-1] if not whi.empty else 0
    latest_ihsg = ihsg_df["Close"].iloc[-1] if not ihsg_df.empty else 0
    prev_whi = whi.iloc[-2] if len(whi) > 1 else latest_whi
    prev_ihsg = ihsg_df["Close"].iloc[-2] if len(ihsg_df) > 1 else latest_ihsg

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Latest IHSG Close",
            value=f"{latest_ihsg:,.2f}",
            delta=f"{latest_ihsg - prev_ihsg:,.2f}",
        )
    with col2:
        st.metric(
            label="Latest WHI Value",
            value=f"{latest_whi:,.2f}",
            delta=f"{latest_whi - prev_whi:,.2f}",
        )
    with col3:
        st.metric(
            label="Stocks Tracked",
            value=f"{available_count} / {len(STOCK_TICKERS)}",
        )

    # ---- Chart ----
    st.markdown("---")
    fig = build_chart(ihsg_df, whi)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Data table (expandable) ----
    with st.expander("📊 View raw WHI data table"):
        display_df = pd.DataFrame({
            "IHSG Close": ihsg_df["Close"],
            "WHI": whi,
        })
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
