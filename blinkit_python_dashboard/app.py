"""Blinkit sales & delivery dashboard (Streamlit + Plotly).

Run:  streamlit run app.py
Data: the cleaned CSVs from the cleaning notebook live in ./data
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Blinkit sales & delivery", page_icon="🟡", layout="wide")

# ───────────────────────── design tokens ─────────────────────────
INK, MUTED, GRID = "#12372A", "#5F7C6E", "#D5E3DA"
CANVAS, PANEL = "#F1F6F3", "#FFFFFF"
GREEN, YELLOW, AMBER, CORAL = "#14934A", "#FFD23F", "#FFA62B", "#EE5A3F"
SAGE, BLUE = "#9CCFB3", "#2F7DA6"
STATUS_COLORS = {"On Time": GREEN, "Slightly Delayed": AMBER, "Significantly Delayed": CORAL}
STATUS_ORDER = list(STATUS_COLORS)
SENTIMENT_COLORS = {"Positive": GREEN, "Neutral": "#C4D5CB", "Negative": CORAL}
CAT_COLORS = [GREEN, YELLOW, CORAL, BLUE, INK, "#8BC34A", AMBER, "#7A5AF8", "#E85D9B", "#2FB5A3", "#9A7B4F"]
FONT = "Instrument Sans, Segoe UI, system-ui, sans-serif"
DISPLAY = "Bricolage Grotesque, Instrument Sans, system-ui, sans-serif"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CHART_CFG = {"displayModeBar": False}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Instrument+Sans:wght@400..700&display=swap');
html, body, [class*="css"], .stApp { font-family: __FONT__; }
.stApp { background: __CANVAS__; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1280px; padding-top: 2.4rem; padding-bottom: 4rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: #E3EEE7; border-right: 1px solid __GRID__; }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.side-title { font-family: __DISPLAY__; font-weight: 700; font-size: 1.35rem; margin-bottom: .15rem; }
.side-note { color: __MUTED__; font-size: .86rem; line-height: 1.45; margin-bottom: 1.1rem; }

/* hero */
.brand { display: flex; align-items: center; gap: .6rem; color: __MUTED__; font-weight: 600; font-size: .95rem; }
.brand .mark { width: 30px; height: 30px; border-radius: 9px 9px 9px 3px; background: __YELLOW__;
               display: grid; place-items: center; }
.hero h1 { font-family: __DISPLAY__; font-weight: 700; font-size: clamp(2rem, 4.2vw, 3.4rem);
           line-height: 1.02; letter-spacing: -0.025em; margin: .9rem 0 .7rem; max-width: 22ch; color: __INK__; padding: 0; }
.hero p { color: __MUTED__; font-size: 1.05rem; line-height: 1.55; max-width: 62ch; margin: 0 0 1.6rem; }

/* kpi strip */
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); background: __PANEL__; border: 1px solid __GRID__;
        border-radius: 26px; overflow: hidden; margin-bottom: 1.6rem; }
.kpi { padding: 1.15rem 1.3rem 1rem; border-left: 1px solid __GRID__; min-width: 0; }
.kpi:first-child { border-left: none; }
.kpi-label { color: __MUTED__; font-size: .88rem; font-weight: 600; }
.kpi-value { font-family: __DISPLAY__; font-weight: 700; font-size: 1.75rem; white-space: nowrap; letter-spacing: -0.02em; line-height: 1.15; margin: .15rem 0 .5rem; }
.kpi-foot { display: flex; align-items: center; justify-content: space-between; gap: .5rem; min-height: 28px; }
.chip { font-size: .78rem; font-weight: 700; padding: .12rem .5rem; border-radius: 999px; white-space: nowrap; }
.chip.good { background: #DDF3E6; color: #0E7437; }
.chip.bad { background: #FDE3DD; color: #B8321B; }
.chip.flat { background: #E9EFEB; color: __MUTED__; }
@media (max-width: 1000px) { .kpis { grid-template-columns: repeat(2, 1fr) !important; } .kpi { border-top: 1px solid __GRID__; } }

/* panels */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .pt) { background: __PANEL__;
        border: 1px solid __GRID__ !important; border-radius: 20px; padding: .75rem .9rem .35rem; }
.pt { font-family: __DISPLAY__; font-weight: 700; font-size: 1.2rem; letter-spacing: -0.01em; color: __INK__; }
.ps { color: __MUTED__; font-size: .9rem; line-height: 1.45; margin: .15rem 0 .5rem; max-width: 70ch; }
.section { font-family: __DISPLAY__; font-weight: 700; font-size: 1.55rem; letter-spacing: -0.015em; margin: 1.8rem 0 .8rem; }

/* tabs */
div[data-baseweb="tab-list"] { gap: .4rem; border-bottom: 2px solid __GRID__; margin-bottom: 1rem; }
button[data-baseweb="tab"] { font-family: __DISPLAY__; font-weight: 600; font-size: 1.02rem; padding: .55rem 1rem;
        border-radius: 12px 12px 0 0; }
button[data-baseweb="tab"][aria-selected="true"] { color: __INK__; background: #FFF4C2; }
div[data-baseweb="tab-highlight"] { background: __YELLOW__ !important; height: 4px !important; }

/* delivery bands (legend for the ruler chart) */
.band { display: flex; gap: .8rem; align-items: flex-start; padding: .8rem 0; border-top: 1px solid __GRID__; }
.band:first-child { border-top: none; padding-top: .2rem; }
.band .sw { width: 14px; height: 44px; border-radius: 5px; flex: none; }
.band .big { font-family: __DISPLAY__; font-weight: 700; font-size: 1.9rem; line-height: 1; letter-spacing: -0.02em; }
.band .lab { font-weight: 600; font-size: .95rem; margin-top: .15rem; }
.band .sub { color: __MUTED__; font-size: .85rem; }

/* small stat tiles */
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin-bottom: 1.1rem; }
.tile { background: __PANEL__; border: 1px solid __GRID__; border-radius: 16px; padding: .9rem 1.1rem; }
.tile .v { font-family: __DISPLAY__; font-weight: 700; font-size: 1.7rem; letter-spacing: -0.02em; }
.tile .l { color: __MUTED__; font-size: .88rem; font-weight: 600; }
@media (max-width: 700px) { .tiles { grid-template-columns: 1fr; } }
</style>
"""
for k, v in {"__FONT__": FONT, "__DISPLAY__": DISPLAY, "__CANVAS__": CANVAS, "__GRID__": GRID, "__MUTED__": MUTED,
             "__YELLOW__": YELLOW, "__INK__": INK, "__PANEL__": PANEL}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)


# ───────────────────────── data ─────────────────────────
DATA = Path(__file__).parent / "data"


def _csv(name, **kw):
    return pd.read_csv(DATA / f"blinkit_{name}_cleaned.csv", **kw)


@st.cache_data(show_spinner="Loading data…")
def load():
    orders = _csv("orders", usecols=[
        "order_id", "customer_id", "order_date", "delivery_status", "order_total", "payment_method",
        "order_month", "order_day_of_week", "order_hour", "promised_delivery_minutes",
        "actual_delivery_minutes", "delivery_time_bucket"], parse_dates=["order_date"])
    orders["delta"] = orders["actual_delivery_minutes"] - orders["promised_delivery_minutes"]

    customers = _csv("customers", usecols=["customer_id", "area", "registration_date", "customer_segment"],
                     parse_dates=["registration_date"])
    orders = orders.merge(customers[["customer_id", "area", "customer_segment"]], on="customer_id", how="left")

    products = _csv("products", usecols=["product_id", "product_name", "category", "brand", "price",
                                         "margin_percentage", "shelf_life_days"])
    items = _csv("order_items").merge(products, on="product_id", how="left")
    items["revenue"] = items["quantity"] * items["unit_price"]

    delivery = _csv("delivery", usecols=["order_id", "distance_km", "reasons_if_delayed"])
    feedback = _csv("feedback", parse_dates=["feedback_date"])
    marketing = _csv("marketing", parse_dates=["date"])
    inventory = _csv("inventory", parse_dates=["date"]).merge(
        products[["product_id", "product_name", "category", "shelf_life_days"]], on="product_id", how="left")
    return dict(orders=orders, customers=customers, items=items, delivery=delivery, feedback=feedback,
                marketing=marketing, inventory=inventory, products=products)


D = load()
ORD = D["orders"]
DMIN, DMAX = ORD["order_date"].min().normalize(), ORD["order_date"].max().normalize()

# "On time" in this dataset means within a grace window of the promise; read the window from the data itself
_rng = ORD.groupby("delivery_status")["delta"].agg(["min", "max"])
GRACE = int(_rng.loc["On Time", "max"])
SLIGHT_LO, SLIGHT_HI = int(_rng.loc["Slightly Delayed", "min"]), int(_rng.loc["Slightly Delayed", "max"])
SIG_LO = int(_rng.loc["Significantly Delayed", "min"])

# months that are not full calendar months in the source data
PARTIAL = set()
if ORD["order_date"].min().day > 1:
    PARTIAL.add(ORD["order_date"].min().strftime("%Y-%m"))
if ORD["order_date"].max().day < ORD["order_date"].max().days_in_month:
    PARTIAL.add(ORD["order_date"].max().strftime("%Y-%m"))


# ───────────────────────── helpers ─────────────────────────
def style(fig, h=340, legend=False, x_grid=False):
    fig.update_layout(
        height=h, margin=dict(l=4, r=4, t=10, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color=INK), showlegend=legend, colorway=CAT_COLORS,
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=x_grid, gridcolor="#E7EFEA", linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#E7EFEA", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", config=CHART_CFG, key=key)


def head(title, sub=""):
    st.markdown(f"<div class='pt'>{title}</div><div class='ps'>{sub}</div>", unsafe_allow_html=True)


def fmt_money(x):
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def spark(values, color=GREEN, w=108, h=30):
    v = [x for x in values if pd.notna(x)]
    if len(v) < 3:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1
    pts = [(i * (w - 6) / (len(v) - 1) + 3, h - 4 - (x - lo) / span * (h - 8)) for i, x in enumerate(v)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    lx, ly = pts[-1]
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><polyline points='{path}' fill='none' stroke='{color}' "
            f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/><circle cx='{lx:.1f}' cy='{ly:.1f}' r='3.2' "
            f"fill='{color}'/></svg>")


def chip(cur, prev, good_up=True, pp=False):
    if prev is None or prev == 0 or pd.isna(prev) or pd.isna(cur):
        return ""
    change = (cur - prev) * 100 if pp else (cur / prev - 1) * 100
    if abs(change) < 0.05:
        return "<span class='chip flat'>no change</span>"
    up = change > 0
    cls = "good" if up == good_up else "bad"
    unit = " pts" if pp else "%"
    return f"<span class='chip {cls}'>{'▲' if up else '▼'} {abs(change):.1f}{unit}</span>"


def strip(items):
    cells = "".join(
        f"<div class='kpi'><div class='kpi-label'>{i['label']}</div><div class='kpi-value'>{i['value']}</div>"
        f"<div class='kpi-foot'>{i.get('chip', '')}{i.get('spark', '')}</div></div>" for i in items)
    st.markdown(f"<div class='kpis' style='grid-template-columns: repeat({len(items)}, 1fr)'>{cells}</div>",
                unsafe_allow_html=True)


def order_kpis(o):
    return dict(orders=len(o), revenue=o["order_total"].sum(), aov=o["order_total"].mean(),
                ontime=(o["delivery_status"] == "On Time").mean(), minutes=o["actual_delivery_minutes"].mean())


# ───────────────────────── sidebar filters ─────────────────────────
PAY_OPTS = sorted(ORD["payment_method"].unique())
SEG_OPTS = sorted(ORD["customer_segment"].dropna().unique())
CAT_OPTS = sorted(D["items"]["category"].dropna().unique())


def reset_filters():
    st.session_state["dates"] = (DMIN.date(), DMAX.date())
    st.session_state["pay"], st.session_state["seg"], st.session_state["cat"] = [], [], []


with st.sidebar:
    st.markdown("<div class='side-title'>Filters</div><div class='side-note'>Everything on the page responds. "
                "Marketing and inventory use the date range only.</div>", unsafe_allow_html=True)
    dates = st.date_input("Order dates", value=(DMIN.date(), DMAX.date()), min_value=DMIN.date(),
                          max_value=DMAX.date(), key="dates")
    pay = st.multiselect("Payment method", PAY_OPTS, placeholder="All methods", key="pay")
    seg = st.multiselect("Customer segment", SEG_OPTS, placeholder="All segments", key="seg")
    cat = st.multiselect("Product category", CAT_OPTS, placeholder="All categories", key="cat")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

if not isinstance(dates, (tuple, list)) or len(dates) != 2:
    st.info("Pick an end date to apply the date range.")
    st.stop()

START, END = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)


def apply_filters(start, end):
    m = (ORD["order_date"] >= start) & (ORD["order_date"] < end)
    if pay:
        m &= ORD["payment_method"].isin(pay)
    if seg:
        m &= ORD["customer_segment"].isin(seg)
    o = ORD[m]
    it = D["items"][D["items"]["order_id"].isin(o["order_id"])]
    if cat:
        it = it[it["category"].isin(cat)]
        o = o[o["order_id"].isin(it["order_id"])]
    return o, it


O, I = apply_filters(START, END)
if O.empty:
    st.warning("No orders match these filters. Widen the date range or clear a filter in the sidebar.")
    st.stop()

span = END - START
PREV, _ = apply_filters(START - span, START)
K, KP = order_kpis(O), (order_kpis(PREV) if len(PREV) >= 30 else None)

# ───────────────────────── hero + KPI strip ─────────────────────────
last_day = (END - pd.Timedelta(days=1)).strftime("%d %b %Y")
st.markdown(
    "<div class='hero'><div class='brand'><span class='mark'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' "
    f"stroke='{INK}' stroke-width='2.4' stroke-linecap='round'><circle cx='12' cy='13' r='7.5'/><path d='M12 9v4l2.5 2'/>"
    "<path d='M9.5 2.5h5'/></svg></span>Blinkit sales and delivery</div>"
    f"<h1>{K['ontime']:.0%} of orders arrive within {GRACE} minutes of the promised time.</h1>"
    f"<p>{K['orders']:,} orders between {START:%d %b %Y} and {last_day}. Customers waited "
    f"{K['minutes']:.0f} minutes on average against a promise of {O['promised_delivery_minutes'].mean():.0f}.</p></div>",
    unsafe_allow_html=True)

freq = "D" if span.days <= 45 else "W" if span.days <= 180 else "MS"
FREQ_NAME = {"D": "daily", "W": "weekly", "MS": "monthly"}[freq]


def trim(series):
    """Drop the first and last bucket when there are enough, since they are usually partial periods."""
    v = series.tolist()
    return v[1:-1] if len(v) >= 6 and freq != "D" else v


ts = O.set_index("order_date").sort_index()
res = ts.resample(freq)
sp = dict(
    orders=trim(res.size()), revenue=trim(res["order_total"].sum()), aov=trim(res["order_total"].mean()),
    ontime=trim(res["delivery_status"].apply(lambda s: (s == "On Time").mean())),
    minutes=trim(res["actual_delivery_minutes"].mean()))
strip([
    dict(label="Orders", value=f"{K['orders']:,}", chip=chip(K["orders"], KP and KP["orders"]), spark=spark(sp["orders"])),
    dict(label="Revenue", value=fmt_money(K["revenue"]), chip=chip(K["revenue"], KP and KP["revenue"]),
         spark=spark(sp["revenue"])),
    dict(label="Average order", value=f"₹{K['aov']:,.0f}", chip=chip(K["aov"], KP and KP["aov"]), spark=spark(sp["aov"])),
    dict(label="Delivered on time", value=f"{K['ontime']:.1%}", chip=chip(K["ontime"], KP and KP["ontime"], pp=True),
         spark=spark(sp["ontime"], GREEN)),
    dict(label="Average delivery time", value=f"{K['minutes']:.1f} min",
         chip=chip(K["minutes"], KP and KP["minutes"], good_up=False), spark=spark(sp["minutes"], AMBER)),
])
st.caption(f"Sparklines show the {FREQ_NAME} trend." + (f" Change chips compare with the previous {span.days} days." if KP is not None else ""))

tab_speed, tab_sales, tab_people, tab_mkt, tab_inv = st.tabs(
    ["Delivery speed", "Sales and products", "Customers and feedback", "Marketing", "Inventory"])

# ───────────────────────── tab: delivery speed ─────────────────────────
with tab_speed:
    left, right = st.columns([2.2, 1], gap="medium")
    with left, st.container(border=True):
        head("How far off the promise was every order?",
             "Each bar counts orders delivered that many minutes before (negative) or after (positive) the promised time.")
        vc = O["delta"].round().astype(int).value_counts().sort_index()
        colors = [GREEN if x <= GRACE else AMBER if x <= SLIGHT_HI else CORAL for x in vc.index]
        labels = [f"{-x} min early" if x < 0 else "right on the promise" if x == 0 else f"{x} min late" for x in vc.index]
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=colors, customdata=labels,
                               hovertemplate="<b>%{y:,} orders</b><br>%{customdata}<extra></extra>"))
        fig.add_vline(x=GRACE + 0.5, line_dash="dot", line_color=INK, opacity=.5,
                      annotation_text=f"{GRACE}-minute grace", annotation_position="top right",
                      annotation_font=dict(size=12, color=MUTED))
        fig.update_layout(bargap=0.12, xaxis_title="Minutes after the promised time (negative means early)",
                          yaxis_title="Orders", xaxis=dict(dtick=5))
        show(style(fig, 360), "ch_ruler")
    with right, st.container(border=True):
        head("Where orders land", "Share of orders in each delivery outcome.")
        share = O["delivery_status"].value_counts(normalize=True)
        rows = [("On Time", "On time", f"within {GRACE} min of the promise"),
                ("Slightly Delayed", "Slightly delayed", f"{SLIGHT_LO}–{SLIGHT_HI} min late"),
                ("Significantly Delayed", "Significantly delayed", f"{SIG_LO}+ min late")]
        html = "".join(
            f"<div class='band'><span class='sw' style='background:{STATUS_COLORS[k]}'></span><div>"
            f"<div class='big'>{share.get(k, 0):.0%}</div><div class='lab'>{lab}</div><div class='sub'>{sub}</div></div></div>"
            for k, lab, sub in rows)
        st.markdown(html, unsafe_allow_html=True)
        late = O[O["delivery_status"] != "On Time"]
        if len(late):
            st.markdown(f"<div class='sub' style='color:{MUTED};font-size:.85rem;padding:.2rem 0 .6rem'>"
                        f"Late orders ran {late['delta'].mean():.0f} minutes over on average.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, st.container(border=True):
        head("When do orders run late?", "Switch the view to see where volume peaks and where lateness concentrates.")
        view = st.radio("View", ["Late share", "Order volume"], horizontal=True, label_visibility="collapsed", key="heat_view")
        if view == "Late share":
            z = (O.assign(late=O["delivery_status"] != "On Time").pivot_table(
                index="order_day_of_week", columns="order_hour", values="late", aggfunc="mean")
                 .reindex(index=DAYS, columns=range(24)))
            scale, ztxt, zfmt = [[0, "#FFF1D1"], [1, CORAL]], "late", ".0%"
        else:
            z = (O.pivot_table(index="order_day_of_week", columns="order_hour", values="order_id", aggfunc="count")
                 .reindex(index=DAYS, columns=range(24)).fillna(0))
            scale, ztxt, zfmt = [[0, "#E6F3EB"], [1, GREEN]], "orders", ",.0f"
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[f"{h}:00" for h in range(24)], y=[d[:3] for d in DAYS], colorscale=scale, xgap=3, ygap=3,
            hovertemplate=f"%{{y}} at %{{x}}<br><b>%{{z:{zfmt}}}</b> {ztxt}<extra></extra>", colorbar=dict(thickness=10, len=.9,
                tickformat=zfmt, outlinewidth=0)))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickmode="array", tickvals=[f"{h}:00" for h in range(0, 24, 3)])
        show(style(fig, 330), "ch_heat")
    with c2, st.container(border=True):
        head("Does distance matter?", "Delivery outcome by distance travelled.")
        dd = O[["order_id", "delivery_status"]].merge(D["delivery"][["order_id", "distance_km"]], on="order_id")
        dd["band"] = pd.cut(dd["distance_km"], [0, 1, 2, 3, 4, 5.01], labels=["Under 1 km", "1–2 km", "2–3 km", "3–4 km", "4–5 km"])
        ct = pd.crosstab(dd["band"], dd["delivery_status"], normalize="index").reindex(columns=STATUS_ORDER).fillna(0)
        cnt = dd["band"].value_counts()
        fig = go.Figure()
        for s in STATUS_ORDER:
            fig.add_bar(y=ct.index.astype(str), x=ct[s], name=s, orientation="h", marker_color=STATUS_COLORS[s],
                        hovertemplate=f"%{{y}}<br>{s}: <b>%{{x:.0%}}</b><extra></extra>")
        fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]), bargap=.28)
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 330, legend=True), "ch_dist")

    c3, c4 = st.columns([1.6, 1], gap="medium")
    with c3, st.container(border=True):
        head("The promise versus the clock", "Average promised and actual delivery minutes per month. The shaded gap is the shortfall.")
        mm = O.groupby("order_month").agg(promised=("promised_delivery_minutes", "mean"),
                                          actual=("actual_delivery_minutes", "mean"), n=("order_id", "size")).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=mm["order_month"], y=mm["promised"], name="Promised", line=dict(color=INK, width=2, dash="dot"),
                        hovertemplate="%{x}<br>Promised <b>%{y:.1f} min</b><extra></extra>")
        fig.add_scatter(x=mm["order_month"], y=mm["actual"], name="Actual", line=dict(color=AMBER, width=3), fill="tonexty",
                        fillcolor="rgba(255,166,43,.18)", customdata=mm["n"],
                        hovertemplate="%{x}<br>Actual <b>%{y:.1f} min</b> (%{customdata} orders)<extra></extra>")
        fig.update_yaxes(title="Minutes", rangemode="tozero")
        show(style(fig, 300, legend=True), "ch_promise")
    with c4, st.container(border=True):
        head("How long customers actually wait", "Orders by real delivery time.")
        order = ["<10 min", "10-20 min", "20-30 min", "30+ min"]
        bk = O["delivery_time_bucket"].value_counts().reindex(order).fillna(0)
        fig = go.Figure(go.Bar(y=[o.replace("-", "–") for o in order], x=bk.values / bk.sum(), orientation="h",
                               marker_color=[GREEN, SAGE, AMBER, CORAL], customdata=bk.values,
                               text=[f"{v:.0%}" for v in bk.values / bk.sum()], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br><b>%{customdata:,.0f} orders</b><extra></extra>"))
        fig.update_xaxes(visible=False, range=[0, max(bk.values / bk.sum()) * 1.2])
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 300), "ch_buckets")

# ───────────────────────── tab: sales and products ─────────────────────────
with tab_sales:
    with st.container(border=True):
        head("Trend by month", "Pick a measure. Lighter bars are partial months in the source data, so they look smaller than they are.")
        metric = st.radio("Measure", ["Revenue", "Orders", "Average order", "On-time rate"], horizontal=True,
                          label_visibility="collapsed", key="trend_metric")
        g = O.groupby("order_month").agg(Revenue=("order_total", "sum"), Orders=("order_id", "size"),
                                         **{"Average order": ("order_total", "mean")},
                                         ontime=("delivery_status", lambda s: (s == "On Time").mean())).reset_index()
        g = g.rename(columns={"ontime": "On-time rate"})
        fmt = {"Revenue": "₹%{y:,.0f}", "Orders": "%{y:,}", "Average order": "₹%{y:,.0f}", "On-time rate": "%{y:.1%}"}[metric]
        fig = go.Figure(go.Bar(
            x=g["order_month"], y=g[metric], marker_color=[SAGE if m in PARTIAL else GREEN for m in g["order_month"]],
            customdata=["partial month" if m in PARTIAL else "full month" for m in g["order_month"]],
            hovertemplate="%{x}<br><b>" + fmt + "</b> (%{customdata})<extra></extra>"))
        fig.update_layout(bargap=.25)
        if metric == "On-time rate":
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        show(style(fig, 330), "ch_trend")

    s1, s2 = st.columns([1.6, 1], gap="medium")
    with s1, st.container(border=True):
        head("Revenue by category", "Line-item value (quantity × price). Hover a bar for units sold.")
        cr = I.groupby("category").agg(revenue=("revenue", "sum"), units=("quantity", "sum")).sort_values("revenue")
        fig = go.Figure(go.Bar(
            y=cr.index, x=cr["revenue"], orientation="h", customdata=cr["units"],
            marker_color=[YELLOW if v == cr["revenue"].max() else GREEN for v in cr["revenue"]],
            hovertemplate="%{y}<br><b>₹%{x:,.0f}</b> from %{customdata:,} units<extra></extra>"))
        fig.update_xaxes(tickprefix="₹", tickformat="~s")
        show(style(fig, 400, x_grid=True), "ch_cat")
    with s2, st.container(border=True):
        head("How customers pay", "Share of orders by payment method.")
        pm = O.groupby("payment_method").agg(n=("order_id", "size"), aov=("order_total", "mean")).reset_index()
        fig = go.Figure(go.Pie(labels=pm["payment_method"], values=pm["n"], hole=.62, sort=False, customdata=pm["aov"],
                               marker=dict(colors=[GREEN, YELLOW, INK, SAGE], line=dict(color="white", width=3)),
                               textinfo="percent", textfont=dict(size=13, family=FONT),
                               hovertemplate="%{label}<br><b>%{value:,} orders</b><br>average ₹%{customdata:,.0f}<extra></extra>"))
        fig.add_annotation(text=f"<b>{len(O):,}</b><br><span style='font-size:12px;color:{MUTED}'>orders</span>",
                           showarrow=False, font=dict(size=24, family=DISPLAY))
        show(style(fig, 400, legend=True), "ch_pay")

    s3, s4 = st.columns([1.15, 1], gap="medium")
    with s3, st.container(border=True):
        head("Where the money and the margin are", "Each bubble is a category: revenue across, average margin up, size is units sold. Click a legend entry to isolate it.")
        cm = I.groupby("category").agg(revenue=("revenue", "sum"), margin=("margin_percentage", "mean"),
                                       units=("quantity", "sum")).reset_index().sort_values("revenue").reset_index(drop=True)
        fig = go.Figure()
        for i, r in cm.iterrows():
            fig.add_scatter(
                x=[r["revenue"]], y=[r["margin"]], mode="markers", name=r["category"],
                marker=dict(size=[r["units"]], sizemode="area", sizeref=2.0 * cm["units"].max() / (38 ** 2),
                            color=CAT_COLORS[i % len(CAT_COLORS)], line=dict(color="white", width=2), opacity=.92),
                hovertemplate=f"<b>{r['category']}</b><br>Revenue ₹%{{x:,.0f}}<br>Margin %{{y:.1f}}%<br>{int(r['units']):,} units<extra></extra>")
        fig.update_xaxes(title="Revenue", tickprefix="₹", tickformat="~s", showgrid=True)
        fig.update_yaxes(title="Average margin (%)")
        fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)))
        show(style(fig, 400, legend=True), "ch_bubble")
    with s4, st.container(border=True):
        head("Ten best-selling products", "By line-item revenue.")
        tp = (I.groupby(["product_id", "product_name", "brand"]).agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
              .reset_index().nlargest(10, "revenue").sort_values("revenue"))
        fig = go.Figure(go.Bar(
            y=tp["product_name"], x=tp["revenue"], orientation="h", marker_color=GREEN,
            customdata=tp[["units", "brand"]].values,
            hovertemplate="<b>%{y}</b> by %{customdata[1]}<br>₹%{x:,.0f} from %{customdata[0]:,} units<extra></extra>"))
        fig.update_xaxes(tickprefix="₹", tickformat="~s", showgrid=True, nticks=4, tickangle=0)
        fig.update_yaxes(tickfont=dict(size=11))
        show(style(fig, 380, x_grid=True), "ch_top")

# ───────────────────────── tab: customers and feedback ─────────────────────────
with tab_people:
    per = O.groupby("customer_id").size()
    st.markdown(
        "<div class='tiles'>"
        f"<div class='tile'><div class='v'>{len(per):,}</div><div class='l'>Customers who ordered</div></div>"
        f"<div class='tile'><div class='v'>{(per > 1).mean():.0%}</div><div class='l'>Ordered more than once</div></div>"
        f"<div class='tile'><div class='v'>{per.mean():.1f}</div><div class='l'>Orders per customer</div></div></div>",
        unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="medium")
    with p1, st.container(border=True):
        head("How often customers come back", "Customers grouped by the number of orders they placed in this view.")
        dist = per.clip(upper=5).value_counts().sort_index()
        names = [str(int(i)) if i < 5 else "5 or more" for i in dist.index]
        fig = go.Figure(go.Bar(x=names, y=dist.values.tolist(), marker_color=[SAGE] + [GREEN] * (len(dist) - 1),
                               text=dist.values, textposition="outside", cliponaxis=False,
                               hovertemplate="%{x} orders<br><b>%{y:,} customers</b><extra></extra>"))
        fig.update_layout(xaxis_title="Orders placed", yaxis_title="Customers")
        fig.update_xaxes(type="category")
        show(style(fig, 320), "ch_freq")
    with p2, st.container(border=True):
        head("Average order by customer segment", "Hover to see how many orders sit behind each bar.")
        sg = O.groupby("customer_segment").agg(aov=("order_total", "mean"), n=("order_id", "size")).sort_values("aov")
        fig = go.Figure(go.Bar(y=sg.index, x=sg["aov"], orientation="h", marker_color=GREEN, customdata=sg["n"],
                               text=[f"₹{v:,.0f}" for v in sg["aov"]], textposition="inside", insidetextanchor="end",
                               textfont=dict(color="white", size=13),
                               hovertemplate="%{y}<br><b>₹%{x:,.0f}</b> average over %{customdata:,} orders<extra></extra>"))
        fig.update_xaxes(visible=False)
        show(style(fig, 320), "ch_seg")

    p3, p4 = st.columns(2, gap="medium")
    with p3, st.container(border=True):
        head("New registrations", "Customers who signed up each month within the selected dates. Partial months are left out.")
        cu = D["customers"]
        cu = cu[(cu["registration_date"] >= START) & (cu["registration_date"] < END)]
        reg = cu.groupby(cu["registration_date"].dt.to_period("M").astype(str)).size()
        reg = reg[~reg.index.isin(PARTIAL)]
        fig = go.Figure(go.Scatter(x=reg.index, y=reg.values, mode="lines+markers", line=dict(color=GREEN, width=3, shape="spline"),
                                   fill="tozeroy", fillcolor="rgba(20,147,74,.12)", marker=dict(size=6, color=GREEN),
                                   hovertemplate="%{x}<br><b>%{y:,} sign-ups</b><extra></extra>"))
        show(style(fig, 300), "ch_reg")
    with p4, st.container(border=True):
        head("Busiest areas", "Ten areas with the most orders.")
        ar = O.groupby("area").size().nlargest(10).sort_values()
        fig = go.Figure(go.Bar(y=ar.index, x=ar.values, orientation="h", marker_color=GREEN,
                               hovertemplate="%{y}<br><b>%{x:,} orders</b><extra></extra>"))
        show(style(fig, 300, x_grid=True), "ch_area")

    st.markdown("<div class='section'>What customers say</div>", unsafe_allow_html=True)
    F = D["feedback"].merge(O[["order_id", "delivery_status"]], on="order_id")
    if F.empty:
        st.info("No feedback matches the current filters.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.markdown(f"<div class='tile'><div class='v'>{F['rating'].mean():.2f} / 5</div><div class='l'>Average rating</div></div>", unsafe_allow_html=True)
        t2.markdown(f"<div class='tile'><div class='v'>{(F['sentiment'] == 'Positive').mean():.0%}</div><div class='l'>Positive feedback</div></div>", unsafe_allow_html=True)
        t3.markdown(f"<div class='tile'><div class='v'>{(F['sentiment'] == 'Negative').mean():.0%}</div><div class='l'>Negative feedback</div></div>", unsafe_allow_html=True)
        st.write("")
        f1, f2, f3 = st.columns(3, gap="medium")
        with f1, st.container(border=True):
            head("Ratings", "Number of reviews at each star level.")
            rc = F["rating"].value_counts().reindex(range(1, 6)).fillna(0)
            fig = go.Figure(go.Bar(x=[f"{i}★" for i in rc.index], y=rc.values,
                                   marker_color=[CORAL, AMBER, YELLOW, SAGE, GREEN],
                                   hovertemplate="%{x}<br><b>%{y:,} reviews</b><extra></extra>"))
            show(style(fig, 290), "ch_rating")
        with f2, st.container(border=True):
            head("Sentiment by topic", "What each feedback category feels like.")
            sc = pd.crosstab(F["feedback_category"], F["sentiment"], normalize="index").reindex(columns=list(SENTIMENT_COLORS)).fillna(0)
            fig = go.Figure()
            for s in SENTIMENT_COLORS:
                fig.add_bar(y=sc.index, x=sc[s], name=s, orientation="h", marker_color=SENTIMENT_COLORS[s],
                            hovertemplate=f"%{{y}}<br>{s}: <b>%{{x:.0%}}</b><extra></extra>")
            fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]), bargap=.3)
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 290, legend=True), "ch_sent")
        with f3, st.container(border=True):
            head("Late orders and ratings", "Average rating by delivery outcome.")
            rs = F.groupby("delivery_status")["rating"].agg(["mean", "size"]).reindex(STATUS_ORDER).dropna()
            fig = go.Figure(go.Bar(x=[s.replace(" ", "<br>", 1) for s in rs.index], y=rs["mean"], customdata=rs["size"],
                                   marker_color=[STATUS_COLORS[s] for s in rs.index],
                                   text=[f"{v:.2f}" for v in rs["mean"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}<br><b>%{y:.2f}</b> average over %{customdata:,} reviews<extra></extra>"))
            fig.update_yaxes(range=[0, 5], title="Average rating")
            show(style(fig, 290), "ch_late_rating")

# ───────────────────────── tab: marketing ─────────────────────────
with tab_mkt:
    M0 = D["marketing"]
    M0 = M0[(M0["date"] >= START) & (M0["date"] < END)]
    mf1, mf2 = st.columns(2)
    chans = mf1.multiselect("Channel", sorted(D["marketing"]["channel"].unique()), placeholder="All channels", key="mk_ch")
    auds = mf2.multiselect("Target audience", sorted(D["marketing"]["target_audience"].unique()), placeholder="All audiences", key="mk_au")
    M = M0
    if chans:
        M = M[M["channel"].isin(chans)]
    if auds:
        M = M[M["target_audience"].isin(auds)]
    if M.empty:
        st.info("No campaigns in this date range and selection.")
    else:
        spend, rev = M["spend"].sum(), M["revenue_generated"].sum()
        mres = M.set_index("date").sort_index().resample("MS")
        strip([
            dict(label="Spend", value=fmt_money(spend), spark=spark(mres["spend"].sum().tolist()[1:-1], AMBER)),
            dict(label="Revenue generated", value=fmt_money(rev), spark=spark(mres["revenue_generated"].sum().tolist()[1:-1])),
            dict(label="Return on ad spend", value=f"{rev / spend:.2f}×" if spend else "n/a"),
            dict(label="Click rate", value=f"{M['clicks'].sum() / M['impressions'].sum():.1%}"),
            dict(label="Clicks that convert", value=f"{M['conversions'].sum() / M['clicks'].sum():.1%}"),
        ])
        m1, m2 = st.columns([1, 1.5], gap="medium")
        with m1, st.container(border=True):
            head("From view to purchase", "Impressions narrow to clicks, then conversions.")
            imp, clk, cnv = M["impressions"].sum(), M["clicks"].sum(), M["conversions"].sum()
            fig = go.Figure(go.Bar(
                y=["Impressions", "Clicks", "Conversions"], x=[imp, clk, cnv], orientation="h", marker_color=[SAGE, GREEN, INK],
                text=[f"{imp / 1e6:.1f}M", f"{clk / 1e6:.2f}M ({clk / imp:.0%} of views)", f"{cnv / 1e3:.0f}K ({cnv / clk:.0%} of clicks)"],
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: <b>%{x:,}</b><extra></extra>"))
            fig.update_xaxes(visible=False, range=[0, imp * 1.9])
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 320), "ch_funnel")
        with m2, st.container(border=True):
            head("Spend and revenue by month", "Bars are money in and out; the line is return on ad spend. Faded months are partial.")
            mg = M.groupby(M["date"].dt.to_period("M").astype(str)).agg(spend=("spend", "sum"), rev=("revenue_generated", "sum")).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fade = [0.4 if m in PARTIAL else 1 for m in mg["date"]]
            fig.add_bar(x=mg["date"], y=mg["spend"], name="Spend", marker=dict(color=YELLOW, opacity=fade), hovertemplate="%{x}<br>Spend <b>₹%{y:,.0f}</b><extra></extra>")
            fig.add_bar(x=mg["date"], y=mg["rev"], name="Revenue", marker=dict(color=GREEN, opacity=fade), hovertemplate="%{x}<br>Revenue <b>₹%{y:,.0f}</b><extra></extra>")
            fig.add_scatter(x=mg["date"], y=mg["rev"] / mg["spend"], name="Return on ad spend", mode="lines+markers",
                            line=dict(color=INK, width=2.5), secondary_y=True, hovertemplate="%{x}<br>Return <b>%{y:.2f}×</b><extra></extra>")
            fig.update_layout(barmode="group", bargap=.25)
            fig.update_yaxes(tickprefix="₹", tickformat="~s", secondary_y=False)
            fig.update_yaxes(ticksuffix="×", showgrid=False, range=[0, 3], dtick=0.5, secondary_y=True)
            show(style(fig, 320, legend=True), "ch_mkt_month")
        m3, m4 = st.columns([1.3, 1], gap="medium")
        with m3, st.container(border=True):
            head("Campaigns compared", "Return on ad spend per campaign. The dotted line is the overall return; hover for spend and revenue.")
            cp = M.groupby("campaign_name").agg(spend=("spend", "sum"), rev=("revenue_generated", "sum")).reset_index()
            cp["roas"] = cp["rev"] / cp["spend"]
            cp = cp.sort_values("roas")
            fig = go.Figure(go.Bar(
                y=cp["campaign_name"], x=cp["roas"], orientation="h", customdata=cp[["spend", "rev"]].values,
                marker_color=[INK if v == cp["roas"].max() else GREEN for v in cp["roas"]],
                text=[f"{v:.2f}×" for v in cp["roas"]], textposition="inside", insidetextanchor="end",
                textfont=dict(color="white", size=12),
                hovertemplate="<b>%{y}</b><br>Return %{x:.2f}×<br>Spend ₹%{customdata[0]:,.0f}<br>Revenue ₹%{customdata[1]:,.0f}<extra></extra>"))
            fig.add_vline(x=rev / spend, line_dash="dot", line_color=INK, opacity=.6)
            fig.update_xaxes(ticksuffix="×", showgrid=True, rangemode="tozero")
            show(style(fig, 360), "ch_camp")
        with m4, st.container(border=True):
            head("Channel and audience", "Return on ad spend for each pairing.")
            hm = M.groupby(["channel", "target_audience"]).apply(lambda t: t["revenue_generated"].sum() / t["spend"].sum(),
                                                                 include_groups=False).unstack()
            fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=[[0, "#FFF1D1"], [1, GREEN]],
                                       xgap=3, ygap=3, text=np.round(hm.values, 2), texttemplate="%{text}×",
                                       colorbar=dict(thickness=10, outlinewidth=0, ticksuffix="×"),
                                       hovertemplate="%{y} to %{x}<br>Return <b>%{z:.2f}×</b><extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 360), "ch_chan_aud")

# ───────────────────────── tab: inventory ─────────────────────────
with tab_inv:
    INV = D["inventory"]
    INV = INV[(INV["date"] >= START) & (INV["date"] < END)]
    if cat:
        INV = INV[INV["category"].isin(cat)]
    if INV.empty:
        st.info("No stock records for this selection.")
    else:
        rec, dam = INV["stock_received"].sum(), INV["damaged_stock"].sum()
        ires = INV.set_index("date").sort_index().resample("MS")
        strip([
            dict(label="Units received", value=f"{rec:,}", spark=spark(ires["stock_received"].sum().tolist()[1:-1])),
            dict(label="Units damaged", value=f"{dam:,}", spark=spark(ires["damaged_stock"].sum().tolist()[1:-1], CORAL)),
            dict(label="Damage rate", value=f"{dam / rec:.1%}", spark=spark((ires["damaged_stock"].sum() / ires["stock_received"].sum()).tolist()[1:-1], CORAL)),
            dict(label="Products tracked", value=f"{INV['product_id'].nunique():,}"),
        ])
        with st.container(border=True):
            head("Received versus damaged", "Monthly units, with the damage rate as a line. Faded months are partial.")
            im = INV.groupby(INV["date"].dt.to_period("M").astype(str)).agg(rec=("stock_received", "sum"), dam=("damaged_stock", "sum")).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fade = [0.4 if m in PARTIAL else 1 for m in im["date"]]
            fig.add_bar(x=im["date"], y=im["rec"], name="Received", marker=dict(color=GREEN, opacity=fade), hovertemplate="%{x}<br>Received <b>%{y:,}</b><extra></extra>")
            fig.add_bar(x=im["date"], y=im["dam"], name="Damaged", marker=dict(color=CORAL, opacity=fade), hovertemplate="%{x}<br>Damaged <b>%{y:,}</b><extra></extra>")
            fig.add_scatter(x=im["date"], y=im["dam"] / im["rec"], name="Damage rate", mode="lines+markers", line=dict(color=INK, width=2.5),
                            secondary_y=True, hovertemplate="%{x}<br>Damage rate <b>%{y:.1%}</b><extra></extra>")
            fig.update_layout(barmode="group", bargap=.25)
            fig.update_yaxes(secondary_y=False)
            fig.update_yaxes(tickformat=".0%", showgrid=False, rangemode="tozero", secondary_y=True)
            show(style(fig, 320, legend=True), "ch_inv_month")
        i1, i2 = st.columns(2, gap="medium")
        with i1, st.container(border=True):
            head("Damage rate by category", "Share of received units that arrived damaged.")
            ic = INV.groupby("category").agg(r=("stock_received", "sum"), d=("damaged_stock", "sum"))
            ic["rate"] = ic["d"] / ic["r"]
            ic = ic.sort_values("rate")
            fig = go.Figure(go.Bar(y=ic.index, x=ic["rate"], orientation="h", customdata=ic[["d", "r"]].values,
                                   marker_color=[CORAL if v == ic["rate"].max() else SAGE for v in ic["rate"]],
                                   hovertemplate="%{y}<br><b>%{x:.1%}</b> (%{customdata[0]:,} of %{customdata[1]:,} units)<extra></extra>"))
            fig.update_xaxes(tickformat=".0%")
            show(style(fig, 380, x_grid=True), "ch_inv_cat")
        with i2, st.container(border=True):
            head("Does shelf life matter?", "Damage rate by how long a product stays sellable.")
            sl = INV.groupby("shelf_life_days").agg(r=("stock_received", "sum"), d=("damaged_stock", "sum"), p=("product_id", "nunique"))
            sl["rate"] = sl["d"] / sl["r"]
            labels = [f"{int(x)} days" if x < 365 else "1 year" for x in sl.index]
            fig = go.Figure(go.Bar(x=labels, y=sl["rate"], marker_color=GREEN, customdata=sl["p"],
                                   text=[f"{v:.0%}" for v in sl["rate"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="Shelf life %{x}<br><b>%{y:.1%}</b> damaged across %{customdata} products<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            fig.update_layout(xaxis_title="Shelf life")
            show(style(fig, 380), "ch_inv_shelf")

# ───────────────────────── data notes ─────────────────────────
with st.expander("Notes on this data"):
    st.markdown(f"""
- **On time** follows the dataset's own delivery status: an order counts as on time when it lands within {GRACE} minutes of the promise, early or late.
  The separate delivery table flags any order after the promise as "late", so its late share is higher.
- **Revenue** in the top strip is the order total. **Category and product figures** use quantity × unit price from the order-items table,
  which does not add up to the order totals in this dataset, so compare shares rather than absolute rupees across the two.
- Discount percentage and margin percentage are identical for every product, so the dashboard uses margin only.
- Every order has a different store and delivery partner ID, so store and rider comparisons are not meaningful and are left out.
- {", ".join(sorted(PARTIAL)) if PARTIAL else "No month"} {"are" if len(PARTIAL) > 1 else "is"} a partial month in the source data.
""")
