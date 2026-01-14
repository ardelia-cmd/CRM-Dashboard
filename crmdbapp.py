import streamlit as st
import pandas as pd
import altair as alt
from datetime import date

from db_utils import create_tunnel_and_engine
from classifier import train_classifier, predict_first_three
from summary_metrics import compute_resp_df

MESSAGES_TABLE = '"Message"'
USER_TABLE = '"User"'
CHAT_TABLE = '"Chat"'


# -------------------- Helpers --------------------
def auto_sept1_start(max_day: date) -> date:
    sep1_this_year = date(max_day.year, 9, 1)
    return sep1_this_year if max_day >= sep1_this_year else date(max_day.year - 1, 9, 1)


def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    if d.month == 12:
        end = date(d.year + 1, 1, 1)
    else:
        end = date(d.year, d.month + 1, 1)
    return start, end


DISPLAY_NAME_MAP = {
    "Eko Education Consultant": "EC - Eko",
    "Lara International Global Network": "EC - Laras",
    "Zelma - Education Consultant": "EC - Zelma",
    "Rahma - Education Consultant": "EC - Rahma",
}


# -------------------- Cache helpers --------------------
@st.cache_resource
def get_engine():
    engine, _ = create_tunnel_and_engine()
    return engine


@st.cache_resource
def get_classifier():
    return train_classifier()


@st.cache_data(ttl=300)
def load_messages_with_joins():
    engine = get_engine()
    df = pd.read_sql(
        f"""
        SELECT
            m."userId",
            u."displayName" AS "displayName",
            m."chatId",
            c."name" AS "chatName",
            m."body",
            m."type",
            m."fromMe",
            m."timestamp"
        FROM {MESSAGES_TABLE} m
        LEFT JOIN {USER_TABLE} u
            ON u."id" = m."userId"
        LEFT JOIN {CHAT_TABLE} c
            ON c."id" = m."chatId"
        WHERE COALESCE(c."name", '') <> 'Educational Consultant'
        """,
        engine,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


@st.cache_data(ttl=300)
def build_pred_and_first():
    df_msg = load_messages_with_joins()
    clf = get_classifier()

    df_pred = predict_first_three(df_msg, clf)

    # Ensure displayName present (messages too, for avg reply hierarchy)
    df_pred["displayName"] = df_pred["displayName"].fillna(df_pred["userId"].astype(str))
    df_pred["displayName_ui"] = df_pred["displayName"].map(DISPLAY_NAME_MAP).fillna(df_pred["displayName"])

    if "pred_lbl_chat" not in df_pred.columns:
        raise ValueError("Column 'pred_lbl_chat' not found. Please confirm classifier output.")

    # First message per chatId for leads
    df_pred = df_pred.sort_values(["chatId", "timestamp"])
    first_msg = df_pred.groupby("chatId", as_index=False).head(1).copy()

    label_map = {
        "Direct Chat": "Direct Chat",
        "Short Form": "Short Form",
        "Long Form": "Long Form",
        "NotFirstMessage": "Not First Message",
    }
    first_msg["lead_type"] = first_msg["pred_lbl_chat"].map(label_map).fillna("Other")
    first_msg["day"] = first_msg["timestamp"].dt.date

    return df_pred, first_msg


# -------------------- Tables --------------------
def make_leads_summary(df_first: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_first.groupby(["displayName_ui", "lead_type"])["chatId"]
        .nunique()
        .reset_index(name="Leads")
    )

    pivot = (
        g.pivot_table(
            index=["displayName_ui"],
            columns="lead_type",
            values="Leads",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    expected = ["Direct Chat", "Short Form", "Long Form", "Not First Message"]
    for col in expected:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total Leads"] = pivot[expected].sum(axis=1)
    pivot = pivot[["displayName_ui"] + expected + ["Total Leads"]].sort_values("Total Leads", ascending=False)
    return pivot.rename(columns={"displayName_ui": "EC"})


def build_avg_reply_table_hierarchy(df_pred_filtered: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses summary_metrics.compute_resp_df hierarchy.
    Output:
      displayName, pred_lbl_chat, avg_reply_in_minutes
    Rows:
      Direct Chat, Short Form, Total (Direct + Short)
    """
    resp = compute_resp_df(df_pred_filtered)

    # attach displayName_ui
    resp = resp.merge(users_df[["userId", "displayName_ui"]], on="userId", how="left")
    resp["displayName_ui"] = resp["displayName_ui"].fillna(resp["userId"].astype(str))

    per_label = (
        resp.groupby(["displayName_ui", "pred_lbl_first"], as_index=False)["response_minutes"]
        .mean()
        .rename(
            columns={
                "pred_lbl_first": "pred_lbl_chat",
                "response_minutes": "avg_reply_in_minutes",
            }
        )
    )

    total = (
        resp.groupby(["displayName_ui"], as_index=False)["response_minutes"]
        .mean()
        .rename(columns={"response_minutes": "avg_reply_in_minutes"})
    )
    total["pred_lbl_chat"] = "Total (Direct + Short)"

    out = pd.concat([per_label, total], ignore_index=True)

    order_map = {"Direct Chat": 1, "Short Form": 2, "Total (Direct + Short)": 3}
    out["__order"] = out["pred_lbl_chat"].map(order_map).fillna(99)
    out = out.sort_values(["displayName_ui", "__order"]).drop(columns="__order")

    out["avg_reply_in_minutes"] = out["avg_reply_in_minutes"].round(2)
    out = out.rename(columns={"displayName_ui": "displayName"})
    return out[["displayName", "pred_lbl_chat", "avg_reply_in_minutes"]]


# -------------------- Charts --------------------
def chart_total_leads_by_day(df_first: pd.DataFrame):
    df = (
        df_first.groupby("day")["chatId"]
        .nunique()
        .reset_index(name="Total Leads")
        .sort_values("day")
    )

    c = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("Total Leads:Q", title="Total leads"),
            tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("Total Leads:Q", title="Leads")],
        )
        .properties(height=280)
    )
    st.altair_chart(c, use_container_width=True)


def chart_small_multiples_by_user(df_first: pd.DataFrame, selected_display_names, n_cols=2, max_users=4):
    selected_display_names = list(selected_display_names)[:max_users]

    df = (
        df_first.groupby(["day", "displayName_ui"])["chatId"]
        .nunique()
        .reset_index(name="Leads")
        .sort_values("day")
    )

    df = df[df["displayName_ui"].isin(selected_display_names)]

    rows = (len(selected_display_names) + n_cols - 1) // n_cols
    idx = 0

    for _ in range(rows):
        cols = st.columns(n_cols, gap="medium")
        for c in cols:
            if idx >= len(selected_display_names):
                break

            name = selected_display_names[idx]
            d = df[df["displayName_ui"] == name]

            with c.container(border=True, height="stretch"):
                st.markdown(f"### {name}")
                if d.empty:
                    st.caption("No data")
                else:
                    ch = (
                        alt.Chart(d)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("day:T", title="Date"),
                            y=alt.Y("Leads:Q", title="Leads"),
                            tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("Leads:Q", title="Leads")],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(ch, use_container_width=True)

            idx += 1


# -------------------- Small UI polish --------------------
def apply_small_css():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1200px; padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================== App ====================
st.set_page_config(page_title="CRM Leads Dashboard", page_icon="📊", layout="wide")
apply_small_css()

st.title("CRM Leads Dashboard")
st.write("")

with st.spinner("Loading data..."):
    df_pred_all, df_first_all = build_pred_and_first()

# ---- Filters ----
st.sidebar.header("Filters")

users_df_all = df_first_all[["userId", "displayName_ui"]].drop_duplicates().sort_values("displayName_ui")
display_names = users_df_all["displayName_ui"].tolist()
display_to_userid = dict(zip(users_df_all["displayName_ui"], users_df_all["userId"]))

labels = sorted(df_first_all["pred_lbl_chat"].dropna().unique().tolist())

max_day = df_first_all["day"].max()
min_day = df_first_all["day"].min()

auto_start = max(auto_sept1_start(max_day), min_day)

date_range = st.sidebar.date_input(
    "Date range",
    value=(auto_start, max_day),
    min_value=min_day,
    max_value=max_day,
)

if hasattr(st, "pills"):
    selected_users = st.sidebar.pills("List EC", options=display_names, default=display_names, selection_mode="multi")
else:
    selected_users = st.sidebar.multiselect("List EC", options=display_names, default=display_names)

if hasattr(st, "pills"):
    selected_labels = st.sidebar.pills("Labels", options=labels, default=labels, selection_mode="multi")
else:
    selected_labels = st.sidebar.multiselect("Labels", options=labels, default=labels)

if not selected_users:
    st.warning("Select at least 1 EC.")
    st.stop()

if not selected_labels:
    st.warning("Select at least 1 label.")
    st.stop()

selected_user_ids = [display_to_userid[n] for n in selected_users]
start_day, end_day = date_range if isinstance(date_range, (list, tuple)) else (min_day, max_day)

# Filter first messages (for leads)
filtered_first = df_first_all[
    (df_first_all["userId"].isin(selected_user_ids))
    & (df_first_all["pred_lbl_chat"].isin(selected_labels))
    & (df_first_all["day"] >= start_day)
    & (df_first_all["day"] <= end_day)
].copy()

# Filter full messages (for avg reply hierarchy)
filtered_msg = df_pred_all[
    (df_pred_all["userId"].isin(selected_user_ids))
    & (df_pred_all["timestamp"].dt.date >= start_day)
    & (df_pred_all["timestamp"].dt.date <= end_day)
].copy()

# This users_df matches the selection (for name mapping in avg table)
users_df_selected = users_df_all[users_df_all["userId"].isin(selected_user_ids)].copy()

# ---- KPI Row ----
st.subheader("Summary")
st.write("")

total_leads = int(filtered_first["chatId"].nunique()) if not filtered_first.empty else 0
total_ec = int(filtered_first["userId"].nunique()) if not filtered_first.empty else 0

latest_day = filtered_first["day"].max() if not filtered_first.empty else None
this_month_leads = 0
prev_month_leads = 0

if latest_day:
    m_start, m_end = month_bounds(latest_day)

    if m_start.month == 1:
        prev_start = date(m_start.year - 1, 12, 1)
    else:
        prev_start = date(m_start.year, m_start.month - 1, 1)
    prev_end = m_start

    this_month_leads = int(
        df_first_all[
            (df_first_all["day"] >= m_start)
            & (df_first_all["day"] < m_end)
            & (df_first_all["userId"].isin(selected_user_ids))
        ]["chatId"].nunique()
    )
    prev_month_leads = int(
        df_first_all[
            (df_first_all["day"] >= prev_start)
            & (df_first_all["day"] < prev_end)
            & (df_first_all["userId"].isin(selected_user_ids))
        ]["chatId"].nunique()
    )

delta_str = None
if prev_month_leads > 0:
    delta_pct = ((this_month_leads - prev_month_leads) / prev_month_leads) * 100.0
    delta_str = f"{delta_pct:+.1f}% vs last month"
elif this_month_leads > 0 and prev_month_leads == 0:
    delta_str = "+100.0% vs last month"

try:
    with st.container(horizontal=True, gap="medium"):
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.metric("Total Leads", f"{total_leads:,}")
        c2.metric("Total EC", f"{total_ec:,}")
        c3.metric("Total Leads This Month", f"{this_month_leads:,}", delta=delta_str)
except TypeError:
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Total Leads", f"{total_leads:,}")
    c2.metric("Total EC", f"{total_ec:,}")
    c3.metric("Total Leads This Month", f"{this_month_leads:,}", delta=delta_str)

st.write("")
st.write("")

# 2) Leads per User table
with st.container(border=True):
    st.markdown("### Leads per User")
    if filtered_first.empty:
        st.info("No data.")
    else:
        leads_table = make_leads_summary(filtered_first)
        st.dataframe(leads_table, use_container_width=True, hide_index=True)

st.write("")
st.write("")

# 3) Avg reply table (HIERARCHY)
with st.container(border=True):
    st.markdown("### Avg Reply (minutes)")
    st.caption("Hierarchy logic: if 2nd message is Chat Bot, reply uses 3rd message.")
    if filtered_msg.empty:
        st.info("No data.")
    else:
        avg_reply_table = build_avg_reply_table_hierarchy(filtered_msg, users_df_selected)
        st.dataframe(avg_reply_table, use_container_width=True, hide_index=True)

st.write("")
st.write("")

# 4) Total leads per day
with st.container(border=True):
    st.markdown("### Total Leads per Day")
    if filtered_first.empty:
        st.info("No data to plot.")
    else:
        chart_total_leads_by_day(filtered_first)

st.write("")
st.write("")

# 5) Total leads per day per user (2x2)
with st.container(border=True):
    st.markdown("### Total Leads per Day per EC (2×2)")
    st.caption("Showing up to 4 selected ECs.")
    if filtered_first.empty:
        st.info("No data to plot.")
    else:
        chart_small_multiples_by_user(filtered_first, selected_users, n_cols=2, max_users=4)

# 6) Raw first message rows table
with st.expander("Raw First Message Rows"):
    cols = ["displayName_ui", "chatName", "timestamp", "pred_lbl_chat", "lead_type", "body"]
    raw = filtered_first[cols].sort_values("timestamp").rename(columns={"displayName_ui": "EC"})
    st.dataframe(raw, use_container_width=True)
