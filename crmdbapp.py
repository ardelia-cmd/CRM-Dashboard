import streamlit as st
import pandas as pd

from db_utils import create_tunnel_and_engine
from classifier import train_classifier, predict_first_three
from summary_metrics import compute_resp_df, compute_final_summary

# change this to your real messages table name
MESSAGES_TABLE = '"Message"'  


@st.cache_resource
def get_engine():
    # one SSH tunnel + engine for the whole app
    engine, _ = create_tunnel_and_engine()
    return engine


@st.cache_resource
def get_classifier():
    # train model once and reuse
    clf = train_classifier()
    return clf


@st.cache_data
def get_final_summary():
    engine = get_engine()
    clf = get_classifier()

    # adjust columns and table name to match your Postgres schema
    df_msg = pd.read_sql(
        f"""
        SELECT 
            "userId",
            "chatId",
            "body",
            "type",
            "fromMe",
            "timestamp"
        FROM {MESSAGES_TABLE}
        """,
        engine,
    )

    # classifier step (adds pred_lbl_chat on first 3 messages per chat)
    df_test = predict_first_three(df_msg, clf)

    # summary step
    resp = compute_resp_df(df_test)
    final_summary = compute_final_summary(resp)
    return final_summary


# =============== Streamlit UI ===============

st.set_page_config(page_title="CRM Chat Response Dashboard", layout="wide")
st.title("CRM Chat Response Dashboard")

with st.spinner("Loading data and computing summary..."):
    try:
        final_summary = get_final_summary()
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        st.stop()

st.success("Data loaded")

st.subheader("Final summary table")
st.write(f"Total rows: **{len(final_summary)}**")

st.sidebar.header("Filters")

user_ids = final_summary["userId"].unique()
labels = final_summary["pred_lbl_first"].unique()

selected_users = st.sidebar.multiselect(
    "Select userId",
    options=user_ids,
    default=list(user_ids),
)

selected_labels = st.sidebar.multiselect(
    "Select first message label",
    options=labels,
    default=list(labels),
)

df_filtered = final_summary[
    final_summary["userId"].isin(selected_users)
    & final_summary["pred_lbl_first"].isin(selected_labels)
].copy()

st.dataframe(df_filtered, use_container_width=True)

st.subheader("Aggregated metrics")
col1, col2, col3 = st.columns(3)

if not df_filtered.empty:
    avg_response_global = df_filtered["avg_minutes"].mean()
    total_chats_global = df_filtered["count_chatId"].sum()
    total_users = df_filtered["userId"].nunique()

    col1.metric("Average response time (minutes)", f"{avg_response_global:.2f}")
    col2.metric("Total chat sessions", int(total_chats_global))
    col3.metric("Unique users", total_users)
else:
    st.info("No data for the current filters.")
