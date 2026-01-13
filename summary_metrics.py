# summary_metrics.py
import pandas as pd

CHATBOT_LABEL = "Chat Bot"


def add_msg_rank(df_labeled):
    df = df_labeled.copy()
    df = df.sort_values(["userId", "chatId", "timestamp"])
    df["msg_rank"] = df.groupby(["userId", "chatId"]).cumcount() + 1
    return df


def compute_resp_df(df_labeled):
    df_filtered = add_msg_rank(df_labeled)

    # keep only conversations where first msg is Direct Chat or Short Form
    first_rows = df_filtered[df_filtered["msg_rank"] == 1].copy()
    valid_first = first_rows[
        first_rows["pred_lbl_chat"].isin(["Direct Chat", "Short Form"])
    ]

    valid_pairs = valid_first[["userId", "chatId"]].drop_duplicates()
    df_filtered = df_filtered.merge(valid_pairs, on=["userId", "chatId"])

    max_rank = df_filtered.groupby(["userId", "chatId"])["msg_rank"].transform(
        "max"
    )
    df_filtered = df_filtered[max_rank >= 3].copy()

    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])

    def compute_response(group):
        g = group.sort_values("msg_rank")

        first = g[g["msg_rank"] == 1].iloc[0]
        second = g[g["msg_rank"] == 2].iloc[0]
        third = g[g["msg_rank"] == 3].iloc[0]

        ts_first = pd.to_datetime(first["timestamp"])

        if second["pred_lbl_chat"] == CHATBOT_LABEL:
            ts_reply = pd.to_datetime(third["timestamp"])
            reply_rank = 3
        else:
            ts_reply = pd.to_datetime(second["timestamp"])
            reply_rank = 2

        response_minutes = (ts_reply - ts_first).total_seconds() / 60.0

        return pd.Series(
            {
                "userId": first["userId"],
                "chatId": first["chatId"],
                "pred_lbl_first": first["pred_lbl_chat"],
                "second_label": second["pred_lbl_chat"],
                "reply_used_rank": reply_rank,
                "ts_first": ts_first,
                "ts_reply": ts_reply,
                "response_minutes": response_minutes,
            }
        )

    resp = (
        df_filtered.groupby(["userId", "chatId"])
        .apply(compute_response)
        .reset_index(drop=True)
    )

    return resp


def compute_final_summary(resp):
    agg_user_label = (
        resp.groupby(["userId", "pred_lbl_first"])
        .agg(
            total_minutes=("response_minutes", "sum"),
            avg_minutes=("response_minutes", "mean"),
            count_chatId=("chatId", "count"),
        )
        .reset_index()
    )

    agg_user_overall = (
        resp.groupby("userId")
        .agg(
            total_minutes=("response_minutes", "sum"),
            avg_minutes=("response_minutes", "mean"),
            count_chatId=("chatId", "count"),
        )
        .reset_index()
    )
    agg_user_overall["pred_lbl_first"] = "Overall"

    final_summary = pd.concat(
        [agg_user_label, agg_user_overall], ignore_index=True
    )
    final_summary = final_summary.sort_values(
        ["userId", "pred_lbl_first"]
    ).reset_index(drop=True)

    return final_summary
