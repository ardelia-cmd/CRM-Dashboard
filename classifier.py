# classifier.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# paths inside your project
TRAIN_JSON_PATH = "label_tr.json"            # copy this from Drive into your repo
STOPWORDS_CSV_PATH = "combined_stopwords.csv"

TARGET_USERS = [
    "cmipr8nxe0bp91cyd7ixhqox7",
    "cmipzr6wr0onm1cydtxhycutj",
    "cmis716m3000lcoydoe9botom",
    "cmis72epf000mcoydwsrzh6sn",
]


def load_stopwords(path: str = STOPWORDS_CSV_PATH):
    """Load combined_stopwords from CSV."""
    df = pd.read_csv(path)
    # assume there is one column with the stopwords
    col = df.columns[0]
    return df[col].dropna().astype(str).tolist()


def clean_fromMe(col: pd.Series) -> pd.Series:
    return (
        col.replace(
            {
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                1: True,
                0: False,
            }
        )
        .astype(bool)
        .astype(int)
    )


def train_classifier(
    train_json_path: str = TRAIN_JSON_PATH,
    stopwords_path: str = STOPWORDS_CSV_PATH,
) -> Pipeline:
    """Train the classifier on label_tr.json."""
    df_train = pd.read_json(train_json_path)

    combined_stopwords = load_stopwords(stopwords_path)

    df_train = df_train.copy()
    df_train["fromMe"] = clean_fromMe(df_train["fromMe"])

    X_train = df_train[["body", "fromMe"]]
    y_train = df_train["Label Chat"].astype(str)

    pp = ColumnTransformer(
        transformers=[
            (
                "body",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words=combined_stopwords,
                ),
                "body",
            ),
            ("fromMe", "passthrough", ["fromMe"]),
        ]
    )

    clf = Pipeline(
        [
            ("preproc", pp),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    clf.fit(X_train, y_train)
    return clf


def predict_first_three(
    df_msg: pd.DataFrame,
    clf: Pipeline,
    target_users=None,
) -> pd.DataFrame:
    """
    Filter df_msg, keep first 3 messages per (userId, chatId),
    and add pred_lbl_chat column.
    """
    if target_users is None:
        target_users = TARGET_USERS

    df_msg = df_msg.copy()

    # filter to target users and text messages
    df_msg = df_msg[df_msg["userId"].isin(target_users)].copy()
    df_msg["type"] = df_msg["type"].astype(str).str.lower()
    df_msg = df_msg[df_msg["type"] == "text"].copy()

    df_msg["timestamp"] = pd.to_datetime(df_msg["timestamp"])

    # take first 3 messages per (userId, chatId)
    df_test = (
        df_msg.sort_values(["userId", "chatId", "timestamp"])
        .groupby(["userId", "chatId"])
        .head(3)
        .fillna("")
        .copy()
    )

    df_test["fromMe"] = clean_fromMe(df_test["fromMe"])

    # predict
    X_test = df_test[["body", "fromMe"]]
    prob = clf.predict_proba(X_test)
    cls = clf.classes_

    max_proba = prob.max(axis=1)
    best_idx = prob.argmax(axis=1)
    raw_pred = cls[best_idx]

    otherpred = np.where(max_proba > 0.45, raw_pred, "NotFirstMessage")
    df_test["pred_lbl_chat"] = otherpred

    return df_test
