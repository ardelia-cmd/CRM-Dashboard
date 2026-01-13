import pandas as pd
from classifier import train_classifier, predict_first_three

df = pd.DataFrame([
    {
        "id": "false_12048803128@c.us_AC7876E9E636758A22D3F41D36720CB8",
        "userId": "cmipr8nxe0bp91cyd7ixhqox7",
        "chatId": "12048803128@c.us",
        "body": "Hi, I’m interested in enrolling my teen in AWMUN XIII, Seoul, South Korea. Could you share the proposal with me?",
        "type": "text",
        "fromMe": 0,
        "timestamp": "2025-11-30 22:58:50"
    },
    {
        "id": "true_12048803128@c.us_A5DB19D8E056799AFC464F17CD64FABA",
        "userId": "cmipr8nxe0bp91cyd7ixhqox7",
        "chatId": "12048803128@c.us",
        "body": """Thank you for your message! I’ll reply to you shortly as i’m currently assisting hundreds of delegates from around the world. I appreciate your patience 🤗🙌""",
        "type": "text",
        "fromMe": 1,
        "timestamp": "2025-11-30 22:58:56"
    },
    {
        "id": "true_12048803128@c.us_3EB0B6344E7383015F94B4",
        "userId": "cmipr8nxe0bp91cyd7ixhqox7",
        "chatId": "12048803128@c.us",
        "body": """Hi Ma'am/Sir! I'm Rahma, an education consultant for AWMUN...""",
        "type": "text",
        "fromMe": 1,
        "timestamp": "2025-12-01 03:43:14"
    },
    {
        "id": "false_16477406887@c.us_3AC7021E763744E58732",
        "userId": "cmipr8nxe0bp91cyd7ixhqox7",
        "chatId": "16477406887@c.us",
        "body": """Hi, I’m interested in enrolling my teen in AYIMUN 20th, Kuala Lumpur. Could you share the proposal with me?""",
        "type": "text",
        "fromMe": 0,
        "timestamp": "2025-10-29 21:41:45"
    }
])

clf = train_classifier()
df_test = predict_first_three(df, clf)
print(df_test)
