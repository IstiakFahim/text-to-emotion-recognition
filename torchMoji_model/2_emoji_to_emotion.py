import pandas as pd
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# === 1️⃣ Define the mapping dictionary ===
label_map = {
    # sadness
    1: (":unamused:", "sadness"),
    2: (":weary:", "sadness"),
    3: (":sob:", "sadness"),
    5: (":pensive:", "sadness"),
    14: (":sleeping:", "sadness"),
    19: (":expressionless:", "sadness"),
    25: (":neutral_face:", "sadness"),
    27: (":disappointed:", "sadness"),
    29: (":tired_face:", "sadness"),
    34: (":cry:", "sadness"),
    35: (":sleepy:", "sadness"),
    43: (":persevere:", "sadness"),
    45: (":sweat:", "sadness"),
    46: (":broken_heart:", "sadness"),
    52: (":confounded:", "sadness"),

    # joy
    0: (":joy:", "joy"),
    6: (":ok_hand:", "joy"),
    7: (":blush:", "joy"),
    10: (":grin:", "joy"),
    11: (":notes:", "joy"),
    15: (":relieved:", "joy"),
    16: (":relaxed:", "joy"),
    17: (":raised_hands:", "joy"),
    20: (":sweat_smile:", "joy"),
    30: (":v:", "joy"),
    31: (":sunglasses:", "joy"),
    33: (":thumbsup:", "joy"),
    36: (":stuck_out_tongue_winking_eye:", "joy"),
    40: (":clap:", "joy"),
    41: (":eyes:", "joy"),
    50: (":wink:", "joy"),
    53: (":smile:", "joy"),
    54: (":stuck_out_tongue_winking_eye:", "joy"),
    57: (":muscle:", "joy"),
    58: (":punch:", "joy"),
    63: (":sparkles:", "joy"),

    # anger
    32: (":rage:", "anger"),
    37: (":triumph:", "anger"),
    44: (":imp:", "anger"),
    55: (":angry:", "anger"),
    56: (":no_good:", "anger"),

    # fear
    12: (":flushed:", "fear"),
    28: (":see_no_evil:", "fear"),
    39: (":mask:", "fear"),
    49: (":speak_no_evil:", "fear"),
    51: (":skull:", "fear"),
    62: (":grimacing:", "fear"),

    # love
    4: (":heart_eyes:", "love"),
    8: (":heart:", "love"),
    13: (":100:", "love"),
    18: (":two_hearts:", "love"),
    23: (":kissing_heart:", "love"),
    24: (":hearts:", "love"),
    47: (":blue_heart:", "love"),
    59: (":purple_heart:", "love"),
    60: (":sparkling_heart:", "love"),

    # surprise
    22: (":confused:", "surprise"),
    26: (":information_desk_person:", "surprise"),
    38: (":raised_hand:", "surprise"),
    42: (":gun:", "surprise"),
    48: (":headphones:", "surprise"),
}


df = pd.read_csv("predictions.csv")

df["TrueEmoji"] = df["TrueLabel"].map(lambda x: label_map.get(x, ("Unknown", "Unknown"))[0])
df["TrueEmotion"] = df["TrueLabel"].map(lambda x: label_map.get(x, ("Unknown", "Unknown"))[1])

df["PredEmoji"] = df["PredLabel"].map(lambda x: label_map.get(x, ("Unknown", "Unknown"))[0])
df["PredEmotion"] = df["PredLabel"].map(lambda x: label_map.get(x, ("Unknown", "Unknown"))[1])

if "TrueEmotion" not in df.columns or "PredEmotion" not in df.columns:
    raise ValueError("Missing 'TrueEmotion' or 'PredEmotion' columns in the CSV!")


df["TrueEmotion"] = df["TrueEmotion"].astype(str).str.strip().str.lower()
df["PredEmotion"] = df["PredEmotion"].astype(str).str.strip().str.lower()

report = classification_report(df["TrueEmotion"], df["PredEmotion"], digits=3)
print("\n=== Classification Report (Emotion Level) ===")
print(report)

emotions = sorted(list(set(df["TrueEmotion"]) | set(df["PredEmotion"])))
cm = confusion_matrix(df["TrueEmotion"], df["PredEmotion"], labels=emotions)

cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
print("\n=== Confusion Matrix ===")
print(cm_df)


cm_df.to_csv("confusion_matrix.csv", index=True)
print("\nConfusion matrix saved as 'confusion_matrix.csv'")