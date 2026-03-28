import glob
import os
import pandas as pd
import numpy as np

files = sorted(glob.glob("data/*.csv"))
dfs = []
for f in files:
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["__source_file__"] = os.path.basename(f)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df["Label"] = df["Label"].astype(str).str.strip()

cols = [
    "Protocol",
    "Flow Packets/s",
    "Flow Bytes/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Flow Duration",
    "Label",
]

print("Available columns:")
for c in cols:
    if c in df.columns:
        print(c)

print("\nLabel counts:")
print(df["Label"].value_counts())

print("\nProtocol counts:")
print(df["Protocol"].value_counts(dropna=False).head(20))

num_cols = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Flow Duration",
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

attack_mask = df["Label"].str.upper() != "BENIGN"
benign = df.loc[~attack_mask]
attack = df.loc[attack_mask]

for c in num_cols:
    if c in df.columns:
        print(f"\n=== {c} ===")
        print("BENIGN percentiles:")
        print(benign[c].quantile([0.5, 0.9, 0.95, 0.99, 0.995, 0.999]))
        print("ATTACK percentiles:")
        print(attack[c].quantile([0.5, 0.9, 0.95, 0.99, 0.995, 0.999]))

if "Protocol" in df.columns:
    proto_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    proto_num = pd.to_numeric(df["Protocol"], errors="coerce")
    df["ProtocolName"] = proto_num.map(proto_map).fillna(df["Protocol"].astype(str))

    print("\nProtocol x Label:")
    print(pd.crosstab(df["ProtocolName"], df["Label"]))