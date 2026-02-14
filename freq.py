import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("usage_data_csv_utf8.csv")

ai = df[df["user_id"] == "AI_TUTOR"]

ai["referenced_sources"] = ai["referenced_sources"].str.replace(r"[\[\]]", "", regex =True)

refs = ai["referenced_sources"].str.split(",").explode()
refs = refs.dropna().str.strip()
refs = refs[refs!=""]

ref_counts = refs.value_counts().sort_values(ascending=False)
# print(ref_counts)

plt.figure(figsize=(12,6))
ref_counts.plot(kind="bar")
plt.xlabel("Reference ID")
plt.ylabel("Frequency")
plt.title("Frequency of Reference ID Usage")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()