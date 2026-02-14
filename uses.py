import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('usage_data_csv_utf8.csv')

students = df[df["user_id"] != "AI_TUTOR"]

usage_counts = students.groupby("user_id").size()

plt.hist(usage_counts, bins=10)
plt.xlabel("Number of Messages sent")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Usage")
plt.show()