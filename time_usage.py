import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df =pd.read_csv("usage_data_csv_utf8.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
students = df[df["user_id"] != "AI_TUTOR"]
students["date"] = students["timestamp"].dt.date
daily_usage = students.groupby("date").size()

# print(daily_usage)

plt.figure(figsize=(12,6))
daily_usage.plot(kind="line")

plt.xlabel("Date")
plt.ylabel("Number of Student Messages")
plt.title("Student Usage Per Day Over Time")

#to make the date labels more frequent in the x-axis, uncomment the next two lines
# ax= plt.gca()
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
