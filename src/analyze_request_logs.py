import pandas as pd
import matplotlib.pyplot as plt
import re

# Log file path
LOG_FILE = "request_decisions.log"

# Function to parse log file and extract relevant data
def parse_log_file(log_file):
    data = {"Timestamp": [], "RequestID": [], "QueueID": [], "Action": []}

    with open(log_file, "r") as file:
        for line in file:
            match = re.search(r"(\d+-\d+-\d+ \d+:\d+:\d+,\d+) - Request ID: (\d+) - Queue ID: (\d+) - Action: (.*)", line)
            if match:
                timestamp, request_id, queue_id, action = match.groups()
                data["Timestamp"].append(pd.to_datetime(timestamp))
                data["RequestID"].append(int(request_id))
                data["QueueID"].append(int(queue_id))
                data["Action"].append(action)

    return pd.DataFrame(data)

# Load data from logs
df = parse_log_file(LOG_FILE)

# Count occurrences of Jockeying and Reneging
action_counts = df["Action"].value_counts()

# Plot Jockeying vs. Reneging Rate
plt.figure(figsize=(10, 5))
action_counts.plot(kind="bar", color=["blue", "red", "green"])
plt.xlabel("Request Action")
plt.ylabel("Count")
plt.title("Jockeying vs. Reneging Rate")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")
plt.show()

# Plot Requests Per Queue Over Time
plt.figure(figsize=(12, 5))
for queue_id in df["QueueID"].unique():
    queue_data = df[df["QueueID"] == queue_id].groupby("Timestamp").count()["RequestID"]
    plt.plot(queue_data, label=f"Queue {queue_id}")

plt.xlabel("Time")
plt.ylabel("Number of Requests")
plt.title("Requests Per Queue Over Time")
plt.legend()
plt.grid()
plt.show()
