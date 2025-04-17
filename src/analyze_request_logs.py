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


'''
 def get_renege_action_outcome(self, queue_id):
        curr_state = self.requestObj.get_queue_state(queue_id) # get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_obs()) > 0: # get_curr_obs_renege(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize - 1
                curr_state["reward"] = self.Observations.get_curr_obs_renege(srv)[0]['reward']
        else:
            if len(self.Observations.get_obs()) > 0: # get_curr_obs_renege(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize - 1
                curr_state["reward"] = self.Observations.get_curr_obs_renege[0]['reward']
                
        return curr_state
        

    def get_jockey_action_outcome(self, queue_id):
        curr_state = self.requestObj.get_queue_state(queue_id) # get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_obs()) > 0: #get_curr_obs_jockey(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize + 1
                curr_state["reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
        else:
            if len(self.Observations.get_obs()) > 0: # get_curr_obs_jockey(srv)) > 0:
                curr_state['at_pose'] = self.queuesize + 1
                curr_state["reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
                
        return curr_state

'''