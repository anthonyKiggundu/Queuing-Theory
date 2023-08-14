#!/usr/bin/python3
# Script to setup the necessary files with their respective headers for the columns

import pandas as pd

file_queue_one = "srv_one_pose_waits_stats.csv"
file_queue_two = "srv_two_pose_waits_stats.csv"

# file = pd.read_csv(file_queue_one)
  
# adding header
headerList = ['Position', 'Waiting']

import csv

# open CSV file and assign header
with open(file_queue_one, 'w') as file:
    dw = csv.DictWriter(file, delimiter=',',
                        fieldnames=headerList)
    dw.writeheader()


with open(file_queue_two, 'w') as file:
    dw = csv.DictWriter(file, delimiter=',',
                        fieldnames=headerList)
    dw.writeheader()

  
# converting data frame to csv
# file.to_csv(file_queue_one, header=headerList, index=False)

# file = pd.read_csv(file_queue_two)
# file.to_csv(file_queue_one, header=headerList, index=False)


