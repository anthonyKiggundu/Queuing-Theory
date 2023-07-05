
` ******************************************`  
`         The Jockeying Simulator`  
` ******************************************`  

The snippet simulates a simple M/M/C queuing systems setup, C=2 (two parallel queues).
We have unfortunately not tested the script on legacy python interpreters, i.e. < Python3.6

We have added a wrapper script (resources directory)that can be used to simulate multiple runs at a go.
Under the same directory you will also find a sample configuration file with some defaults.

In the case that this wrapper or the config file is not used, the script can be run using a command like:

`python3 ./of_latest.py --service_rate_one $_serv_rate_one 
                       --service_rate_two $_serv_rate_two 
                       --arrival_rate $arrival_rate 
                       --jockeying_threshold $jockey_thresholds 
                       --run_id ${spinner}`

The `jockeying_threshold` parameter is optional, such that in absence of this parameter
the jockeying behaviour is triggered based on the waiting time.

- For pull requests please send an email to: antonkingdee@yahoo.com

Please cite the tooling in the an experimental use-cases as:

`article{joc_simulator,
author = {Anthony Kiggundu, Bin Han, Dennis Krummacher, Hans-D Schotten},  
title = {On Resource Allocation in Communication Networks: A Decision Model of the Jockeying Behavior in Queues},  
journal = {},  
pages = {},  
volume = {},  
month = {December},  
year = {2023},  
doi = {}  
}` 

