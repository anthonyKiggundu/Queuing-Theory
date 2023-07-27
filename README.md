
` ******************************************`  
`         The Jockeying Simulator`  
` ******************************************`  
![queue_other](https://github.com/anthonyKiggundu/Queuing-Theory/assets/12003998/ff6e428e-5871-49df-b957-a3cf643a4f2d)

**Dependencies**
- Python 3.6
- Plus other corresponding libraries as listed in the import section

**Get Started**
- The snippet simulates a simple M/M/C queuing systems setup, C=2 (two parallel queues).
We have unfortunately not tested the script on legacy python interpreters, i.e. < Python3.6

- We have added a wrapper script (resources directory)that can be used to simulate multiple runs at a go.
Under the same directory you will also find a sample configuration file with some defaults.

- In the case that this wrapper or the config file is not used, the script can be run using a command like:

`python3 ./of_latest.py --service_rate_one $_serv_rate_one 
                       --service_rate_two $_serv_rate_two 
                       --arrival_rate $arrival_rate 
                       --jockeying_threshold $jockey_thresholds 
                       --run_id ${spinner}`

- The `jockeying_threshold` parameter is optional, such that in absence of this parameter
the jockeying behaviour is triggered based on the waiting time.

**Notes**
- 

**Contact**
- For pull requests please send an email to:

**Acknowledgements**
- This work was done under the auspice of the Open6GHub Project.

**Citations**
- Please use the `Cite this repository` link on the right pane in case you if you intend to cite the tooling for experimental use-cases.  

**License**




