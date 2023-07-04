#!/bin/bash

jockey_thresholds=1
MAXCOUNT=2

spinner=1

while [ "$spinner" -le $1 ]; 
do
    count=1
    while [ "$count" -le $MAXCOUNT ]; do
       number[$count]=$(( ($RANDOM % 4) + 1 ))
       let "count += 1"
    done

    arrival_rate=7 #${number[2]}
    # deltalambda controls the difference between the service rate of either queues
    
    deltaLambda=${number[1]}
    serv_rate_one=$(( $arrival_rate + $deltaLambda ))
    serv_rate_two=$(($arrival_rate - $deltaLambda))

    _serv_rate_one=$( echo "scale=2;$serv_rate_one  / 2" | bc -l )
    _serv_rate_two=$(echo "scale=2;$serv_rate_two  / 2" | bc -l)


    #if [ ${_serv_rate_one%%.*} -gt 0.0 ] && [ ${_serv_rate_two%%.*} -gt 0.0 ]
    #then
       python3 ./of_latest.py --service_rate_one $_serv_rate_one --service_rate_two $_serv_rate_two --arrival_rate $arrival_rate --jockeying_threshold $jockey_thresholds --run_id ${spinner};
    #fi

    let "spinner += 1"
    
done
# Dump the simulation parameters to a log file
echo -e "${spinner}\n" >> /home/dfkiterminal/Queuing_technology/constant/simulation_runs_count.txt

# done

