#!/usr/bin/python3
'''
M/M/C queue where C=2(simulates two servers providing resources to incoming customers)
and at the beignning the selection of a queue to join is random assuming the customer
does not posses prior knowledge about the queue.
Then, jockeying behaviour based on the difference between the queues with the customer
jockeying to the shorter queue if it is not the current queue.

ⓒ 2022 Anthony L. Kiggundu - Open6GHub, German Research Center for Aritificial Intelligence [DFKI]
The code can be possibly extended to have more servers[c=num] for more queuing capacity
'''

from __future__ import print_function
from termcolor import colored
from queue import PriorityQueue
from textwrap import dedent
from collections import defaultdict
from json import dumps
from collections.abc import Iterable
from pandas._libs.tslibs.timestamps import Timestamp
from pathlib import Path
from os import path
from json import dumps, loads
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from tqdm import tqdm

import atexit
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import logging
import signal
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import argparse
import csv
import unittest
import ast
import pickle

arr_rate = 2 #2 #5  3
srv_rate = 1
time_init = 0
global_count = 1

global_jockeying_list_in_one = []
global_jockeying_list_in_two = []
global_arrivals_in_one = []
global_arrivals_in_two = []
lst_done_processed = []

processed_in_one = []
processed_in_two = []

jockeyed_waits_one = {}
jockeyed_waits_two = {}
srv_one_cust_pose_dict = {}
srv_two_cust_pose_dict = {}
srv_one_cust_time_dict = {}
srv_two_cust_time_dict = {}

srv_one_pose_time_dict = {}
srv_two_pose_time_dict = {}

count_processed_in_one = 0
count_processed_in_two = 0

flag_no_jockeys = False

enqueued_in_srv_one_cust_times = {}
enqueued_in_srv_two_cust_times = {}
jockeyed_customer_server_one_dict = {}
jockeyed_customer_server_two_dict = {}
customer_processes_in_one = {}
customer_processes_in_two = {}
dict_server_customer_queue_two = {}
dict_server_customer_queue_one = {}
jockeying_rates_with_threshold_in_one = {}
jockeying_rates_with_threshold_in_two = {}
jockeyed_pose_plus_wait_time_queue_one = []
jockeyed_pose_plus_wait_time_queue_two = [] 
jockeyed_pose_plus_wait_time_queue_one_etxra = []
jockeyed_pose_plus_wait_time_queue_two_extra = []
 

lst_jockeyed_pose_wait_time_queue_one = []
lst_jockeyed_pose_wait_time_queue_two = [] 
lst_wait_threshold_diff_serv_rates_one = []
lst_wait_threshold_diff_serv_rates_two = []

# variables to capture waiting times of the jockeyed tenants
jockeyed_in_one_processing_time = {}
jockeyed_in_two_processing_time = {}

base_dir = os.path.dirname(os.path.abspath(__file__))
serv_rates_jockeying_file_one = base_dir+'/constant/queue_one_serv_rates_jockeying_rates_stats.csv'
serv_rates_jockeying_file_two = base_dir+'/constant/queue_two_serv_rates_jockeying_rates_stats.csv'
jockey_stats_file_one = base_dir+'/constant/srv_one_jockeying_stats.csv'
jockey_stats_file_two = base_dir+'/constant/srv_two_jockeying_stats.csv'
que_length_jockey_stats_one = base_dir+'/constant/que_one_length_jockey_rates.csv'
que_length_jockey_stats_two = base_dir+'/constant/que_two_length_jockey_rates.csv'
jockey_queu_length_waiting_time_one = base_dir+'/constant/que_one_length_waits.csv'
jockey_queu_length_waiting_time_two = base_dir+'/constant/que_two_length_waits.csv'
service_rates_difference_file = base_dir+'/constant/diff_service_rates.txt'
jockey_details_source_one = base_dir+'/constant/que_one_jockeying_details.csv'
jockey_details_source_two = base_dir+'/constant/que_two_jockeying_details.csv'

time_spent_in_pose_queue_one = base_dir+'/constant/que_one_pose_wait.csv'
time_spent_in_pose_queue_two = base_dir+'/constant/que_two_pose_wait.csv'

def init_logger():
    logger_format = "%(asctime)s %(levelname)-8.8s [%(funcName)24s():%(lineno)-3s] %(message)s"
    formatter = logging.Formatter(logger_format)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = init_logger()
logging.getLogger('matplotlib.font_manager').disabled = True

class TestSum(unittest.TestCase):
    '''
       TODOs::
       - Tests for the commandline inputs (present or not, their data-types)
       - Tests for function return types (ensure no None types or empty returns)
       - Tests for files read and written to
       - Tests for process terminations
    '''

    def test_list_int(self):
        """
          Test that it can sum a list of integers
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


class ProcessHandler(object):
    def __init__(self):
        #self.event = event
        # main sigint/sigterm handlers
        self.main_stop_event = multiprocessing.Event()
        self.worker_stop_event = multiprocessing.Event()

    def main_sigint_handler(self, signum, frame):
        logger.debug('Parent: Termination request received!')
        self.main_stop_event.set()

    def main_sigterm_handler(self, signum, frame):
        logger.debug('Parent: Termination request received!')
        self.main_stop_event.set()


    # children sigint/sigterm handlers, let main process handle this
    def children_sigint_handler(self, signum, frame):
        logger.debug('Child: Termination request received!')

    def children_sigterm_handler(self, signum, frame):
        logger.debug('Child: Termination request received!')

    def terminate_process(self, proc_to_terminate, queue_tasks):
        self.worker_stop_sent = False
        self.proc_to_terminate = proc_to_terminate

        max_allowed_shutdown_seconds = 10
        shutdown_et = None

        # children: capture sigint/sigterm
        signal.signal(signal.SIGINT, self.children_sigint_handler)
        signal.signal(signal.SIGTERM, self.children_sigterm_handler)
        # main: capture sigint/sigterm
        signal.signal(signal.SIGINT, self.main_sigint_handler)
        signal.signal(signal.SIGTERM, self.main_sigterm_handler)

                # send 'stop' to worker()?
        if self.main_stop_event.is_set() and not self.worker_stop_sent:
            logger.debug('Sending stop to worker ...')
            self.worker_stop_event.set()
            self.worker_stop_sent = True
            shutdown_et = int(time.time()) + max_allowed_shutdown_seconds

        try:
            print(colored('Terminating jockeyed {}','blue').format(colored(proc_to_terminate,'red')) + colored(' process in current queue ...','blue'))
            self.proc_to_terminate.terminate()
            self.proc_to_terminate.kill()
            new_size_queue = queue_tasks.qsize() - 1
            #print("Decrementing task count in queue after terminating. Current task count in queue now is %01d"%(new_size_queue))
        except Exception as e:
            logger.debug('Exception {}, e = {}'.format(type(e).__name__, e))


def save_threshold_timetoservice_queue(dict_thresh_time_in_service, filename):
    try:        
        df = pd.DataFrame(dict_thresh_time_in_service, columns = ["Jockey_Threshold","Total_Waiting"])
        # print("************** ", df.columns, filename)
        df.to_csv (base_dir+'/constant/'+filename, mode='a',index=False, header=False)
    except OSError:
        if not filename:
            print("Could not open/read file:", filename)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

def save_pose_waits_to_file_one(srv_one_pose_wait_dict):

    try:
        filename = base_dir+'srv_one_pose_waits_stats.csv'
        df = pd.DataFrame(srv_one_pose_wait_dict, columns = ["Position","Waiting"])
        df.to_csv (base_dir+'/constant/srv_one_pose_waits_stats.csv', mode='a',index=False, header=False)
    except OSError:
        if not filename:
            print("Could not open/read file:", filename)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


def save_serv_rate_jockeying_rate_in_queues(data_source, queue_serv_rate, jockeying_rate):
    service_jockeying_rates_dict = {}
    service_jockeying_rates_dict.update({queue_serv_rate:jockeying_rate})

    try:

        header = ["Service_rate","Jockeying_rate"]
        with open(data_source, 'a', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            for k,v in service_jockeying_rates_dict.items():
                writer.writerow([k,v])

        f.close()

    except OSError:
        if not data_source:
            print("Could not open/read file:", data_source)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
        

def save_pose_waits_to_file_two(srv_two_pose_wait_dict):

    try:
        filename = base_dir+'/constant/srv_two_pose_waits_stats.csv'
        df = pd.DataFrame(srv_two_pose_wait_dict, columns = ["Position","Waiting"])
        df.to_csv (filename, mode='a',index=False, header=False)
    except OSError:
        if not fielname:
            print("Could not open/read file:", filename)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise



def save_jockey_details_to_file(jockeying_rates_with_threshold_in_server, queue_name):
    if queue_name == "Server1":
        try:
            filename = base_dir+'/constant/constant/srv_one_jockeying_stats.csv'

            header = ["Jockeying_threshold","Jockeying_rate"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in jockeying_rates_with_threshold_in_server.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not filename:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    else:
        try:
            filename = base_dir+'/constant/srv_two_jockeying_stats.csv'

            header = ["Jockeying_threshold","Jockeying_rate"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in jockeying_rates_with_threshold_in_server.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not fielname:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

def save_pose_waits_to_file(pose_waits_dict, queue_name):
    if queue_name == "Server1":
        try:
            filename = time_spent_in_pose_queue_one # base_dir+'/constant/constant/srv_one_pose_waits.csv'

            header = ["Position","Time in Position"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in pose_waits_dict.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not filename:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    else:
        try:
            filename = time_spent_in_pose_queue_two #base_dir+'/constant/srv_two_pose_waits.csv'

            header = ["Position","Time in Position"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in pose_waits_dict.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not fielname:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


def read_pose_waiting_times_in_queue_one( pose_in_queue, curr_queue, datasource, serv_time):
    index = {}
    try:
        with open(datasource) as f:
            cr = csv.reader(f)
            try:
                next(cr) # skip header row
            except StopIteration: 
                return
                
            for row in cr:
                index.setdefault(row[0], []).append(row[1])

        for c, v in index.items():
            total = 0.0
            for k in v:
                if k != "":
                    total = total + float(k.replace("'",""))

            if int(c) == pose_in_queue:
                return float(total / len(v))
            else:
                srv_avg_waiting_time_at_pose = get_expected_by_littles_law(len(curr_queue.queue), serv_time)
                return srv_avg_waiting_time_at_pose

    except OSError:
        print("Could not open/read file:", datasource)
        sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


def read_pose_waiting_times_in_queue_two( pose_in_queue, curr_queue, datasource, serv_time):
    index = {}
    try:
        with open(datasource) as f:
            cr = csv.reader(f)
            # if not cr[0]:
            try:
                next(cr) # skip header row
            except StopIteration: 
                return
                
            for row in cr:
                index.setdefault(row[0], []).append(row[1])

        for c, v in index.items():
            total = 0.0
            for k in v:
                if k != "":
                    total = total + float(k.replace("'",""))

            if int(c) == pose_in_queue:
                return float(total / len(v))
            else:
                srv_avg_waiting_time_at_pose = get_expected_by_littles_law(len(curr_queue.queue), serv_time)
                return srv_avg_waiting_time_at_pose

    except OSError:
        print ("Could not open/read file:", datasource)
        sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise



def generate_arrivals(num, arrivals_batch_num, run):
    customers = []
    for i in range(num):
        customer = "Run%01d_Batch%01d"%(run, arrivals_batch_num)+"_"+"Customer%01d"%(i+1)
        customers.append( customer)

    return customers


def get_process_id_to_terminate(customer_id, srv_one_processes, srv_two_processes):
    #customer_id = list(customer_id.values())[0]

    for t in srv_one_processes.items():        
        if t[0] == customer_id:
            process_to_terminate = t[1]
            print(colored("%s", 'green') % (customer_id) + " jockeying now, terminating process %s" % (
                process_to_terminate))
            return process_to_terminate

    for j in srv_two_processes.items():
        if j[0] == customer_id:
            process_to_terminate = j[1]
            print(colored("%s", 'green') % (customer_id) + " jockeying now, terminating process %s" % (
                process_to_terminate))
            return process_to_terminate

    # process_to_terminate =  srv_process_details.get(customer_id)
    # print(colored("%s", 'green') % (customer_id) + " is going to terminate with process %s" % (process_to_terminate))


def get_expected_customer_queue_waiting_times(datasource, pose, queue_size): #curr_queue):
    
    try:
        df = pd.read_csv(datasource, sep=',')
        columns = list(df.columns.values)  
    
        col_position = columns[0]
        col_waiting_time = columns[1]      
            
        grp_fields = df.groupby(col_position).agg(mean_pose=('Position', 'mean'), mean_waiting_time=('Waiting','mean')).reset_index(drop=True)
        # print(grp_fields.mean_pose.tolist(), "\n",grp_fields.mean_waiting_time.tolist(), pose)
        
        count = 0
        for field in grp_fields.mean_pose: #range(len(grp_fields.mean_pose)):        
            if float(pose) == field: #grp_fields.mean_pose[count]:
                expected_wait = grp_fields.mean_waiting_time[count]
                # print("-------<< ", float(pose), expected_wait)
                return expected_wait
                continue
            else:
                if args.arrival_rate > 0:
                    # print("===>>>> ", curr_queue.qsize(), args.arrival_rate)
                    expected_wait = get_expected_by_littles_law(pose, args.arrival_rate) # queue_size,
                    # print("******** ",float(pose),  expected_wait)
                    return expected_wait
                    continue
                else:
                    continue
            count = count + 1
    except OSError:
        print ("Could not open/read file:", datasource)
        sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    
    # return expected_wait
    
    '''
    index = {}
    try:
        print("SHOW ME STUFF HERE ====>> ", curr_queue.qsize(), args.arrival_rate)
        with open(datasource) as f:
            cr = csv.reader(f)
            try:
                next(cr) # skip header row
            except StopIteration:                
                expected_wait = get_expected_by_littles_law(curr_queue, args.arrival_rate) #serv_time)
                return expected_wait
                
            for row in cr:
                index.setdefault(row[0], []).append(row[1])

        for c, v in index.items():
            total = 0.0
            for k in v:
                if k != "":
                    total = total + float(k.replace("'",""))

            if int(c) == pose:
                expected_wait = float(total / len(v))
            else:
                # srv_avg_waiting_time_at_pose = get_expected_by_littles_law(curr_queue, serv_time)
                expected_wait = get_expected_by_littles_law(curr_queue, args.arrival_rate) #serv_time)
                # return srv_avg_waiting_time_at_pose
                return expected_wait

    except OSError:
        print ("Could not open/read file:", datasource)
        sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    
    '''
        
    # return expected_wait

def compute_jockey_wait_in_pose(customer_id, queue_size, preferred_que_name, queue_name, srv_times_list, all_serv_times): #, unused_waiting_times):
    
    '''
      TODO::
        Algorithm is to get the unused waiting times and assign one of these
        to the jockeyed. That is, given the jockey lands at the end of the queue,
        first get queue size, then get the respective waiting times for each customer
        and add this to the waiting time assigned to the jockey - this is the total time the
        jockey will spend in the queue when it lands the the end.

    '''
    # now_in_preferred_queue = set(list())  # Need to bring in the preferred_queue object here to
    
    get_customer_queue_waiting_times(preferred_queue)

    if preferred_que_name == "Server1":
        if queue_size > len(srv_times_list):
            wait_in_pose = srv_times_list[0]
        else:
            wait_in_pose = compute_time_spent_in_service(srv_times_list, queue_size, all_serv_times)
    else:
        if queue_size > len(srv_times_list):
            wait_in_pose = srv_times_list[0]
        else:
            # wait_in_pose = compute_time_spent_in_service(srv_times_list, preferred_que_name, queue_size, all_serv_times)
            wait_in_pose = compute_time_spent_in_service(srv_times_list, queue_size, all_serv_times)
    return wait_in_pose


def save_jockey_waiting_time(jockeyed_customer_details, filename):
    # print(jockeyed_customer_details)

    try:        
        # header = ["Jockeying_threshold","Jockey_count","Service_rate_diff"]        
        with open(filename, 'a', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(jockeyed_customer_details)
        f.close()

    except OSError:
        if not filename:
            print("Could not open/read file:", filename)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


def get_total_amount_of_wait_for_jockey(wait_in_pose, customer_id, srv_cust_time_dict, wait_time_of_jockey_dict, wait_time_of_jockey_dict_other ):
    # Read from the file of already jockeyed customers, then find the customer_id
    # If the customer_id already exists and has a suffix of _jockey, then read it's 
    # already spent time before jockeying, to the most recent time assigned when it jockeys
    # Return this as the total waiting time the jockeyed customer has spent in the system.
    
    
    wait_one = [value for key, value in wait_time_of_jockey_dict.items() if customer_id in key]
    # print("----- ", wait_one)
    if len(wait_one ) > 1 or len(wait_one) == 1:
        wait_one = float(sum(wait_one))
    #else:
    #    return #wait_one = float(wait_one[0])
        
    wait_two = [value for key, value in wait_time_of_jockey_dict_other.items() if customer_id in key]
    # print("**** ", wait_two)
    if len(wait_two ) > 1 or len(wait_two ) == 1:
        wait_two = float(sum(wait_two))
    #else:
    #    return #wait_two = float(wait_two[0])
    
    for cust in srv_cust_time_dict.keys():
        if cust == customer_id or cust in customer_id:
            waiting_time_at_initial_queue_join = srv_cust_time_dict[cust]
        else:
            continue

        
        if wait_one and waiting_time_at_initial_queue_join and wait_in_pose:
            total_wait_in_queue = waiting_time_at_initial_queue_join + wait_one + wait_in_pose
            return total_wait_in_queue
        
        elif wait_two and waiting_time_at_initial_queue_join and wait_in_pose:
            total_wait_in_queue = waiting_time_at_initial_queue_join + wait_two + wait_in_pose
            return total_wait_in_queue
        elif wait_two and waiting_time_at_initial_queue_join and wait_in_pose and wait_one:
            total_wait_in_queue = waiting_time_at_initial_queue_join + wait_two + wait_in_pose + wait_one
            return total_wait_in_queue
        else:
            continue
        # addition_wait_in_queue = max([value for key, value in wait_time_of_jockey_dict.items() if customer_id in key])
        # if addition_wait_in_queue and waiting_time_at_initial_queue_join and wait_in_pose:        
        #    total_wait_in_queue = waiting_time_at_initial_queue_join + addition_wait_in_queue + wait_in_pose            
        #    return total_wait_in_queue
            
        #else:
        #    continue 


def jockey_or_not(jockeys, other_jockeys, customer_id, curr_queue, preferred_que, preferred_que_name,
                      queue_size, srv_one_cust_time_dict, srv_two_cust_time_dict, queue_name,
                      dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times):


    customer_id = customer_id+"_jockey"     
    
    jockey_count = customer_id.count('jockey')
    dict_customer_id_total_wait_one = {}
    dict_customer_id_total_wait_two  = {}  
    lst_customer_id_total_wait_one = []
    lst_customer_id_total_wait_two  = []
    jockeyed_pose_plus_wait_time_queue_one_extra = []
    jockeyed_pose_plus_wait_time_queue_two_extra = []
    
    lst_customer_id_total_wait_serv_diff_one = []
    lst_customer_id_total_wait_serv_diff_two  = []
    
    lst_len_wait_jockey_count_one  = []
    lst_len_wait_jockey_count_two  = []
    ls_multiple_xteristics = []
    
    # no_jockeyed_thresh_count_wait_diff = []
    
    data_source_one = base_dir+"/constant/srv_one_pose_waits_stats.csv"
    data_source_two = base_dir+"/constant/srv_two_pose_waits_stats.csv"
    
    data_customer_id_jockey_wait_one = base_dir+"/constant/srv_one_jockey_waiting_time.csv"
    data_customer_id_jockey_wait_two = base_dir+"/constant/srv_two_jockey_waiting_time.csv"
    
    data_customer_id_jockey_wait_serv_diff_one = base_dir+"/constant/srv_one_jockey_waiting_time_serv_diff.csv"
    data_customer_id_jockey_wait_serv_diff_two = base_dir+"/constant/srv_two_jockey_waiting_time_serv_diff.csv"
    # data_customer_id_jockey_wait_serv_diff = base_dir+"/constant/all_queues_jockey_waiting_time_serv_diff.csv"
    
    # store the queue length, waiting time to see influence on the number of times jockeyed
    data_length_wait_jockey_count_one = base_dir+"/constant/srv_one_len_wait_count.csv"
    data_length_wait_jockey_count_two = base_dir+"/constant/srv_two_len_wait_count.csv"
    
    # datasource_jockey_details_two = base_dir+'/constant/que_two_jockeying_details.csv'
    # datasource_diffs_two = base_dir+'/constant/que_two_jockeying_serv_diff.csv'
    datasource_diffs = base_dir+'/constant/all_queues_jockeying_serv_diff.csv'
    # datasource_jockey_details_one = base_dir+'/constant/que_one_jockeying_details.csv'
    datasource_jockey_details = base_dir+'/constant/all_que_jockeying_threshold_count_diff.csv'
    datasource_jockey_details_extra = base_dir+'/constant/all_que_jockeying_threshold_count_totalwaitingtime_diff.csv'
    # datasource_diffs_one = base_dir+'/constant/que_one_jockeying_serv_diff.csv'
    datasource_multiple_xters = base_dir+'/constant/multiple_xteristics.csv'
    
    if isinstance(customer_id, list):
        customer_id = ",".join(customer_id)
    
        
    if queue_name == "Server1":        

        now_in_other_queue =  len(list(set(preferred_que.queue)))+1 #preferred_que.qsize()+1 #
        wait_in_pose = get_expected_customer_queue_waiting_times(data_source_two, now_in_other_queue, len(list(set(curr_queue.queue))))   
        jockeyed_waits_two.update({customer_id:wait_in_pose})
        
        lst_wait_threshold_diff_serv_rates_two.append([args.jockeying_threshold, wait_in_pose, abs(args.service_rate_one - args.service_rate_two)])
        lst_jockeyed_pose_wait_time_queue_two.append([now_in_other_queue, wait_in_pose])
        save_jockey_waiting_time(lst_wait_threshold_diff_serv_rates_two, datasource_diffs)
        save_jockey_waiting_time(lst_jockeyed_pose_wait_time_queue_two, base_dir+"/constant/que_two_length_waits_jockeyed.csv")
        jockeyed_pose_plus_wait_time_queue_two.append([args.jockeying_threshold, jockey_count, abs(args.service_rate_one - args.service_rate_two)])
        
        save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_two, datasource_jockey_details) #datasource_jockey_details_two)
        
        total_wait_jockey = get_total_amount_of_wait_for_jockey(wait_in_pose, customer_id, srv_one_cust_time_dict, jockeyed_waits_two, jockeyed_waits_one) 
        
        # print(total_wait_jockey)
        jockeyed_pose_plus_wait_time_queue_two_extra.append([args.jockeying_threshold, jockey_count, total_wait_jockey,  abs(args.service_rate_one - args.service_rate_two)])
        
        ls_multiple_xteristics.append([args.jockeying_threshold, jockey_count, total_wait_jockey,  abs(args.service_rate_one - args.service_rate_two), now_in_other_queue])
        #save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_two_extra, datasource_jockey_details_extra)
        
        if total_wait_jockey is not None:
            # print("................... ", total_wait_jockey, type(total_wait_jockey))
            total_wait_jockey = float(str(total_wait_jockey).lstrip('[').rstrip(']'))
        
            # lst_customer_id_total_wait_two.append([customer_id, jockey_count, total_wait_jockey])
            # save_jockey_waiting_time(lst_customer_id_total_wait_two, data_customer_id_jockey_wait_two)
        
            lst_len_wait_jockey_count_two.append([now_in_other_queue,total_wait_jockey, jockey_count])
        
            lst_customer_id_total_wait_serv_diff_two.append([now_in_other_queue,total_wait_jockey, jockey_count, abs(args.service_rate_two - args.service_rate_one)])
            save_jockey_waiting_time(lst_len_wait_jockey_count_two, data_length_wait_jockey_count_two)
            
            save_jockey_waiting_time(lst_customer_id_total_wait_serv_diff_two, data_customer_id_jockey_wait_serv_diff_two)
            print(colored("%s choosing %s for jockeying with %01d customers. %6.7f seconds will be needed until service completion",'white') % (customer_id, preferred_que_name, now_in_other_queue, total_wait_jockey)) #wait_in_pose))
            save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_two_extra, datasource_jockey_details_extra)
            
            save_jockey_waiting_time(ls_multiple_xteristics, datasource_multiple_xters) 
        
    else:

        now_in_other_queue = len(list(set(preferred_que.queue)))+1 # preferred_que.qsize()+1 
    
        wait_in_pose = get_expected_customer_queue_waiting_times(data_source_one, now_in_other_queue, len(list(set(curr_queue.queue))))
        jockeyed_waits_one.update({customer_id:wait_in_pose})        
            
        lst_wait_threshold_diff_serv_rates_one.append([args.jockeying_threshold, wait_in_pose, abs(args.service_rate_one - args.service_rate_two)])
        
        lst_jockeyed_pose_wait_time_queue_one.append([now_in_other_queue, wait_in_pose])
        save_jockey_waiting_time(lst_wait_threshold_diff_serv_rates_one, datasource_diffs)
        save_jockey_waiting_time(lst_jockeyed_pose_wait_time_queue_one, base_dir+"/constant/que_one_length_waits_jockeyed.csv")
        jockeyed_pose_plus_wait_time_queue_one.append([args.jockeying_threshold, jockey_count, abs(args.service_rate_one - args.service_rate_two)])
        save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_one, datasource_jockey_details) #datasource_jockey_details_one)
        
        total_wait_jockey = get_total_amount_of_wait_for_jockey(wait_in_pose, customer_id, srv_two_cust_time_dict, jockeyed_waits_one, jockeyed_waits_two)
        
        jockeyed_pose_plus_wait_time_queue_one_extra.append([args.jockeying_threshold, jockey_count, total_wait_jockey,  abs(args.service_rate_one - args.service_rate_two)])
        
        ls_multiple_xteristics.append([args.jockeying_threshold, jockey_count, total_wait_jockey,  abs(args.service_rate_one - args.service_rate_two), now_in_other_queue])
        # save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_one_extra, datasource_jockey_details_extra)
        
        if total_wait_jockey is not None:
            # print ("************",total_wait_jockey, type(total_wait_jockey))
            total_wait_jockey = float(str(total_wait_jockey).lstrip('[').rstrip(']'))
        
            lst_customer_id_total_wait_one.append([customer_id, jockey_count, total_wait_jockey])
            save_jockey_waiting_time(lst_customer_id_total_wait_one, data_customer_id_jockey_wait_one)
        
            lst_len_wait_jockey_count_one.append([now_in_other_queue,total_wait_jockey, jockey_count]) # abs(args.service_rate_two - args.service_rate_one)
            lst_customer_id_total_wait_serv_diff_one.append([now_in_other_queue,total_wait_jockey, jockey_count, abs(args.service_rate_one - args.service_rate_two)])
        
            save_jockey_waiting_time(lst_len_wait_jockey_count_one, data_length_wait_jockey_count_one)
            save_jockey_waiting_time(lst_customer_id_total_wait_serv_diff_one, data_customer_id_jockey_wait_serv_diff_one)           
    
            print(colored("%s choosing %s for jockeying with %01d customers. %6.7f seconds will be needed until service completion",'white') % (customer_id, preferred_que_name, now_in_other_queue, total_wait_jockey)) #wait_in_pose))
            save_jockey_waiting_time(jockeyed_pose_plus_wait_time_queue_one_extra, datasource_jockey_details_extra)
        
            save_jockey_waiting_time(ls_multiple_xteristics, datasource_multiple_xters)

    populate_task_into_queue(preferred_que, customer_id)

    proc = multiprocessing.Process(target=simple_process, args=(
        preferred_que, jockeyed_waits_one, jockeyed_waits_two, preferred_que_name, jockeys,
        other_jockeys, curr_queue, jockeyed_customer_server_one_dict, jockeyed_customer_server_two_dict,
        srv_lst_arrivals, all_serv_times, unused_waiting_times))
    # proc.daemon = True
    proc.start()

    # print(colored("%s process started on %s with PID -> %01d","white")%(customer_id, preferred_que_name, proc.pid))
    
    return 


'''
   Want to remove duplicate entries in the jockeying list here
'''

def filter_jockeys(jockeys, jockeys_in_other_queue):

    filtered_jockeys_list = []    
    
    temp_list_jockeys = jockeys
    for temp in temp_list_jockeys:
        split_temp = temp.split("_jockey")[0]
        temp_in_other_queue = any(split_temp in string for string in jockeys_in_other_queue)
        for jockey in jockeys:
            if jockey != split_temp and jockey != temp_in_other_queue:
                filtered_jockeys_list.append(jockey)
                
    return filtered_jockeys_list


def jockeying_init(curr_queue, jockeys, other_jockeys, srv_process_details, preferred_queue, preferred_queue_id,
            srv_lst_arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict,
            queue_name, dict_server_customer_queue, srv_times_list, all_serv_times, unused_waiting_times): #  srv_cust_pose_dict,
            

        jockeys = filter_jockeys(jockeys, other_jockeys)
        for jockey in range(len(jockeys)):

            print(colored("%s has been identified for jockeying...",'red')%(jockeys[jockey]))
            jockey_or_not(jockeys, other_jockeys, jockeys[jockey], curr_queue, preferred_queue, preferred_queue_id,
                          curr_queue.qsize(), srv_one_cust_time_dict, srv_two_cust_time_dict,
                          queue_name, dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times)
    
        simple_process(curr_queue, srv_one_cust_time_dict, srv_two_cust_time_dict,
                           queue_name, jockeys, other_jockeys, preferred_queue, dict_server_customer_queue_one,
                           dict_server_customer_queue_two, srv_lst_arrivals, all_serv_times, unused_waiting_times)

        return preferred_queue_id


def compare_waiting_times(queue_size, preferred_queue, data_source_one,
                          data_source_two, serv_time, customer_id, queue_name):
    if isinstance(customer_id, dict):
        customer_id = list(customer_id.values())[0]

    if serv_time is None:
        # assign a random service time here if service time is None type
        serv_time = np.random.exponential(args.service_rate_one, 1)

    if queue_name == "Server1":
        xpected_in_preferred_que = read_pose_waiting_times_in_queue_two(preferred_queue.qsize() + 1,
                                                                        preferred_queue, data_source_two, serv_time)

        preferred_queue_name = "Server2"
        if xpected_in_preferred_que is None:
            xpected_in_preferred_que = get_expected_by_littles_law(queue_size, serv_time)

        if serv_time > xpected_in_preferred_que:
            print(colored("FYI %s -> tenants who jockeyed to %s at position %01d spent %s ..", 'green') % (
                customer_id, preferred_queue_name, preferred_queue.qsize() + 1, xpected_in_preferred_que))
            jockey_flag = True
        else:
            jockey_flag = False


        enqueued_in_srv_one_cust_times.update({customer_id: xpected_in_preferred_que})
        jockeyed_customer_server_one_dict.update({customer_id: preferred_queue_name})

    else:
        xpected_in_preferred_que = read_pose_waiting_times_in_queue_one(preferred_queue.qsize() + 1,
                                                                        preferred_queue, data_source_one, serv_time)
        if xpected_in_preferred_que is None:
            xpected_in_preferred_que = get_expected_by_littles_law(queue_size, serv_time)

        preferred_queue_name = "Server1"

        if serv_time > xpected_in_preferred_que:
            print(colored("FYI %s -> tenants who jockeyed to %s at position %01d spent %s ..", 'green') % (
                customer_id, preferred_queue_name, preferred_queue.qsize() + 1, xpected_in_preferred_que))
            jockey_flag = True
        else:
            jockey_flag = False

        enqueued_in_srv_two_cust_times.update({customer_id: xpected_in_preferred_que})
        jockeyed_customer_server_two_dict.update({customer_id: preferred_queue_name})

    return jockey_flag


def compare_queue_sizes(curr_queue, preferred_queue):

    queue_size_diff: Any = curr_queue.qsize() - preferred_queue.qsize()
    jockey_flag: bool = False

    if queue_size_diff >= args.jockeying_threshold:
        jockey_flag = True
    else:
        jockey_flag = False

    return jockey_flag


def jockey_selector(curr_queue, serv_time, customer_id, queue_name, preferred_queue, jockeying_threshold, preferred_queue_name):

    jockey_flag = False
    if isinstance(jockeying_threshold, int) or isinstance(jockeying_threshold, float):
        if curr_queue.qsize() > 0 or preferred_queue.qsize() > 0:
            queue_size_diff: Any = curr_queue.qsize() - preferred_queue.qsize() #abs(curr_queue.qsize() - preferred_queue.qsize())
            '''
               Only jockey to an alternative queue if the difference between either queues is
               greater than the jockeying threshold and that difference is less than the size of the current queue
            '''
            if queue_size_diff >= jockeying_threshold: # and queue_size_diff < curr_queue.qsize():
                if curr_queue.qsize() > preferred_queue.qsize():

                    print(colored("FYI::=> %s -> Customer about to jockey at position %01d to %s ..",'green')%(customer_id, preferred_queue.qsize()+1 ,preferred_queue_name) )
                    jockey_flag = True
                    enqueued_in_srv_one_cust_times.update({customer_id:serv_time})
                    jockeyed_customer_server_one_dict.update({customer_id:preferred_queue_name})

                else:
                    print(colored("FYI::=> %s -> Customer about to jockey at position %01d to %s ..",'green')%(customer_id, preferred_queue.qsize()+1 ,preferred_queue_name) )
                    jockey_flag = True
                    enqueued_in_srv_two_cust_times.update({customer_id:serv_time})
                    jockeyed_customer_server_two_dict.update({customer_id:preferred_queue_name})
    else:

        try:
            data_source_one = base_dir+"/constant/srv_one_pose_waits_stats.csv"
            data_source_two = base_dir+"/constant/srv_two_pose_waits_stats.csv"

            jockey_flag = compare_waiting_times(len(curr_queue.queue), preferred_queue, data_source_one, data_source_two,
                                  serv_time, customer_id, queue_name)

        except FileNotFoundError:
            msg = "Sorry, the file does not exist."
            raise msg

    return jockey_flag


def random_jockey_selector(queue_obj, still_in_queue):

    if len(still_in_queue) > 1:
        selector = np.random.randint(1,len(still_in_queue))
        customer_to_jockey = still_in_queue.get(selector)
        if not customer_to_jockey:
            customer_to_jockey = list(still_in_queue.values())[selector-1] # over impatient customer might jockey from position 1 to longer queue

        return customer_to_jockey
        
        
'''
   Returns from the long list of potential remaining customers in the list that
   want to jockey only a few (random numbe based on the service rates) with the 
   maximum waiting times in the queue. 
'''

def get_max_waiting_times_list(still_in_queue, srv_cust_time_dict):
    
    selected = []
    wait_times = []
    selected_details = {}
    randstartparam = int(min(args.service_rate_one, args.service_rate_two))
    
    for customer in still_in_queue:
        #for cust, time in srv_one_cust_time_dict.items():
        for cust, time in srv_cust_time_dict.items():
            if cust == customer:
                wait_times.append(time)
                selected_details.update({cust:time})                 
    
    n = random.randint(1, randstartparam)

    # First, sort the List
    wait_times.sort()    
    # Now, get the largest N integers from the list
    if n < len(wait_times):
        
        only_selected = wait_times[-n:]    
    else:
        only_selected = wait_times[-1:]            
    
    if len(only_selected) > 0:
        
        for selected_time in only_selected:
            counter = 0
            if selected_time in list(selected_details.values()):                
                selected.append(list(selected_details.keys())[counter])
            counter = counter + 1
              
        return selected

def bulk_random_jockey_selector(curr_queue, still_in_queue, srv_one_cust_time_dict, srv_two_cust_time_dict):
    '''
       compute the customers to bulky jockey by getting those with the maximum 
       waiting but selecting only a few from the maximum list randomly.
    '''
    
    candidates_to_move = get_max_waiting_times_list(still_in_queue, srv_one_cust_time_dict, srv_two_cust_time_dict)

    bulk_jockey_list = []
    
    try:

        if isinstance(candidates_to_move, list) and len(candidates_to_move) > 1:
            bulk_jockey_list = random.sample(candidates_to_move, len(candidates_to_move))           
        else:           
            if candidates_to_move and len(candidates_to_move) > 1:
                bulk_jockey_list = random.sample(list(candidates_to_move.values()),len(candidates_to_move)) 
                #bulk_jockey_list = random.sample(list(still_in_queue.values()),len(still_in_queue))

    except ValueError:
        print('Sample size exceeded population size.')

    return bulk_jockey_list


def in_curr_queue_at_pose(customer_to_switch, still_in_queue):

    real_pose = 1

    if isinstance(still_in_queue, list):
        for item in still_in_queue:
            if item[2] == customer_to_switch:
                return real_pose
            else:
                real_pose = real_pose + 1
    else:
        for cust_pose, cust in still_in_queue.items():
            if cust == customer_to_switch:
                return cust_pose #real_pose # cust_pose
            else:
                cust_pose = cust_pose + 1


def re_evaluate_jockeying_decision_based_on_both_queue_metrics(customers_in_motion, queue_, preferred_queue,
                                      queue_name, customer_processes_in_one, customer_processes_in_two, jockeys,
                                      other_jockeys, srv_cust_pose_dict, dict_server_customer_queue_one,
                                      dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict,
                                      srv_two_cust_time_dict):

    customer_id = get_last_in_curr_queue(queue_, customers_in_motion)

    data_source_one = base_dir + "/constant/srv_one_pose_waits_stats.csv"
    data_source_two = base_dir + "/constant/srv_two_pose_waits_stats.csv"
    jockey_flag = compare_waiting_times(queue_, preferred_queue, data_source_one,
                          data_source_two, serv_time, customer_id, queue_name)

    if jockey_flag and args.jockeying_threshold is not None:
        pass

def re_evaluate_jockeying_based_on_waiting_time(customers_in_motion, still_in_preferred_queue, queue_, preferred_queue, queue_name,
                                      customer_processes_in_one, customer_processes_in_two, jockeys,
                                      other_jockeys, dict_server_customer_queue_one,
                                      dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict,
                                      srv_two_cust_time_dict, all_serv_times, unused_waiting_times):

    customer_last_in_queue = get_last_in_curr_queue(queue_, customers_in_motion)

    customer_last_in_queue_id = list(customer_last_in_queue.values())[0]
    pose_of_last_customer_in_queue = list(customer_last_in_queue.keys())[0]

    customer_now_queued_in_server = on_which_server(customer_last_in_queue, dict_server_customer_queue_one,
                                                    dict_server_customer_queue_two)

    data_source_one = base_dir + "/constant/srv_one_pose_waits_stats.csv"
    data_source_two = base_dir + "/constant/srv_two_pose_waits_stats.csv"

    if queue_name == "Server1": # customer_now_queued_in_server
        # time_to_spend_in_current_queue = srv_one_cust_time_dict.get(customer_last_in_queue_id)
        time_to_spend_in_current_queue = find_customer_times(customer_last_in_queue_id, srv_one_cust_time_dict, srv_two_cust_time_dict)
        xpected_in_preferred_que = read_pose_waiting_times_in_queue_one(pose_of_last_customer_in_queue + 1,
                                                                        preferred_queue, data_source_two,
                                                                        time_to_spend_in_current_queue)
        if xpected_in_preferred_que is None:
            xpected_in_preferred_que = get_expected_by_littles_law(len(list(queue_.queue)), time_to_spend_in_current_queue)

        jockey_flag = compare_waiting_times(queue_, preferred_queue, data_source_one,
                                            data_source_two, time_to_spend_in_current_queue,
                                            customer_last_in_queue, queue_name)

        if jockey_flag:
            preferred_que_name = "Server2"
            proc_id_to_kill = get_process_id_to_terminate(customer_last_in_queue_id, customer_processes_in_one,
                                                          customer_processes_in_two)
            serv_time = serv_time_one[0]
            srv_times_list = serv_time_one
            print(colored("%s in %s at position %s jockeying to %s to position %01d. Terminating process ID: %s ",
                          "yellow") % (
                      customer_last_in_queue_id, customer_now_queued_in_server, pose_of_last_customer_in_queue,
                      preferred_que_name, preferred_queue.qsize() + 1, proc_id_to_kill))

            jockeyed_customer_server_one_dict.update({customer_last_in_queue_id: preferred_que_name})

            dict_server_customer_queue = dict_server_customer_queue_one
            # kill returned processID matching the customer to move
            ProcessHandler().terminate_process(proc_id_to_kill, queue_)
            # customer_to_switch = list(customer_to_switch.values())[0]

            jockey_or_not(jockeys, other_jockeys, customer_last_in_queue_id, queue_, preferred_queue, preferred_que_name,
                          queue_.qsize(), serv_time_one, serv_time_two, queueID,
                          dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times)

            jockey_rate = get_jockeying_rate(jockeyed_customer_server_one_dict, srv_lst_arrivals, queue_.qsize(), preferred_queue.qsize()) #global_arrivals_in_one)
            #print(colored("Customers jockeyed from %s at a rate of %6.4f",'yellow', attrs=["blink"])%(queue_name, jockey_rate))

    else:
        time_to_spend_in_current_queue = find_customer_times(customer_last_in_queue_id, srv_one_cust_time_dict, srv_two_cust_time_dict)

        xpected_in_preferred_que = read_pose_waiting_times_in_queue_one(pose_of_last_customer_in_queue + 1,
                                                                        preferred_queue, data_source_two, time_to_spend_in_current_queue)
        if xpected_in_preferred_que is None:
            xpected_in_preferred_que = get_expected_by_littles_law(len(list(queue_.queue)), time_to_spend_in_current_queue)

        jockey_flag = compare_waiting_times(queue_, preferred_queue, data_source_one,
                                            data_source_two, time_to_spend_in_current_queue, customer_last_in_queue_id
                                            , queue_name)

        if jockey_flag:

            preferred_que_name = "Server1"
            proc_id_to_kill = get_process_id_to_terminate(customer_last_in_queue_id, customer_processes_in_one,
                                                          customer_processes_in_two)
            serv_time = serv_time_two[0]
            srv_times_list = serv_time_one
            # srv_process_details = srv_one_process_details
            # at_pose = list(customer_to_switch.keys())[0]
            print(colored("%s in %s at position %s jockeying to %s to position %01d. Terminating process ID: %s ", "yellow") % (
                        customer_last_in_queue_id, customer_now_queued_in_server, pose_of_last_customer_in_queue,
                        preferred_que_name, preferred_queue.qsize() + 1, proc_id_to_kill))

            jockeyed_customer_server_two_dict.update({customer_last_in_queue_id: preferred_que_name})

            dict_server_customer_queue = dict_server_customer_queue_one
            # kill returned processID matching the customer to move
            ProcessHandler().terminate_process(proc_id_to_kill, queue_)

            jockey_or_not(jockeys, other_jockeys, customer_last_in_queue_id, queue_, preferred_queue, preferred_que_name,
                          queue_.qsize(), serv_time_one, serv_time_two, queueID,
                          dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times)

            jockey_rate = get_jockeying_rate(jockeyed_customer_server_two_dict, queue_.qsize(), srv_lst_arrivals,  preferred_queue.qsize()) # global_arrivals_in_two)

            # print(colored("Customers jockeyed from %s at a rate of %6.4f",'yellow', attrs=["blink"])%(queue_name, jockey_rate))

    #return jockey_flag


def pose_updated(still_in_queue, still_in_preferred_queue, customer_id):

    customer_id = customer_id[0]
    new_queue_poses = 0
    
    if len(still_in_queue) > 0:
        pose = 1
        for customer in still_in_queue:
            if customer == customer_id:
                new_queue_poses = pose
            pose = pose + 1
        #return new_queue_poses
        
    if len(still_in_preferred_queue) > 0:
        pose = 1
        for customer in still_in_preferred_queue:
            if customer == customer_id:
                new_queue_poses = pose
            pose = pose + 1
            
    return new_queue_poses


def save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queue_name):

    length_jockey_rate_dict = {}
    length_jockey_rate_dict.update({len(set(still_in_queue)):jockey_rate_in_curr_queue})

    if queue_name == "Server1":
        try:
            filename = base_dir+'/constant/que_one_length_jockey_rates.csv'

            header = ["Queue_length","Jockeying_rate"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in length_jockey_rate_dict.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not filename:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    else:
        try:
            filename = base_dir+'/constant/que_two_length_jockey_rates.csv'

            header = ["Queue_length","Jockeying_rate"]
            with open(filename, 'a', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
            #writer.writerow(header) # write the header

                for k,v in length_jockey_rate_dict.items():
                    writer.writerow([k,v])

            f.close()

        except OSError:
            if not filename:
                print("Could not open/read file:", filename)
                sys.exit()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


def switch_customers_in_bulk_shuffled(customer_to_switch_in_curr_queue, preferred_que, srv_lst_arrivals, queue_name):

    final_entries = []

    for item in customer_to_switch_in_curr_queue:
        extracted_cust_id = []
        if len(item) == 3:
            item = item[2]
            extracted_cust_id.append(item)
        else:
            extracted_cust_id.append(item)

    if queue_name == "Server1":
        no_duplicates = [e for e in srv_lst_arrivals if e not in processed_in_one]
        combined_customer_list = extracted_cust_id + no_duplicates
        # TODO: Below - trying to keep track of which jockey customer to move to which queue after reordering
        final_entries.append(random.sample(combined_customer_list, len(combined_customer_list)))
    else:
        no_duplicates = [e for e in srv_lst_arrivals if e not in processed_in_two]
        combined_customer_list = extracted_cust_id + no_duplicates
        final_entries.append(random.sample(combined_customer_list, len(combined_customer_list)))

    return final_entries


def get_estimated_wait_in_queue_pose(pose, queue_name, serv_time, data_source):

    
    if queue_name == "Server1":
        avg_wait_que = read_pose_waiting_times_in_queue_two(pose, preferred_queue, data_source, serv_time)#data_source_one, serv_time)
                
    else:
        avg_wait_que = read_pose_waiting_times_in_queue_two(pose, preferred_queue, data_source, serv_time)#data_source_two, serv_time)
        
    return avg_wait_que
        
        
# plot of queue size (x-axis independent variable) with waiting time as a dependent variable (y-axis)
# plot of jockeying size (x-axis independent variable) with waiting time a dependent variable (y-axis)
# 
        
def process_jockeying_re_evaluating(curr_queue, preferred_queue, customer_to_switch_in_curr_queue, customer_now_queued_in_server, srv_times_list, 
                                    srv_lst_arrivals, still_in_queue, still_in_preferred_queue, jockeys, other_jockeys, 
                                    all_serv_times, queueID, srv_one_cust_time_dict, srv_two_cust_time_dict, unused_waiting_times): 
                                    
    sorted_list = sorted(list(set(still_in_queue)))
    # print("******* ", customer_to_switch_in_curr_queue)
    # customer_to_switch_in_curr_queue = ', '.join(customer_to_switch_in_curr_queue)
        
    if isinstance(still_in_queue, list):   
                                  
        if  customer_now_queued_in_server == "Server1":
            # print("------------- CALLED RE_EVALUATE-------------------") #, avg_wait_in_alt_queue_pose, avg_wait_in_curr_queue_pose)            
            #pose_in_curr_queue = pose_updated(list(set(curr_queue.queue)),list(set(preferred_queue.queue)), customer_to_switch_in_curr_queue)
            # pose_in_curr_queue = in_curr_queue_at_pose(customer_to_switch_in_curr_queue, list(set(still_in_queue)))
            
            # print("******* ",customer_to_switch_in_curr_queue,  sorted_list)
            # if customer_to_switch_in_curr_queue in in_list or customer_to_switch_in_curr_queue+"_jockey" == in_list:
            if customer_to_switch_in_curr_queue in sorted_list:
                pose_in_curr_queue = int(sorted_list.index(customer_to_switch_in_curr_queue))+1 
            #else:
            #    return 
            # if not pose_in_curr_queue:
            #    return         
            dict_server_customer_queue = dict_server_customer_queue_one
            preferred_que_name = "Server2"
        else:
            # Since it is a priority queue, elements are ordered in the way they are added,
            # so we sort the list to get the customers list and return the index as a position in the queue
            # Then add one since the indices start with zero
            if customer_to_switch_in_curr_queue in sorted_list:
                pose_in_curr_queue = int(sorted_list.index(customer_to_switch_in_curr_queue))+1
            #else:            
            #    return 
            dict_server_customer_queue = dict_server_customer_queue_two
            preferred_que_name = "Server1"
    
    # Fixme: As an example ['Batch3_customer4'] - the variable customer_to_switch_in_curr_queue sometimes 
    #        contains a list therefore to strip of the square brackets below is a fast walk-around.
    if isinstance(customer_to_switch_in_curr_queue, list):
        customer_to_switch_in_curr_queue = customer_to_switch_in_curr_queue[0]
    
    pose_in_alt_queue = len(list(set(preferred_queue.queue)))+1
    
    
    if customer_now_queued_in_server == "Server1":
        # definately should be read from the file since the customer has since changed position
        # or alternatively build a function for how_much_more_customer_stays()
        # avg_wait_in_curr_queue_pose = srv_one_cust_time_dict.get(customer_to_switch_in_curr_queue)
        data_source = base_dir + "/constant/srv_one_pose_waits_stats.csv"
        serv_time = srv_one_cust_time_dict.get(customer_to_switch_in_curr_queue)
        if customer_to_switch_in_curr_queue in sorted_list:
            pose_in_curr_queue = int(sorted_list.index(customer_to_switch_in_curr_queue))+1
        # avg_wait_in_curr_queue_pose = read_pose_waiting_times_in_queue_two( pose_in_curr_queue, curr_queue, data_source, serv_time)
            avg_wait_in_curr_queue_pose = get_expected_customer_queue_waiting_times(data_source, pose_in_curr_queue, len(list(curr_queue.queue)))
        
            if avg_wait_in_curr_queue_pose: 
                alt_data_source = base_dir + "/constant/srv_two_pose_waits_stats.csv"       
                avg_wait_in_alt_queue_pose = get_expected_customer_queue_waiting_times(alt_data_source, pose_in_alt_queue, len(list(curr_queue.queue)))
                
                # if  avg_wait_in_curr_queue_pose and avg_wait_in_alt_queue_pose:          
                if avg_wait_in_alt_queue_pose < avg_wait_in_curr_queue_pose:        
                    
                    if customer_now_queued_in_server == "Server1":
                        global_jockeying_list_in_one.append(customer_to_switch_in_curr_queue)
                    else:
                        global_jockeying_list_in_two.append(customer_to_switch_in_curr_queue)
                    # at_pose = in_curr_queue_at_pose(customer_to_switch_in_curr_queue, still_in_queue)
                    
                    proc_id_to_kill = get_process_id_to_terminate(customer_to_switch_in_curr_queue, customer_processes_in_one, customer_processes_in_two)
                    if proc_id_to_kill:
                        
                        ProcessHandler().terminate_process(proc_id_to_kill, curr_queue)
                        
                        print(colored("%s in %s at position %s jockeying to %s to position %01d. Terminating process ID: %s ", "yellow") % (
                                customer_to_switch_in_curr_queue, customer_now_queued_in_server, pose_in_curr_queue, preferred_que_name, preferred_queue.qsize() + 1, proc_id_to_kill))
                        
                        jockey_or_not(jockeys, other_jockeys, customer_to_switch_in_curr_queue, curr_queue, preferred_queue, preferred_que_name,
                                curr_queue.qsize(), srv_one_cust_time_dict, srv_two_cust_time_dict, customer_now_queued_in_server,
                                dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times) 
                                    
                else:
                    return
        
    else:
        
        # avg_wait_in_curr_queue_pose = srv_two_cust_time_dict.get(customer_to_switch_in_curr_queue)
        serv_time = srv_two_cust_time_dict.get(customer_to_switch_in_curr_queue)
        data_source = base_dir + "/constant/srv_two_pose_waits_stats.csv"
        # plus 1 below because an index is returned starting at 0
        if customer_to_switch_in_curr_queue in sorted_list:
            pose_in_curr_queue = int(sorted_list.index(customer_to_switch_in_curr_queue))+1
        # avg_wait_in_curr_queue_pose = read_pose_waiting_times_in_queue_two( pose_in_curr_queue, curr_queue, data_source, serv_time)
            avg_wait_in_curr_queue_pose = get_expected_customer_queue_waiting_times(data_source, pose_in_curr_queue, len(list(curr_queue.queue)))
     
            if avg_wait_in_curr_queue_pose:
                alt_data_source = base_dir + "/constant/srv_one_pose_waits_stats.csv"       
                avg_wait_in_alt_queue_pose = get_expected_customer_queue_waiting_times(alt_data_source, pose_in_alt_queue, len(list(curr_queue.queue)))
        
                # if  avg_wait_in_curr_queue_pose and avg_wait_in_alt_queue_pose:          
                if avg_wait_in_alt_queue_pose < avg_wait_in_curr_queue_pose:        
                    
                    if customer_now_queued_in_server == "Server1":
                        global_jockeying_list_in_one.append(customer_to_switch_in_curr_queue)
                    else:
                        global_jockeying_list_in_two.append(customer_to_switch_in_curr_queue)
                    # at_pose = in_curr_queue_at_pose(customer_to_switch_in_curr_queue, still_in_queue)
                    
                    proc_id_to_kill = get_process_id_to_terminate(customer_to_switch_in_curr_queue, customer_processes_in_one, customer_processes_in_two)
                    if proc_id_to_kill:
                        
                        ProcessHandler().terminate_process(proc_id_to_kill, curr_queue)
                        
                        print(colored("%s in %s at position %s jockeying to %s to position %01d. Terminating process ID: %s ", "yellow") % (
                                customer_to_switch_in_curr_queue, customer_now_queued_in_server, pose_in_curr_queue, preferred_que_name, preferred_queue.qsize() + 1, proc_id_to_kill))
                        
                        jockey_or_not(jockeys, other_jockeys, customer_to_switch_in_curr_queue, curr_queue, preferred_queue, preferred_que_name,
                                curr_queue.qsize(), srv_one_cust_time_dict, srv_two_cust_time_dict, customer_now_queued_in_server,
                                dict_server_customer_queue, srv_times_list, srv_lst_arrivals, all_serv_times, unused_waiting_times) 
                                    
                else:
                    return

# TODO: check that this customer is not a new arrival -> in the direction of tagging to 
# TODO: to also address the mentioned point of differentiating between an arrival and a jockey.

def re_evaluate_jockeying_based_on_jockeying_threshold(still_in_queue, still_in_preferred_queue, curr_queue, preferred_queue, queueID,
                                  customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys,
                                  dict_server_customer_queue_one, dict_server_customer_queue_two,
                                  srv_lst_arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict, all_serv_times, unused_waiting_times):

    # we generate a new service time for the new jockey = 1 customer
    serv_time_one = np.random.exponential(args.service_rate_one, 1)
    serv_time_two = np.random.exponential(args.service_rate_two, 1)

    serv_time_one = np.append(list(srv_one_cust_time_dict.values()), serv_time_one)
    serv_time_two = np.append(list(srv_two_cust_time_dict.values()), serv_time_two)  

    if len(still_in_queue) >= 1:
                
        if queueID == "Server1":
            customer_to_switch_in_curr_queue = get_max_waiting_times_list(still_in_queue, srv_one_cust_time_dict)   
        else:
            customer_to_switch_in_curr_queue = get_max_waiting_times_list(still_in_queue, srv_two_cust_time_dict)   
             
        
        if customer_to_switch_in_curr_queue:
            if len(customer_to_switch_in_curr_queue) > 1: 
                customer_to_switch_in_curr_queue = list(set(customer_to_switch_in_curr_queue))
                customerid = customer_to_switch_in_curr_queue[0]
                customer_now_queued_in_server = on_which_server(customerid, dict_server_customer_queue_one, dict_server_customer_queue_two)
            else:
                customerid = customer_to_switch_in_curr_queue[0]
                customer_now_queued_in_server = on_which_server(customerid, dict_server_customer_queue_one, dict_server_customer_queue_two)
        else:      
            customer_now_queued_in_server = queueID
            
        customer_to_switch_in_curr_queue_pose = in_curr_queue_at_pose(customer_to_switch_in_curr_queue, still_in_queue)

        if customer_to_switch_in_curr_queue:
            
            customers_with_arrivals_shuffled_curr = switch_customers_in_bulk_shuffled(customer_to_switch_in_curr_queue, preferred_queue, srv_lst_arrivals, queueID)
            customers_with_arrivals_shuffled_curr = list(flatten(customers_with_arrivals_shuffled_curr))            
            
            if customers_with_arrivals_shuffled_curr:
                if queueID == "Server1":
                    preferred_que_name = "Server2"
                else:
                    preferred_que_name = "Server1"
        
                if len(customers_with_arrivals_shuffled_curr) == 1:                            
                         
                    serv_time = serv_time_one[0]
                    srv_times_list = serv_time_one
        
                    # jockeyed_customer_server_one_dict.update({customers_with_arrivals_shuffled_curr[0]:preferred_que_name})
                    '''
                        Possibilities in the case of simultaneous jockeying from both queues.
                            i) Read jockeying rate from file or use the currently running statistics.
                            ii) Jockey the tenant with the longest serving time in the current queue and
                                this service time is still greater when switched to the next available position
                                in the alternative queue.
                            iii) The jockeying is done randomly - based on the random number corresponding
                                to a customer position.
                    '''
                    
                    process_jockeying_re_evaluating(curr_queue, preferred_queue, customers_with_arrivals_shuffled_curr,
                                customer_now_queued_in_server, srv_times_list, srv_lst_arrivals, still_in_queue, still_in_preferred_queue,
                                jockeys, other_jockeys, all_serv_times, queueID, srv_one_cust_time_dict, srv_two_cust_time_dict, unused_waiting_times) #  srv_cust_pose_dict,
                    
                    if queueID == "Server1":
                        jockeyed_customer_server_one_dict.update({customers_with_arrivals_shuffled_curr[0]:preferred_que_name})
                        jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_one_dict, global_arrivals_in_one, 
                                                curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))
                        jockeying_rates_with_threshold_in_one.update({args.jockeying_threshold:jockey_rate_in_curr_queue})
                        save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, customer_now_queued_in_server)
                        
                        save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_one, args.service_rate_one, jockey_rate_in_curr_queue)
                        save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                    else:
                        jockeyed_customer_server_two_dict.update({customers_with_arrivals_shuffled_curr[0]:preferred_que_name})
                        jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_two_dict, global_arrivals_in_two,
                                            curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))  
                        jockeying_rates_with_threshold_in_two.update({args.jockeying_threshold:jockey_rate_in_curr_queue}) 
                        save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, customer_now_queued_in_server)     
                        
                        save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_two, args.service_rate_two, jockey_rate_in_curr_queue)
                        save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                                            
                    
                elif len(customers_with_arrivals_shuffled_curr) > 1: 
                    
                    if queueID == "Server1":
                        curr_preferred_queue_id = "Server2"
                    else:
                        curr_preferred_queue_id = "Server1"
                    
                    if len(unused_waiting_times) > 0:
                        srv_times_list = unused_waiting_times
                    else:
                        # if no unused service times in the originally generated dictionary
                        # simply generate a new list of service times for each jockey candidate.
                        srv_times_list = np.random.exponential(len(customers_with_arrivals_shuffled_curr), 1)                                    
                    
                    count = 0
                                        
                    for customer_to_switch in customers_with_arrivals_shuffled_curr:
                        
                        # jockeyed_customer_server_two_dict.update({customer_to_switch: curr_preferred_queue_id})
                        process_jockeying_re_evaluating(curr_queue, preferred_queue, customer_to_switch, 
                                                customer_now_queued_in_server, srv_times_list, srv_lst_arrivals, still_in_queue, still_in_preferred_queue,
                                                jockeys, other_jockeys, all_serv_times, queueID, srv_one_cust_time_dict, srv_two_cust_time_dict, unused_waiting_times)
                        
                        if queueID == "Server1":
                            # TODO:: modularize(create a seperate function block) save these details to reduce redundant code
                            jockeyed_customer_server_one_dict.update({customers_with_arrivals_shuffled_curr[count]:preferred_que_name})
                            jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_one_dict, global_arrivals_in_one,
                                        curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))
                            jockeying_rates_with_threshold_in_one.update({args.jockeying_threshold:jockey_rate_in_curr_queue})
                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, customer_now_queued_in_server)
                            
                            save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_one, args.service_rate_one, jockey_rate_in_curr_queue)
                            save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                        else:
                            jockeyed_customer_server_two_dict.update({customers_with_arrivals_shuffled_curr[count]:preferred_que_name})
                            jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_two_dict, global_arrivals_in_two,
                                                curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))  
                            jockeying_rates_with_threshold_in_two.update({args.jockeying_threshold:jockey_rate_in_curr_queue}) 
                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, customer_now_queued_in_server)           
                            
                            save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_two, args.service_rate_two, jockey_rate_in_curr_queue)
                            save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                        
                        count = count + 1

    if len(still_in_preferred_queue) >= 1:
        if queueID == "Server1":
            customer_to_switch_in_preferred_queue = get_max_waiting_times_list(still_in_preferred_queue, srv_one_cust_time_dict)
        else:
            customer_to_switch_in_preferred_queue = get_max_waiting_times_list(still_in_preferred_queue, srv_two_cust_time_dict)
            
        if customer_to_switch_in_preferred_queue:
            if len(customer_to_switch_in_preferred_queue) > 1: 
                customer_to_switch_in_preferred_queue = list(set(customer_to_switch_in_preferred_queue))

            customer_to_switch_in_preferred_queue_pose = in_curr_queue_at_pose(customer_to_switch_in_preferred_queue, still_in_preferred_queue)

        # use one of the customers to retrieve which queue the entire batch is in
        if customer_to_switch_in_preferred_queue:
            customerid = customer_to_switch_in_preferred_queue[0]
            customer_now_queued_in_server = on_which_server(customerid, dict_server_customer_queue_one, dict_server_customer_queue_two)
        else:
            customer_now_queued_in_server = queueID
        
        if customer_to_switch_in_preferred_queue:
            customers_with_arrivals_shuffled_preferred = switch_customers_in_bulk_shuffled(customer_to_switch_in_preferred_queue, preferred_queue, srv_lst_arrivals, queueID)
            customers_with_arrivals_shuffled_preferred = list(flatten(customers_with_arrivals_shuffled_preferred))

            if customers_with_arrivals_shuffled_preferred: # el
                if queueID == "Server1":
                    curr_preferred_queue_id = "Server2"
                else:                
                    curr_preferred_queue_id = "Server1"
                    
                if len(unused_waiting_times) > 0:
                    srv_times_list = unused_waiting_times
                else:
                    '''
                        if no unused service times in the originally generated dictionary exist
                        simply generate a new list of service times for each jockey candidate.
                    '''
                    srv_times_list = np.random.exponential(len(customers_with_arrivals_shuffled_preferred), 1)
                    
                if len(customers_with_arrivals_shuffled_preferred) == 1:                
                        
                    serv_time = serv_time_two[0]
                    srv_times_list = serv_time_two
                    srv_process_details = srv_two_process_details
        
                    process_jockeying_re_evaluating(curr_queue, preferred_queue, customers_with_arrivals_shuffled_preferred,
                                                    customer_now_queued_in_server, srv_times_list,
                                                    srv_lst_arrivals, still_in_queue, still_in_preferred_queue, jockeys, other_jockeys,
                                                    all_serv_times, queueID, srv_one_cust_time_dict, srv_two_cust_time_dict, unused_waiting_times)

                    if queueID == "Server1":
                        jockeyed_customer_server_one_dict.update({customers_with_arrivals_shuffled_preferred[0]: curr_preferred_queue_id})
                        jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_one_dict, global_arrivals_in_one, curr_queue.qsize(), preferred_queue.qsize() ) # global_arrivals_in_one)
                        jockeying_rates_with_threshold_in_one.update({args.jockeying_threshold:jockey_rate_in_curr_queue})
                        save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, customer_now_queued_in_server)
                        
                        save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_one, args.service_rate_one, jockey_rate_in_curr_queue)
                        save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                    else:
                        jockeyed_customer_server_two_dict.update({customers_with_arrivals_shuffled_preferred[0]: curr_preferred_queue_id})
                        jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_two_dict, global_arrivals_in_two, curr_queue.qsize(), preferred_queue.qsize()) # global_arrivals_in_two)  
                        jockeying_rates_with_threshold_in_two.update({args.jockeying_threshold:jockey_rate_in_curr_queue}) 
                        save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, customer_now_queued_in_server)                    
                        save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_two, args.service_rate_two, jockey_rate_in_curr_queue)
                        save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                
                elif len(customers_with_arrivals_shuffled_preferred) > 1:
                                                        
                    count = 0
                    for customer_to_switch in customers_with_arrivals_shuffled_preferred:                       
                        process_jockeying_re_evaluating(curr_queue, preferred_queue, customer_to_switch,
                                        customer_now_queued_in_server, srv_times_list, srv_lst_arrivals, still_in_queue, 
                                        still_in_preferred_queue, jockeys, other_jockeys, all_serv_times, queueID, 
                                        srv_one_cust_time_dict, srv_two_cust_time_dict, unused_waiting_times)
                                    
                        if queueID == "Server1":
                            jockeyed_customer_server_one_dict.update({customers_with_arrivals_shuffled_preferred[count]: curr_preferred_queue_id})
                            jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_one_dict, global_arrivals_in_one, curr_queue.qsize(), preferred_queue.qsize()) # global_arrivals_in_one)
                            jockeying_rates_with_threshold_in_one.update({args.jockeying_threshold:jockey_rate_in_curr_queue})
                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, customer_now_queued_in_server)
                            save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_one, args.service_rate_one, jockey_rate_in_curr_queue)
                            save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                        else:
                            jockeyed_customer_server_two_dict.update({customers_with_arrivals_shuffled_preferred[count]: curr_preferred_queue_id})
                            jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_two_dict, global_arrivals_in_two, curr_queue.qsize(), preferred_queue.qsize()) # global_arrivals_in_two)  
                            jockeying_rates_with_threshold_in_two.update({args.jockeying_threshold:jockey_rate_in_curr_queue}) 
                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, customer_now_queued_in_server)             
                            
                            save_serv_rate_jockeying_rate_in_queues(serv_rates_jockeying_file_two, args.service_rate_two, jockey_rate_in_curr_queue)
                            save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
                            
                        count = count + 1
                                
    else:
        
        print("No customer is interested in anymore jockeying....")
        return


def get_expected_by_littles_law(curr_queue, serv_time):
    
    try:
        # return (curr_queue.qsize()/serv_time)        
        return (curr_queue/serv_time)
          
    except ZeroDivisionError:
        print('Service Time = %6.7f: Cannot divide by zero.'%(serv_time))
        sys.exit(1)
        

def get_left_overs(left_overs, jockeys):
    still_in_queue = []
    for left_over in left_overs:
        t = str(left_over[2])
        if t not in jockeys:
            still_in_queue.append( t)
            return still_in_queue
        else:
            return still_in_queue


def customer_after_exit_new_positions(customers_in_process):
    customer_motion_updates = {}
    for customer in customers_in_process:        
        new_position = int(customer[0])
        customerid = str(customer[2])
        customer_motion_updates.update({new_position:customerid})

    return customer_motion_updates


def leftover_includes_a_jockey(left_behind, customerid):
    if len(left_behind) > 0:
        if customerid in left_behind:
            task_is_jockey = True
        else:
            task_is_jockey = False
    else:
        task_is_jockey = False

    return task_is_jockey


'''
       This is not the best approach for finding out which queue the
       customer that wants to jockey is currently enqueued.
       The approach requires many iterations making it less optimal
'''


def on_which_server(customerid, dict_srv_one, dict_srv_two):

    for t in dict_srv_one.items():
        if t[0] == customerid:
            required_value = t[1]
            return required_value

    for j in dict_srv_two.items():
        if j[0] == customerid:
            required_value = j[1]
            return required_value

'''
   FIXME:
       A poor approach to finding what I am looking for.
       Find a way of keeping track of customers
'''


def find_customer_times(customerid, srv_one_cust_time_dict, srv_two_cust_time_dict):
    for t in srv_one_cust_time_dict.items():
        if t[0] == customerid:
            required_value = t[1]
            return required_value

    for j in srv_two_cust_time_dict.items():
        if j[0] == customerid:
            required_value = j[1]
            return required_value

    for k in jockeyed_waits_one.items():
        if k[0] == customerid:
            required_value = k[1]
            return required_value

    for z in jockeyed_waits_two.items():
        if z[0] == customerid:
            required_value = z[1]
            return required_value


def get_last_in_curr_queue(curr_queue, customers_in_motion):

    last_in_queue_pose = {}
    if len(customers_in_motion) > 0:
        k, last_value = _, customers_in_motion[k] = customers_in_motion.popitem()
        last_in_queue_pose.update({k:last_value})

        return last_in_queue_pose


def print_left_in_queue(sorted_left_in_queue):
    for key, value in sorted_left_in_queue:
        return value+" at position "+key


def start_customer_process(customer, queue_name):
    srv_one_process_details: Dict[Any, Optional[int]] = {}
    srv_two_process_details: Dict[Any, Optional[int]] = {}

    if queue_name == "Server1":
        proc = multiprocessing.Process(target=mock_process, args=(customer, queue_name))
        proc.daemon = True
        proc.start()
        srv_one_process_details.update({customer:proc})

        return srv_one_process_details
    else:
        proc = multiprocessing.Process(target=mock_process, args=(customer, queue_name))
        proc.daemon = True
        proc.start()
        srv_two_process_details.update({customer: proc})

        return srv_two_process_details



def cleanup_queue_listings_of_duplicates_in_curr(jockeyed_customer_server_dict, queue_):    
    
    customers_being_processed_in_curr_queue = []
    
    lst_jockeys = jockeyed_customer_server_dict.values()
    
    for t in list(queue_.queue):
        if t in lst_jockeys:
            continue
        else:
            customers_being_processed_in_curr_queue.append(t)
        
    return customers_being_processed_in_curr_queue
    

def cleanup_queue_listings_of_duplicates_in_preferred(jockeyed_customer_server_dict, preferred_queue):
    
    customers_being_processed_in_preferred_queue = []
    lst_jockeys = jockeyed_customer_server_dict.values()
    
    for t in list(preferred_queue.queue):
        if t in lst_jockeys:
            continue
        else:
            customers_being_processed_in_preferred_queue.append(t)
            
    return customers_being_processed_in_preferred_queue

    
def simple_process(queue_, srv_one_cust_time_dict, srv_two_cust_time_dict,
                       queue_name, jockeys, other_jockeys, preferred_queue, dict_server_customer_queue_one,
                       dict_server_customer_queue_two, srv_lst_arrivals, all_serv_times, unused_waiting_times):

    srv_pose_wait_dict = {}

    task = queue_.get()
    lst_done_processed.append(task)
        
    proc = multiprocessing.Process(target=mock_process, args=(task, queue_name))
    proc.daemon = True
    proc.start()

    # customer process store
    if queue_name == "Server1":
        preferred_queue_name = "Server2"
        customer_processes_in_one.update({task: proc.pid})
        customers_being_processed_in_curr_queue = cleanup_queue_listings_of_duplicates_in_curr(jockeyed_customer_server_one_dict, queue_)
        customers_being_processed_in_other_queue = cleanup_queue_listings_of_duplicates_in_preferred(jockeyed_customer_server_two_dict, preferred_queue) 
        
    else:
        customer_processes_in_two.update({task: proc.pid})
        preferred_queue_name = "Server1"
        
        customers_being_processed_in_curr_queue = cleanup_queue_listings_of_duplicates_in_curr(jockeyed_customer_server_two_dict, queue_)
        customers_being_processed_in_other_queue = cleanup_queue_listings_of_duplicates_in_preferred(jockeyed_customer_server_one_dict, preferred_queue)
        
    actual_server = on_which_server(task, dict_server_customer_queue_one, dict_server_customer_queue_two)
    jockey_on_server = on_which_server(task, jockeyed_customer_server_one_dict, jockeyed_customer_server_two_dict)
    
    customers_being_processed_in_curr_queue = [ x for x in customers_being_processed_in_curr_queue if x not in lst_done_processed ]
    customers_being_processed_in_other_queue = [ x for x in customers_being_processed_in_other_queue if x not in lst_done_processed ] 
        
    customers_being_processed_in_curr_queue = filter_inputs(jockeyed_waits_two, jockeyed_waits_one, customers_being_processed_in_curr_queue)
    customers_being_processed_in_other_queue = filter_inputs(jockeyed_waits_two, jockeyed_waits_one, customers_being_processed_in_other_queue)
    
    customers_being_processed_in_curr_queue = filter_jockeys(customers_being_processed_in_curr_queue, customers_being_processed_in_other_queue)
    customers_being_processed_in_other_queue = filter_jockeys(customers_being_processed_in_other_queue, customers_being_processed_in_curr_queue)   
    
    # srv_lst_arrivals = []
    
    # TODO: Fix left over displayed not to include those jockeyed but original name appearing in list
    
    lst_threshold_timetoservice_one = {}
    lst_threshold_timetoservice_two = {}
    no_jockeyed_thresh_count_diff = []
    
    jockey_count = 0
    datasource_jockey_details = base_dir+'/constant/all_que_jockeying_threshold_count_diff.csv'
    
    if queue_.qsize() == 0:
        if queue_name == "Server1":
            dest_filename = "threshold_timetoservice_queue_one.csv"
            if task in jockeyed_waits_one and task not in global_jockeying_list_in_one: # and not in global_jockeying_list_in_two:
                time_to_service = jockeyed_waits_one.get(task)
                print("%s left %s of size %01d after %s " % (task, jockey_on_server, len(customers_being_processed_in_curr_queue), time_to_service)) 
                
            else:
                if task in list(srv_one_cust_time_dict.keys()):
                    time_to_service = srv_one_cust_time_dict.get(task)                
                    print("%s left %s of size %01d after %s " % (task, actual_server, len(customers_being_processed_in_curr_queue), time_to_service))
                    
            queue_.task_done()
            
            lst_threshold_timetoservice_one.update({args.jockeying_threshold:time_to_service})
            # print(" ***** writing to one *****************", lst_threshold_timetoservice_one)
            save_threshold_timetoservice_queue(lst_threshold_timetoservice_one, dest_filename)

            processed_in_one.append(task)
 
        else:
            dest_filename = "threshold_timetoservice_queue_two.csv"
            if task in jockeyed_waits_two and task not in global_jockeying_list_in_two: # and not in global_jockeying_list_in_one:
                time_to_service = jockeyed_waits_two.get(task)            
                print("%s left %s of size %01d after %s " % (task, jockey_on_server, len(customers_being_processed_in_curr_queue), time_to_service))
            else:
                if task in list(srv_two_cust_time_dict.keys()):
                    time_to_service = srv_two_cust_time_dict.get(task)
                    print("%s left %s of size %01d after %s "% ( task, actual_server, len(customers_being_processed_in_curr_queue), time_to_service ))
            queue_.task_done()
            
            lst_threshold_timetoservice_two.update({args.jockeying_threshold:time_to_service})
            
            save_threshold_timetoservice_queue(lst_threshold_timetoservice_two, dest_filename)
            processed_in_two.append(task)   
    
        if not args.jockeying_threshold:

            if len(customers_being_processed_in_curr_queue) > 0: 

                print(colored("Jockey candidates in alternative queue %s :=> %s", "green")%(preferred_queue_name, list(set(customers_being_processed_in_other_queue)))) 
                print(colored("Jockey candidates in current queue %s given the waiting times at the respective "
                    "positions :=> %s ", "green") % (queue_name, customers_being_processed_in_curr_queue))

                re_evaluate_jockeying_based_on_waiting_time(list(set(queue_.queue)),list(set(preferred_queue.queue)), queue_, preferred_queue, queue_name,
                                      customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, dict_server_customer_queue_one,
                                      dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict, all_serv_times, unused_waiting_times)

        else:
            from_one_to_two_jockeys = {}
            from_two_to_one_jockeys = {}
            
            # diffs_in_lengths = abs(len(customers_being_processed_in_curr_queue) - len(customers_being_processed_in_other_queue)) #abs(queue_.qsize() - preferred_queue.qsize())
            other_diff = abs(len(list(set(queue_.queue))) - len(list(set(preferred_queue.queue))) )
            
            if  other_diff >= args.jockeying_threshold:
                # FIXME: update the customer positions in the queue
                # new_poses_in_curr_queue = pose_updater(customers_in_motion, queue_.qsize())
                # new_poses_in_preferred_queue = pose_updater(customer_in_motion_in_preferred_queue, preferred_queue.qsize())
                if len(list(set(queue_.queue))) > 1: # 0:
                    print(colored("Jockey candidates in current queue %s given the queue size differences %01d :=> %s",
                        "green") % (queue_name,  other_diff, list(set(queue_.queue))))
                       
                    from_one_to_two_jockeys.update({queue_name:customers_being_processed_in_curr_queue})
                    
                    re_evaluate_jockeying_based_on_jockeying_threshold(list(set(customers_being_processed_in_curr_queue)), list(set(customers_being_processed_in_other_queue)), queue_, preferred_queue, queue_name,
                                                               customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, dict_server_customer_queue_one,
                                                               dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict,srv_two_cust_time_dict, all_serv_times,
                                                               unused_waiting_times)
                                                               
                if len(list(set(preferred_queue.queue))) > 1: #0:
                    print(colored("Jockey candidates in alternative queue %s :=> %s","green")%(preferred_queue_name, list(set(preferred_queue.queue))))
                    
                    from_one_to_two_jockeys.update({preferred_queue_name:customers_being_processed_in_other_queue})

                    re_evaluate_jockeying_based_on_jockeying_threshold(list(set(queue_.queue)), list(set(preferred_queue.queue)), queue_, preferred_queue, queue_name,
                                                               customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, dict_server_customer_queue_one,
                                                               dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict,srv_two_cust_time_dict, all_serv_times,
                                                               unused_waiting_times)
                

    else:      
        
        if jockey_on_server:
            actual_server = jockey_on_server

        if len(customers_being_processed_in_curr_queue) > 0 and queueID == "Server1":  # actual_server == "Server1":
            dest_filename = "threshold_timetoservice_queue_one.csv"
            if task in jockeyed_waits_one:
                time_to_service: Optional[Any] = jockeyed_waits_one.get(task)
            else:
                time_to_service: Optional[Any] = srv_one_cust_time_dict.get(task)                        

            if time_to_service and actual_server:
                if task not in global_jockeying_list_in_one:
                    print("%s left %s of size %01d after %s. Left over in %s :=> %s" % (
                            task, actual_server, len(list(set(customers_being_processed_in_curr_queue))), time_to_service, actual_server,
                            list(set(customers_being_processed_in_curr_queue))))
                            
            queue_.task_done()
            
            lst_threshold_timetoservice_one.update({args.jockeying_threshold:time_to_service})
            # print(" ***** writing to one *****************", lst_threshold_timetoservice_one)
            save_threshold_timetoservice_queue(lst_threshold_timetoservice_one, dest_filename)
            srv_pose_wait_dict.update({queue_.qsize(): time_to_service})
            srv_one_process_details.update({proc.pid:proc})
            save_pose_waits_to_file_one(list(srv_pose_wait_dict.items()))

            processed_in_one.append(task)            

        elif len(customers_being_processed_in_curr_queue) > 0 and queueID == "Server2": #actual_server == "Server2":
            
            dest_filename = "threshold_timetoservice_queue_two.csv"
            if task in jockeyed_waits_two:
                time_to_service = jockeyed_waits_two.get(task)
            else:
                time_to_service = srv_two_cust_time_dict.get(task)

            if time_to_service or actual_server :
                if task not in global_jockeying_list_in_two:
                    print("%s left %s of size %01d after %s. Left over in %s :=> %s" % (
                            task, actual_server, len(list(set(customers_being_processed_in_curr_queue))), time_to_service, actual_server,
                            list(set(customers_being_processed_in_curr_queue))))
                            
            queue_.task_done()
            
            lst_threshold_timetoservice_two.update({args.jockeying_threshold:time_to_service})
            # print(" ***** writing to two *****************", lst_threshold_timetoservice_two)
            save_threshold_timetoservice_queue(lst_threshold_timetoservice_two, dest_filename)
            srv_pose_wait_dict.update({queue_.qsize(): time_to_service})
            srv_two_process_details.update({proc.pid:proc})
            save_pose_waits_to_file_two(list(srv_pose_wait_dict.items()))

            processed_in_two.append(task)                           


        if not args.jockeying_threshold:
            if len(list(set(queue_.queue))) > 0:
                print(colored("Jockey candidates in current queue %s given the waiting times at the respective "
                    "positions :=> %s ", "green") % (queue_name, list(set(customers_being_processed_in_curr_queue) - set(customers_being_processed_in_other_queue))))
                    
                print(colored("Jockey candidates in alternative queue %s :=> %s", "green")%(preferred_queue_name, list(set(customers_being_processed_in_other_queue) -set(customers_being_processed_in_curr_queue))))

                re_evaluate_jockeying_based_on_waiting_time(customers_being_processed_in_curr_queue, customers_being_processed_in_other_queue, 
                                    queue_, preferred_queue, queue_name, customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, 
                                    dict_server_customer_queue_one, dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict,
                                    srv_two_cust_time_dict, all_serv_times, unused_waiting_times)
        else:
            
            diffs_in_lengths = abs(len(customers_being_processed_in_curr_queue) - len(customers_being_processed_in_other_queue))
            
            other_diff = abs(len(list(set(queue_.queue))) - len(list(set(preferred_queue.queue))) )
            
            # print("===============>> ", other_diff, args.jockeying_threshold)
            
            if other_diff >= args.jockeying_threshold:
                
                '''
                     > 1 because it does not make sense for a customer that is in position one to jockey 
                     since that customer is next up for processing such that the cost of terminating the current job
                     and moving it into the other queue plus the jockeying costs together exceed the benefits of jockeying to 
                     the other queue even if that alternative queue is empty.
                '''
                if len(list(set(queue_.queue))) > 1: 
                    print(colored("Jockey candidates in current queue %s given the queue size differences %01d :=> %s ",
                        "green") % (queue_name, other_diff, list(set(customers_being_processed_in_curr_queue) - set(customers_being_processed_in_other_queue))))
                        
                    re_evaluate_jockeying_based_on_jockeying_threshold(customers_being_processed_in_curr_queue, customers_being_processed_in_other_queue, 
                                    queue_, preferred_queue, queue_name, customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys,
                                    dict_server_customer_queue_one, dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict, 
                                    srv_two_cust_time_dict, all_serv_times, unused_waiting_times)
                                    
                if len(list(set(preferred_queue.queue))) > 1: 
                    print(colored("Jockey candidates in alternative queue %s :=> %s","green")%(preferred_queue_name, list(set(customers_being_processed_in_other_queue) - set(customers_being_processed_in_curr_queue))))
                    
                    re_evaluate_jockeying_based_on_jockeying_threshold(list(set(queue_.queue)), list(set(preferred_queue.queue)), 
                                    queue_, preferred_queue, queue_name, customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys,
                                    dict_server_customer_queue_one, dict_server_customer_queue_two, srv_lst_arrivals, srv_one_cust_time_dict, 
                                    srv_two_cust_time_dict, all_serv_times, unused_waiting_times)
        
        no_jockeyed_thresh_count_diff.append([args.jockeying_threshold, jockey_count, abs(args.service_rate_one - args.service_rate_two)])
        
        save_jockey_waiting_time(no_jockeyed_thresh_count_diff, datasource_jockey_details)

    return task #queue_.qsize()
    
    
def filter_inputs(jockeyed_waits_two, jockeyed_waits_one, customers_being_processed_in_curr_queue):
    
    filtered_list_being_processed = []
    for customer in customers_being_processed_in_curr_queue:
        if customer in jockeyed_waits_one.values():
            continue
        elif customer in jockeyed_waits_two.values():
            continue
        else:
            filtered_list_being_processed.append(customer)
            
    return filtered_list_being_processed            


def mock_process(task, queue_name):
    return


def populate_task_into_queue(queue_, customer):
    # plus one because tenant always lands at the end of the queue
    queue_.put(customer, len(list(set(queue_.queue)))+1)# queue_.qsize()+1)
    return queue_


def get_jockeying_rate(jockeys_in_queue, srv_arrivals, queue_size_now, alt_queue_size_now):
    # based on koenigsberg work
    # k_{i}(w_i - w_j) => k is the likelihood that jockeying occurs
    # => k = (number of arrivals - number of serviced)/ number of arrivals
    
    k = abs(len(srv_arrivals) - len(jockeys_in_queue))/len(srv_arrivals)
    # print("JOCKEYING RATE ===>> ", k, queue_size_now, alt_queue_size_now)
    jockeying_rate = k * abs(queue_size_now - alt_queue_size_now)
    
    return jockeying_rate
    
    # return (len(jockeys_in_queue)/len(srv_arrivals))


'''
    The total time a customer spends in the queue is computed from the list of
    service times depending on what position that customer gets into the queue
'''


def compute_time_spent_in_service(service_time_list, position_in_queue, all_serv_time):

    total_time_till_served = 0
    
    # if curr_queue_name == "Server1":
    if position_in_queue == 1 or position_in_queue == 0:
        total_time_till_served = total_time_till_served + service_time_list[0]
    elif position_in_queue > 1 and position_in_queue <= len(service_time_list):
        total_time_till_served = total_time_till_served + sum(service_time_list[:position_in_queue])
    elif position_in_queue > 1 and position_in_queue > len(service_time_list):
        # total_time_till_served = total_time_till_served + sum(all_serv_time[-position_in_queue:])
        # Instead of the above computation, we use first principles => the formulae W =  W_q + 1/mean service rate.
        # W_q = L_q/arrival rate -> Little's Law
        # to compute this waiting time (in the system - given that the customer might jockey) 
        # in the case the position the customer lands in exceeds the number of service times generated
        waiting_in_queue = get_expected_by_littles_law(position_in_queue, args.arrival_rate)
        total_time_till_served = waiting_in_queue + (1 /sum(service_time_list))
    '''
    else:
        if position_in_queue == 1 or position_in_queue == 0:
            total_time_till_served = total_time_till_served + service_time_list[0]
        elif position_in_queue > 1 and position_in_queue <= len(service_time_list):
            total_time_till_served = total_time_till_served + sum(service_time_list[-position_in_queue:])
        elif position_in_queue > 1:
            print("******** ", all_serv_time[-position_in_queue:])
            total_time_till_served = total_time_till_served + sum(all_serv_time[-position_in_queue:])
    '''

    return total_time_till_served


'''
    How much time did the customer spend in a position before moving forward
    Find customer that just left in srv_pose_wait_dict, check srv_cust_pose_dict for customer
    and extract the customer wait period. Then subtract this wait time from the previous's
    wait time to get the time the customer that just left spent in a given position
'''


def compute_waiting_time_in_a_given_pose(processed_customer, srv_cust_pose_dict, srv_cust_wait_dict):

    for customer in list(srv_cust_pose_dict): # , pose   items()):
        if customer in srv_cust_wait_dict:
            pose = srv_cust_pose_dict[customer]
            if pose != 0:
                wait = srv_cust_wait_dict[customer]
                pose = pose + 1
                if pose < len(srv_cust_wait_dict):

                    waiting_times = list(srv_cust_wait_dict.values())
                    time_in_pose = waiting_times[pose] - wait
                    return time_in_pose
                else:
                    break


def start_processing_queues(queue_, srv_one_cust_time_dict, srv_two_cust_time_dict, srv_cust_pose_dict,
                            jockeys, other_jockeys, queue_name, preferred_queue, srv_lst_arrivals,
                            dict_server_customer_queue_one, dict_server_customer_queue_two, srv_one_times_list,
                            srv_two_times_list, all_serv_times, unused_waiting_times, local_arrivals):                                

    if len(jockeys) > 0 and queue_name == "Server1":
        preferred_queue_id = "Server2"
        flag_no_jockeys = True

        jockeyed_to = jockeying_init(queue_, jockeys, other_jockeys, srv_two_process_details, preferred_queue,
                                     preferred_queue_id, srv_lst_arrivals, srv_one_cust_time_dict,
                                     srv_two_cust_time_dict, queue_name, dict_server_customer_queue_one, 
                                     srv_two_times_list, all_serv_times, unused_waiting_times)


    elif len(jockeys) > 0 and queue_name == "Server2" :
        preferred_queue_id = "Server1"
        flag_no_jockeys = True

        jockeyed_to = jockeying_init(queue_, jockeys, other_jockeys, srv_two_process_details, preferred_queue,
                                     preferred_queue_id, srv_lst_arrivals, srv_two_cust_time_dict,
                                     srv_one_cust_time_dict, queue_name, dict_server_customer_queue_two, 
                                     srv_one_times_list, all_serv_times, unused_waiting_times) 
                                     
    #else:
    #flag_no_jockeys = True

    if queue_.qsize() > 0:

        if queue_name == "Server1":

            current_user = simple_process(queue_, srv_one_cust_time_dict, srv_two_cust_time_dict,
                           queue_name, jockeys, other_jockeys, preferred_queue, dict_server_customer_queue_one, 
                           dict_server_customer_queue_two, srv_lst_arrivals, all_serv_times, unused_waiting_times)
            return current_user

        if queue_name == "Server2":
            current_user = simple_process(queue_, srv_two_cust_time_dict, srv_one_cust_time_dict,
                           queue_name, jockeys, other_jockeys, preferred_queue, dict_server_customer_queue_two,
                           dict_server_customer_queue_one, srv_lst_arrivals, all_serv_times, unused_waiting_times)
                           
            return current_user


def is_list_of_arrays_input(still_in_queue_element, still_in_queue):
    list_trimmed = []

    if len(still_in_queue_element) == 3:
        for element in still_in_queue:
            list_trimmed.append(element[2])

        return list_trimmed

    else:
        return still_in_queue

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
            
            
def get_arrivals_jockeys_given_jockeying_threshold(still_in_queue, still_in_preferred_queue, curr_queue, preferred_queue, queueID,
                                  customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys,
                                  dict_server_customer_queue_one, dict_server_customer_queue_two,
                                  srv_lst_arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict, all_serv_times, unused_waiting_times):
    
    jockey_candidates = []
    
    customerid = still_in_queue[0]
    
    customer_now_queued_in_server = on_which_server(customerid, dict_server_customer_queue_one, dict_server_customer_queue_two) 
       
    if len(still_in_queue) >= 1:
        if customer_now_queued_in_server == "Server1":
            customer_to_switch_in_curr_queue = get_max_waiting_times_list(still_in_queue, srv_one_cust_time_dict)        
        else:
            customer_to_switch_in_curr_queue = get_max_waiting_times_list(still_in_queue, srv_two_cust_time_dict)                
        
        if customer_to_switch_in_curr_queue:
            # a count of how many times a customer is jockeying                    
            if len(customer_to_switch_in_curr_queue) >= 1: 
                customer_to_switch_in_curr_queue = list(set(customer_to_switch_in_curr_queue))
                count = 1
                for customer in customer_to_switch_in_curr_queue:
                    customer = customer+"_jockey"                    
                    jockey_candidates.append(customer)          
                    if "_jockey" in customer:
                        count = count + 1
            
        customer_to_switch_in_curr_queue_pose = in_curr_queue_at_pose(customer_to_switch_in_curr_queue, still_in_queue)         

        if jockey_candidates: 
            for jockey in jockey_candidates:
                if queueID == "Server1":
                    preferred_que_name = "Server2"
                    jockeyed_customer_server_two_dict.update({jockey:preferred_que_name})
                else:
                    preferred_que_name = "Server1"
                    jockeyed_customer_server_one_dict.update({jockey:preferred_que_name})
                    
            jockey_rate_in_curr_queue = get_jockeying_rate(jockeyed_customer_server_one_dict, global_arrivals_in_one, 
                                                        curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))
            jockey_rate_in_other_queue = get_jockeying_rate(jockeyed_customer_server_two_dict, global_arrivals_in_two,
                                        curr_queue.qsize(), preferred_queue.qsize()) # len(still_in_queue), len(still_in_preferred_queue))
            save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, customer_now_queued_in_server)
            save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, customer_now_queued_in_server)
            save_jockeying_rate_queue_length(jockey_rate_in_curr_queue, still_in_queue, queueID)
            save_jockeying_rate_queue_length(jockey_rate_in_other_queue, still_in_queue, queueID)
                    
            customers_with_arrivals_shuffled_curr = switch_customers_in_bulk_shuffled(jockey_candidates, preferred_queue, srv_lst_arrivals, queueID)
            customers_with_arrivals_shuffled_curr = list(set(flatten(customers_with_arrivals_shuffled_curr)))
            
            return customers_with_arrivals_shuffled_curr
        

def plot_results_3D(data_source_one, data_source_two):
    try:
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        # set up the axes for the first plot
        ax = Axes3D(fig)
        #ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        df_one = pd.read_csv(data_source_one, sep=',') 
        # df_two = pd.read_csv(data_source_two, sep=',')
        
        columns = list(df_one.columns.values)  
        # col_jockeying_threshold = columns[0]
        col_jockeying_count = columns[1]      
        col_service_rate_diff = columns[2] 
        
        X, Y = np.meshgrid(np.array(df_one[col_jockeying_count].values.tolist()), np.array(df_one[col_service_rate_diff].values.tolist()))
        
        Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
        
        # surf = ax.contour3D(X, Y, Z, 50, cmap='binary')
        
        # Z = np.reshape(df_one[col_service_rate_diff].values.tolist(), (-1, 2)) #len(df_one)))
        
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
                       
        fig.colorbar(surf, shrink=0.5, aspect=10)

    except OSError:
        if not data_source_one or not data_source_two:
            print("Could not open/read file one of the data sources.")
            sys.exit()

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
          
    ax.set_xlabel("Number of times jockeyed")
    ax.set_ylabel("Difference in service times")
    ax.set_zlabel("")
      
    # ax.legend()
    plt.show()

def plot_results(data_source_one, data_source_two):

    try:
        fig, ax = plt.subplots()
        x_one_punkte = []
        y_one_punkte = []
        x_two_punkte = []
        y_two_punkte = []
        df_one = pd.read_csv(data_source_one, sep=',') # os.getcwd()+'/'+
        df_two = pd.read_csv(data_source_two, sep=',') 
        
        columns_in_source = list(df_one.columns.values)        
        col1 = columns_in_source[0]
        col2 = columns_in_source[1]      
        data_one = pd.DataFrame(df_one.groupby(col1, as_index = False)[col2].mean())
        data_two = pd.DataFrame(df_two.groupby(col1, as_index = False)[col2].mean())
            
        x_one_punkte.append(data_one.index.get_level_values(0).tolist())
        x_two_punkte.append(data_two.index.get_level_values(0).tolist())
        for i in data_one.values:        
            y_one_punkte.append(i[1])
        
        for i in data_two.values:        
            y_two_punkte.append(i[1])
            
        x_one_punkte = np.array(x_one_punkte).ravel()
        x_two_punkte = np.array(x_two_punkte).ravel() 
        
        ax.plot(x_one_punkte, y_one_punkte, label="Queue One")
        ax.plot(x_two_punkte, y_two_punkte, label="Queue Two")
        

    except OSError:
        if not data_source_one or not data_source_two:
            print("Could not open/read file one of the data sources.")
            sys.exit()

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
          
    ax.set_ylabel(col1)
    ax.set_xlabel(col2)
      
    ax.legend()
    plt.show()
    

class ServiceLines(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self.all_items = set()
        self.counter = 0

    def put(self, item, priority):
        if item not in self.all_items:
            self.all_items.add(item)
            PriorityQueue.put(self, (priority, self.counter, item))
            self.counter += 1

    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, *args, **kwargs)
        return item

    def generate_queues(self, num_of_queues):
        dict_queues = {}

        for i in range(num_of_queues):
            code_string = "Server%01d" % (i+1)
            queue_object = PriorityQueue()
            dict_queues.update({code_string: queue_object})

        return dict_queues


def save_diff_serv_rates(diff_service_rates, service_rates_difference_file):
    try:
        
        with open(service_rates_difference_file, 'a') as filehandle:
            for item in diff_service_rates:            
                filehandle.write("%s\n" % item)            
    except OSError:
        if not fielname:
            print("Could not open/read file:", service_rates_difference_file)
            sys.exit()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='Jockeying_Simulator',
    formatter_class=argparse.RawTextHelpFormatter,
    description=dedent('''\
        ****************************************************************
        *  Simulating two to N parallel queues with Jockeying!  (o o)  *
        *                                                       \ ^ /  *
        ****************************************************************
        *                                                              *
        * Tooling takes on arguments to control the behaviour. A few   *
        * arguments are mandatory, the jockeying threshold is optional.*
        * If no number of queues is specified then by default          *
        * two queues are setup.                                        *
        * The queue_status_info parameter controls whether the prior   *
        * information is provided about the queues at admission is     *
        * availed or not.                                              *
        * By default customers switch queues based on the mean waiting *
        * times, otherwise the jockeying threshold (difference between *
        * queue sizes) is used.                                        *
        *                                                              *
        ****************************************************************
        *                              END                             *
        ****************************************************************
        '''))

    '''
       TODO:: - make jockeying_threshold optional, also allow inputs for all parameters as floats
              - In the case of multiple queues, let the service rates be set randomly for each queue
                so the script should not take parameters like service_rate_one or service_rate_two.
              - Intension is to control how long the script runs using the number_of_batches parameter

    '''
    parser.add_argument('--service_rate_one', dest='service_rate_one', type=float, required=True,
                        help='Rate at which customers are processed in the first queue!')
    parser.add_argument('--service_rate_two', dest='service_rate_two', type=float, required=True,
                        help='Rate at which customers are processed in the second queue!')
    parser.add_argument('--arrival_rate', dest='arrival_rate', type=int, required=True, default=np.random.poisson(arr_rate),
                        help='Rate at which customers arrive for processing!')
    parser.add_argument('--jockeying_threshold', dest='jockeying_threshold', type=float, nargs='?',
                        help='When a customer is moved to another queue is controlled by this value')
    parser.add_argument('--num_of_queues', dest='num_of_queues', type=int, required=False ,nargs='?',
                        help='How many queues should be used in the simulation')
    parser.add_argument('--hybrid_metrics', dest='hybrid', type=int, required=False, nargs='?',
                        help='Use both the jockeying threshold and waiting time metrics to decide if to jockey')
    parser.add_argument('--run_id', dest='run_id', type=int, required=True, nargs='?',
                        help='This value is picked up from the script wrapper and attached to every customer so as to diferentiate the simulation run ID')

    args = parser.parse_args()
        
    time_now = time.time()
    # The parameter below regulates the simulation time period t
    # Increasing on decreasing the it's value needs increasing the processing period
    # Line 2394
    time_end = time_now + 10 #0 # 0 # 0 20 #
    
    num_of_queues = args.num_of_queues
    diff_service_rates = []

    if isinstance(num_of_queues,int) and num_of_queues > 0:
        dict_queues = ServiceLines().generate_queues(num_of_queues)
        print("Setting up the following queues ==> ", dict_queues)
    else:
        print("Simulation run %01d will proceed with only two queues......!!"%( args.run_id))
        dict_queues = ServiceLines().generate_queues(2)
        #queue_1 = ServiceLines()
        #queue_2 = ServiceLines()

    arrivals_batch_num = 1    
    run = args.run_id
    
    srv_one_lst_arrivals = []
    srv_two_lst_arrivals = [] 
    srv_one_process_details = {}
    srv_two_process_details = {}

    if args.jockeying_threshold is not None:
        jockey_threshold = args.jockeying_threshold
    else:
        jockey_threshold = ""

    if len(sys.argv) != 0:

        all_serv_time_one = []
        all_serv_time_two = []
        new_times_one = []
        new_times_two = []
        new_joins_in_one = []
        new_joins_in_two = []
        all_service_times_queue_one = []
        all_service_times_queue_two = []
        jockeying_rates_in_queue_one = []
        jockeying_rates_in_queue_two = []
        unused_waiting_times_in_one = []
        unused_waiting_times_in_two = []
        jockeys_in_one = []
        jockeys_in_two = []

        '''
           FIXME:
              There is a problem with customer jockeying that need fixing, popping things out
              when jockeying

              z.B:: dictionary below containing the same key:value pairs
                  Batch14_Customer4 process started on Server1 with PID -> 662993
                  Customers jockeyed from Server2 at a rate of 0.0714
                  The following {4: 'Batch14_Customer4', 5: 'Batch14_Customer4'} are re-evaluating their
                  status in Server1 given the queue size differences 3
        '''        
        
        while time_now < time_end:
            
            num_arrivals = np.random.poisson(args.arrival_rate)
            print(colored("%01d customers need access to services..",'yellow')%(num_arrivals)) #, attrs=['blink']
            
            queue_1 = list(dict_queues.values())[0]
            queue_2 = list(dict_queues.values())[1]
            
            # counter = 0
            # for t in tqdm(range(num_arrivals), desc="Executed:"):        
            if num_arrivals > 0:
                # jockeys_in_one = []
                # jockeys_in_two = []
            
                arrivals = generate_arrivals(num_arrivals, arrivals_batch_num, run)

                serv_time_one = np.random.exponential(args.service_rate_one, num_arrivals)
                serv_time_two = np.random.exponential(args.service_rate_two, num_arrivals)

                # print("----------->", serv_time_one, serv_time_two, args.jockeying_threshold)
                t = random.randint(1, num_arrivals)

                new_joins_in_one = 0
                new_joins_in_two = 0
                
                shuffled = []

                #if queue_1.qsize() > 0 or queue_2.qsize() > 0:
                if list(dict_queues.values())[0].qsize() > 0 or list(dict_queues.values())[1].qsize() > 0:
                    diff_queue_sizes = abs(int(list(dict_queues.values())[0].qsize() - list(dict_queues.values())[1].qsize()))                    
                    
                    if diff_queue_sizes >= args.jockeying_threshold:
                        still_in_queue_one = list(list(dict_queues.values())[0].queue)
                        still_in_queue_two = list(list(dict_queues.values())[1].queue)
                                                                
                        if still_in_queue_one:
                            queueID = list(dict_queues.keys())[0]
                            
                        if still_in_queue_two:
                            queueID = list(dict_queues.keys())[1]
                        
                            
                        if queueID == "Server1":
                            jockeys = jockeys_in_one
                            other_jockeys = jockeys_in_two
                            
                            shuffled = get_arrivals_jockeys_given_jockeying_threshold(still_in_queue_one, still_in_queue_two, queue_1, queue_2, queueID,
                                            customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, dict_server_customer_queue_one, 
                                            dict_server_customer_queue_two, arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict, serv_time_one, serv_time_one)
                        else:
                            jockeys = jockeys_in_two
                            other_jockeys = jockeys_in_one
                            
                            shuffled = get_arrivals_jockeys_given_jockeying_threshold(still_in_queue_two, still_in_queue_one, queue_2, queue_1, queueID,
                                            customer_processes_in_one, customer_processes_in_two, jockeys, other_jockeys, dict_server_customer_queue_one, 
                                            dict_server_customer_queue_two, arrivals, srv_one_cust_time_dict, srv_two_cust_time_dict, serv_time_two, serv_time_two)
                        
                if shuffled and len(shuffled) > 0:
                    
                    shuffled = list(flatten(shuffled))
                    for i in range(len(shuffled)):
                        local_arrivals_in_one = []
                        local_arrivals_in_two = []
                        
                        if shuffled[i] in still_in_queue_one:
                            curr_queue = queue_1
                            preferred_queue = queue_2
                            preferred_queue_id = list(dict_queues.keys())[1]
                            jockey_or_not(jockeys, other_jockeys, shuffled[i], curr_queue, preferred_queue, 
                                        preferred_queue_id, curr_queue.qsize(), srv_one_cust_time_dict, srv_two_cust_time_dict, list(dict_queues.keys())[0], 
                                        dict_server_customer_queue_one, serv_time_one, arrivals, serv_time_one, serv_time_one)
                                                                                    
                                        
                        elif shuffled[i] in still_in_queue_two:
                            curr_queue = queue_2
                            preferred_queue = queue_1
                            preferred_queue_id = list(dict_queues.keys())[0]
                            jockey_or_not(jockeys, other_jockeys, shuffled[i], curr_queue, preferred_queue, 
                                        preferred_queue_id, curr_queue.qsize(), srv_one_cust_time_dict, srv_two_cust_time_dict, list(dict_queues.keys())[1],
                                        dict_server_customer_queue_two, serv_time_two, arrivals, serv_time_two, serv_time_two)
                        
                        else:
                            #if queue_1.qsize() <= queue_2.qsize():
                            if len(queue_1.queue) <= len(queue_2.queue):
                                    
                                queueID = list(dict_queues.keys())[0]                                
                                preferred_que_name = list(dict_queues.keys())[1]
                                preferred_queue = queue_2
                                #if len(list(queue_1.queue)) == 0:
                                #    queue_size_now = len(list(queue_1.queue))+1
                                #else:
                                queue_size_now = len(list(queue_1.queue))
                                alt_queue_size_now = len(list(queue_2.queue))
                                                                
                                task_list_one = populate_task_into_queue(queue_1, shuffled[i])
                                dict_server_customer_queue_one.update({shuffled[i]: queueID})
                                processing_time = compute_time_spent_in_service(serv_time_one, queue_size_now,
                                                                                all_serv_time_one)
                                print("%s joined %s at position %01d. Queue2 has %01d customers. %6.7f seconds will be needed to service completion." % (
                                    shuffled[i], queueID, queue_size_now+1, alt_queue_size_now, processing_time))
                                srv_one_cust_time_dict.update({shuffled[i]: processing_time})
                                srv_one_pose_time_dict.update({len(queue_1.queue):serv_time_one[i-1]})
    
                                srv_one_cust_pose_dict.update({shuffled[i]: len(queue_1.queue)})  # i+1})
                                srv_one_lst_arrivals.append(shuffled[i])
                                # spawn a dummy process here
    
                                customer_processes_in_one.update(start_customer_process(shuffled[i], queueID))
    
                                jockey_flag: bool = jockey_selector(queue_1, processing_time, shuffled[i], queueID,
                                                                    preferred_queue, jockey_threshold, preferred_que_name)
                                if jockey_flag:
                                    jockeys_in_one.append(shuffled[i])
                                    srv_two_cust_time_dict.update({shuffled[i]: processing_time})  # time_in})
                                    srv_two_cust_pose_dict.update({shuffled[i]: len(queue_1.queue)})
    
                                if len(jockeys_in_one) > 1:
                                    jockey_rate = get_jockeying_rate(jockeys_in_one, srv_one_lst_arrivals, len(list(set(queue_1.queue))), len(list(set(queue_2.queue)))) #queue_size_now, alt_queue_size_now)
                                    print(colored("Customers jockeyed from %s at a rate of %6.4f", 'yellow',
                                                attrs=["blink"]) % (queueID, jockey_rate))
                                    if jockey_threshold:
                                        jockeying_rates_with_threshold_in_one.update({jockey_threshold: jockey_rate})
                                    else:
                                        jockeying_rates_in_queue_one.append(jockey_rate)
    
                                all_service_times_queue_one.append(processing_time)
    
                                save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, queueID)
                                save_pose_waits_to_file(srv_one_pose_time_dict, queueID)
                                new_joins_in_one += 1
    
                            # elif queue_2.qsize() <= queue_1.qsize():
                            elif len(list(queue_2.queue)) <= len(list(queue_1.queue)):
    
                                time_in = time.time()
                                task_list_two = populate_task_into_queue(queue_2, shuffled[i])
                                # queueID = "Server2"
                                queueID = list(dict_queues.keys())[1]
                                #preferred_que_name = "Server1"
                                preferred_que_name = list(dict_queues.keys())[0]
                                preferred_queue = queue_1  
                                queue_size_now = len(list(set(queue_2.queue)))

                                alt_queue_size_now = len(list(set(queue_1.queue)))
                                    
                                dict_server_customer_queue_two.update({shuffled[i]: queueID})
                                # print("%s size ===>> %01d Wait::: %06.4f "%( queueID, alt_queue_size_now, float(alt_queue_size_now/args.arrival_rate)))
                                processing_time = compute_time_spent_in_service(serv_time_two, queue_size_now,
                                                                                all_serv_time_two)
                                print(
                                    "%s joined %s at position %01d. Queue1 has %01d customers. %6.7f seconds will be needed to service completion"
                                    % (shuffled[i], queueID, queue_size_now+1, alt_queue_size_now, processing_time))
    
                                srv_two_cust_time_dict.update({shuffled[i]: processing_time})
                                srv_two_pose_time_dict.update({queue_2.qsize():serv_time_one[i-1]})
                                
                                srv_two_cust_pose_dict.update({shuffled[i]: len(queue_2.queue)})  # i+1})
                                srv_two_lst_arrivals.append(shuffled[i])
                                # spawn process here
                                customer_processes_in_two.update(start_customer_process(shuffled[i], queueID))
    
                                jockey_flag: bool = jockey_selector(queue_2, processing_time, shuffled[i], queueID,
                                                                    preferred_queue, jockey_threshold, preferred_que_name)
                                if jockey_flag:
                                    jockeys_in_two.append(shuffled[i])
                                    srv_one_cust_time_dict.update({shuffled[i]: processing_time})  # time_in})
                                    srv_one_cust_pose_dict.update({shuffled[i]: len(queue_2.queue)})
    
                                if len(jockeys_in_two) > 1:
                                    jockey_rate = get_jockeying_rate(jockeys_in_two, srv_two_lst_arrivals, queue_size_now, alt_queue_size_now)
                                    print(colored("Customers jockeyed from %s at a rate of %6.4f", 'yellow',
                                                attrs=["blink"]) % (queueID, jockey_rate))
                                    if jockey_threshold:
                                        jockeying_rates_with_threshold_in_two.update({jockey_threshold: jockey_rate})
                                    else:
                                        jockeying_rates_in_queue_two.append(jockey_rate)
    
                                all_service_times_queue_two.append(processing_time)
    
                                save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, queueID)
                                save_pose_waits_to_file(srv_two_pose_time_dict, queueID)
                                new_joins_in_two += 1
                                
                else:
                    for i in range(len(arrivals)):
                        local_arrivals_in_one = []
                        local_arrivals_in_two = []

                        # if queue_1.qsize() <= queue_2.qsize():
                        if len(queue_1.queue) <= len(queue_2.queue):
                            
                            queueID = list(dict_queues.keys())[0]                                
                            preferred_que_name = list(dict_queues.keys())[1]
                            
                            preferred_queue = queue_2
                            queue_size_now = len(list(queue_1.queue))
                            
                            #if len(list(queue_2.queue)) == 0:
                            alt_queue_size_now = len(list(queue_2.queue))

                            task_list_one = populate_task_into_queue(queue_1, arrivals[i] )
                            dict_server_customer_queue_one.update({arrivals[i]:queueID})
                            # print("%s size ===>> %01d Wait::: %06.4f "%( queueID, alt_queue_size_now, float(alt_queue_size_now/args.arrival_rate)))
                            processing_time = compute_time_spent_in_service(serv_time_one, queue_size_now, all_serv_time_one)
                            print("%s joined %s at position %01d. Queue2 one has %01d customers. %6.7f seconds will be needed to service completion."%(arrivals[i], queueID, queue_size_now+1, alt_queue_size_now, processing_time))
                            srv_one_cust_time_dict.update({arrivals[i]: processing_time})
                            srv_one_pose_time_dict.update({len(queue_1.queue):serv_time_one[i-1]})
                            
                            srv_one_cust_pose_dict.update({arrivals[i]: len(queue_1.queue)}) #i+1})
                            srv_one_lst_arrivals.append(arrivals[i])
                            # spawn a dummy process here

                            customer_processes_in_one.update(start_customer_process(arrivals[i], queueID))

                            jockey_flag: bool = jockey_selector(queue_1, processing_time, arrivals[i], queueID, preferred_queue, jockey_threshold, preferred_que_name)
                            if jockey_flag:
                                jockeys_in_one.append(arrivals[i]+"_jockey")
                                srv_two_cust_time_dict.update({arrivals[i]: processing_time}) # time_in})
                                srv_two_cust_pose_dict.update({arrivals[i]: len(queue_1.queue)})

                            if len(jockeys_in_one) > 1:
                                jockey_rate = get_jockeying_rate(jockeys_in_one, srv_one_lst_arrivals,  len(list(set(queue_1.queue))), len(list(set(queue_2.queue))))
                                print(colored("Customers jockeyed from %s at a rate of %6.4f",'yellow', attrs=["blink"])%(queueID, jockey_rate))
                                if jockey_threshold:
                                    jockeying_rates_with_threshold_in_one.update({jockey_threshold:jockey_rate})
                                else:
                                    jockeying_rates_in_queue_one.append(jockey_rate)

                            all_service_times_queue_one.append(processing_time)

                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_one, queueID)
                            save_pose_waits_to_file(srv_one_pose_time_dict, queueID)
                            new_joins_in_one += 1

                        #elif queue_2.qsize() <= queue_1.qsize():
                        elif len(list(queue_2.queue)) <= len(list(queue_2.queue)):

                            time_in = time.time()
                            task_list_two = populate_task_into_queue(queue_2, arrivals[i] )
                            #queueID = "Server2"
                            #preferred_que_name = "Server1"
                            queueID = list(dict_queues.keys())[1]                                
                            preferred_que_name = list(dict_queues.keys())[0]
                            preferred_queue = queue_1

                            queue_size_now = len(list(queue_2.queue))

                            alt_queue_size_now = len(list(queue_1.queue))
                            # queue_size_now = len(list(queue_2.queue))
                            dict_server_customer_queue_two.update({arrivals[i]:queueID})
                            # print("%s size ===>> %01d Wait::: %6.4f "%( queueID, alt_queue_size_now, float(alt_queue_size_now/args.arrival_rate)))
                            processing_time = compute_time_spent_in_service(serv_time_two, queue_size_now, all_serv_time_two)
                            print("%s joined %s at position %01d. Queue1 has %01d customers. %6.7f seconds will be needed to service completion"
                                        %(arrivals[i], queueID, queue_size_now+1, alt_queue_size_now, processing_time))

                            # Total expected waiting time till service completion
                            srv_two_cust_time_dict.update({arrivals[i]: processing_time})
                            # Time customer will spend at the landing position
                            srv_two_pose_time_dict.update({len(queue_2.queue):serv_time_two[i-1]})
                            
                            srv_two_cust_pose_dict.update({arrivals[i]: len(queue_2.queue)}) #i+1})
                            srv_two_lst_arrivals.append(arrivals[i])
                            # spawn process here
                            customer_processes_in_two.update(start_customer_process(arrivals[i], queueID))

                            jockey_flag: bool = jockey_selector(queue_2, processing_time, arrivals[i], queueID, preferred_queue, jockey_threshold, preferred_que_name)
                            if jockey_flag:
                                jockeys_in_two.append(arrivals[i]+"_jockey")
                                srv_two_cust_time_dict.update({arrivals[i]: processing_time}) #time_in})
                                srv_two_cust_pose_dict.update({arrivals[i]: len(queue_2.queue)})


                            if len(jockeys_in_two) > 1:
                                jockey_rate = get_jockeying_rate(jockeys_in_two, srv_two_lst_arrivals, len(list(set(queue_2.queue))), len(list(set(queue_1.queue)))) #queue_size_now, alt_queue_size_now)
                                print(colored("Customers jockeyed from %s at a rate of %6.4f",'yellow', attrs=["blink"])%(queueID, jockey_rate))
                                if jockey_threshold:
                                    jockeying_rates_with_threshold_in_two.update({jockey_threshold:jockey_rate})
                                else:
                                    jockeying_rates_in_queue_two.append(jockey_rate)

                            all_service_times_queue_two.append(processing_time)

                            save_jockey_details_to_file(jockeying_rates_with_threshold_in_two, queueID)
                            save_pose_waits_to_file(srv_two_pose_time_dict, queueID)
                            new_joins_in_two += 1

                unused_waiting_times_in_one = serv_time_one[new_joins_in_one:]
                unused_waiting_times_in_two = serv_time_two[new_joins_in_two:]

                global_arrivals_in_one.append(srv_one_lst_arrivals)
                global_arrivals_in_two.append(srv_two_lst_arrivals)

                '''
                    Process queues for a random period t
                    It was suggested that inorder to keep the randomness low, 
                    we should fix some of these environment parameters
                    
                '''
                # Assuming the rate at which customers enter queue is 
                # equal to the rate at which they leave the system
                
                count_processed_in_one = 0  
                count_processed_in_two = 0
                
                for count in range(args.arrival_rate): # (t+(min(queue_1.qsize(), queue_2.qsize()))):
                    q_selector = random.randint(1, 2)
                    if q_selector == 1:
                        # queueID = "Server1"
                        queueID = list(dict_queues.keys())[0]                                
                        preferred_queue = queue_2
                        current_user = start_processing_queues(queue_1, srv_one_cust_time_dict, srv_two_cust_time_dict,
                                                srv_one_cust_pose_dict, jockeys_in_one, jockeys_in_two, queueID,
                                                preferred_queue, srv_one_lst_arrivals, dict_server_customer_queue_one,
                                                dict_server_customer_queue_two, serv_time_one, serv_time_two,
                                                all_serv_time_one, unused_waiting_times_in_one, local_arrivals_in_one)
                        count_processed_in_one += 1
                    else:
                        # queueID = "Server2"
                        queueID = list(dict_queues.keys())[1]                                
                        preferred_queue = queue_1
                        current_user = start_processing_queues(queue_2, srv_two_cust_time_dict, srv_one_cust_time_dict,
                                                srv_two_cust_pose_dict, jockeys_in_two, jockeys_in_two, queueID,
                                                preferred_queue, srv_two_lst_arrivals, dict_server_customer_queue_two,
                                                dict_server_customer_queue_one, serv_time_one, serv_time_two,
                                                all_serv_time_two, unused_waiting_times_in_two, local_arrivals_in_two)
                        count_processed_in_two += 1
            # time.sleep(0.5)
                    
            else:
                print("There are currently no customers interested in your services..")                

                continue

            time_now += serv_time_one[0]
            
            arrivals_batch_num += 1

            all_serv_time_one.extend(serv_time_one)
            all_serv_time_two.extend(serv_time_two)
                        
        # print("STATUS::----->> ", flag_no_jockeys)
        if flag_no_jockeys == False:
            jockey_count = 0
            no_jockeyed_thresh_count_diff = []
            no_jockeyed_thresh_count_wait_diff = []
            datasource_diffs = base_dir+'/constant/all_queues_jockeying_serv_diff.csv'
            datasource_jockey_details = base_dir+"/constant/all_que_jockeying_threshold_count_diff.csv"
            datasource_jockey_details_extra = base_dir+"/constant/all_que_jockeying_threshold_count_totalwaitingtime_diff.csv"
            no_jockeyed_thresh_count_diff.append([args.jockeying_threshold, jockey_count, abs(args.service_rate_one - args.service_rate_two)])            
    
            save_jockey_waiting_time(no_jockeyed_thresh_count_diff, datasource_jockey_details)
            save_jockey_waiting_time(no_jockeyed_thresh_count_diff, datasource_diffs)
            
            if queueID == "Server1":
                total_wait_time = srv_one_cust_time_dict.get(current_user)
                if total_wait_time is not None:
                # print("DUMP -->> ", total_wait_time, current_user)
                    no_jockeyed_thresh_count_wait_diff.append([args.jockeying_threshold, jockey_count, total_wait_time ,abs(args.service_rate_one - args.service_rate_two)]) 
                    save_jockey_waiting_time(no_jockeyed_thresh_count_wait_diff, datasource_jockey_details_extra)
            else:
                total_wait_time = srv_two_cust_time_dict.get(current_user)
                if total_wait_time is not None:
                    no_jockeyed_thresh_count_wait_diff.append([args.jockeying_threshold, jockey_count, total_wait_time ,abs(args.service_rate_one - args.service_rate_two)]) 
                # print("DUMP ==>> ", total_wait_time, current_user)
                           
                    save_jockey_waiting_time(no_jockeyed_thresh_count_wait_diff, datasource_jockey_details_extra)

        diff_service_rates.append(abs(args.service_rate_one - args.service_rate_two))
        save_diff_serv_rates(diff_service_rates, service_rates_difference_file) 
        
        count_processed_in_one = 0  
        count_processed_in_two = 0

# write number of Monte Carlo runs to file
    
# save_runs_count_to_file(args.run_id)
 
# plot_results_3D(jockey_details_source_one, jockey_details_source_two)
# plot_results(serv_rates_jockeying_file_one, serv_rates_jockeying_file_two)
# plot_results(jockey_stats_file_one, jockey_stats_file_two)
# plot_results(que_length_jockey_stats_one, que_length_jockey_stats_two)
# plot_results(jockey_queu_length_waiting_time_one, jockey_queu_length_waiting_time_two)
        
            
'''
    - There are duplicate in the re-ordred list, get rid of them by checking
    - There are also cases where the finished processed also show up repetitively in the leaving customers.
    - Re-ordered appears now mixed with the jockey candidates from either queues and this messes up thing
        Need to make each jockey choose a given queue that is opposite to the current one.
    - Fix the infinite looping that was observed and understand the cause of the the problem. Actually the problem 
    arises in the current state when we set the arrival rate to about 5 with a low jockeying rate say 2. 
        This orchestate the looping attack where one customer keeps jockeying from one queue to another.
        Hence no new arrivals are allowed into any of the queue due to the back and forth jockeying
    - Fix the getting stuck when the jockeying threshold is set to 1
    - * - Sometthing that was mentiones by Bin Han about mixing the jockekying threshold with waiting time metrics
    - * - If a customer has just jockeyed to an alternative queue and then it appears again in the list of candidate jockeys then .........?
'''
