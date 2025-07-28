#!/usr/bin/env python3
"""
Test script to validate the new per-server rate computation methods.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from activity import RequestQueue

def test_per_server_rate_computation():
    """Test the new per-server rate computation methods."""
    print("Testing per-server rate computation...")
    
    # Create a RequestQueue instance
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef)
    
    # Add some test requests to both servers
    print("Adding test requests...")
    for i in range(5):
        requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
    
    # Manually modify some request IDs to simulate reneging and jockeying
    if requestObj.state_subscribers:
        if len(requestObj.state_subscribers) > 0:
            requestObj.state_subscribers[0].customerid += "_reneged"
        if len(requestObj.state_subscribers) > 1:
            requestObj.state_subscribers[1].customerid += "_jockeyed"
    
    if requestObj.nn_subscribers:
        if len(requestObj.nn_subscribers) > 0:
            requestObj.nn_subscribers[0].customerid += "_reneged"
    
    # Test the new per-server methods
    print("\nTesting per-server rate methods...")
    
    # Test for server "1"
    state_reneging_1 = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "1")
    state_jockeying_1 = requestObj.compute_jockeying_rate_per_server(requestObj.state_subscribers, "1")
    nn_reneging_1 = requestObj.compute_reneging_rate_per_server(requestObj.nn_subscribers, "1")
    nn_jockeying_1 = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "1")
    
    print(f"Server 1 - State Subscribers: Reneging={state_reneging_1:.2f}, Jockeying={state_jockeying_1:.2f}")
    print(f"Server 1 - NN Subscribers: Reneging={nn_reneging_1:.2f}, Jockeying={nn_jockeying_1:.2f}")
    
    # Test for server "2"
    state_reneging_2 = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "2")
    state_jockeying_2 = requestObj.compute_jockeying_rate_per_server(requestObj.state_subscribers, "2")
    nn_reneging_2 = requestObj.compute_reneging_rate_per_server(requestObj.nn_subscribers, "2")
    nn_jockeying_2 = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "2")
    
    print(f"Server 2 - State Subscribers: Reneging={state_reneging_2:.2f}, Jockeying={state_jockeying_2:.2f}")
    print(f"Server 2 - NN Subscribers: Reneging={nn_reneging_2:.2f}, Jockeying={nn_jockeying_2:.2f}")
    
    # Test the summary method
    print("\nTesting summary method...")
    summary = requestObj.get_rates_summary_per_server_and_source()
    print("Summary:", summary)
    
    # Verify subscriber counts
    print(f"\nSubscriber counts:")
    print(f"State subscribers: {len(requestObj.state_subscribers)}")
    print(f"NN subscribers: {len(requestObj.nn_subscribers)}")
    
    # Check server_id assignment
    print(f"\nServer ID assignments:")
    for i, req in enumerate(requestObj.state_subscribers):
        print(f"State subscriber {i}: server_id={req.server_id}")
    for i, req in enumerate(requestObj.nn_subscribers):
        print(f"NN subscriber {i}: server_id={req.server_id}")
    
    print("\nTest completed successfully!")

def test_example_usage():
    """Demonstrate example usage of the new methods."""
    print("\n" + "="*50)
    print("EXAMPLE USAGE")
    print("="*50)
    
    # Create RequestQueue
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef)
    
    # Simulate adding requests and some renege/jockey
    print("1. Creating simulation with requests and actions...")
    for i in range(10):
        requestObj.addNewRequest(expected_time_to_service_end=8.0, batchid=i)
    
    # Simulate some reneging and jockeying
    if len(requestObj.state_subscribers) >= 2:
        requestObj.state_subscribers[0].customerid += "_reneged"
        requestObj.state_subscribers[1].customerid += "_jockeyed"
    if len(requestObj.nn_subscribers) >= 2:
        requestObj.nn_subscribers[0].customerid += "_jockeyed"  
        requestObj.nn_subscribers[1].customerid += "_reneged"
    
    print("\n2. Computing rates per server and information source...")
    
    # Example 1: Get rates for specific server and subscriber type
    server_1_state_reneging = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "1")
    server_1_nn_jockeying = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "1")
    
    print(f"Server 1 state subscribers reneging rate: {server_1_state_reneging:.2f}")
    print(f"Server 1 NN subscribers jockeying rate: {server_1_nn_jockeying:.2f}")
    
    # Example 2: Get comprehensive summary
    print("\n3. Comprehensive rates summary:")
    summary = requestObj.get_rates_summary_per_server_and_source()
    
    for server_name, server_data in summary.items():
        print(f"\n{server_name.replace('_', ' ').title()}:")
        for info_source, rates in server_data.items():
            print(f"  {info_source.replace('_', ' ').title()}:")
            print(f"    Requests: {rates['count']}")
            print(f"    Reneging Rate: {rates['reneging_rate']:.2f}")
            print(f"    Jockeying Rate: {rates['jockeying_rate']:.2f}")

if __name__ == "__main__":
    test_per_server_rate_computation()
    test_example_usage()