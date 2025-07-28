#!/usr/bin/env python3
"""
Comprehensive example demonstrating the new per-server rate computation functionality.

This script shows how to:
1. Use the new compute_reneging_rate_per_server and compute_jockeying_rate_per_server methods
2. Analyze rates by server and information source (state_subscribers vs nn_subscribers)
3. Generate visualizations showing the differences
4. Compare the new methods with the old aggregate methods
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from activity import RequestQueue

def create_simulation_with_mixed_behaviors():
    """Create a simulation with different behaviors across servers and information sources."""
    print("Creating RequestQueue simulation...")
    
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef)
    
    # Add a significant number of requests to get meaningful statistics
    print("Adding requests to simulation...")
    for i in range(20):
        requestObj.addNewRequest(expected_time_to_service_end=8.0, batchid=i)
    
    print(f"Total requests created: {len(requestObj.state_subscribers) + len(requestObj.nn_subscribers)}")
    print(f"State subscribers: {len(requestObj.state_subscribers)}")
    print(f"NN subscribers: {len(requestObj.nn_subscribers)}")
    
    # Simulate different behaviors per server and information source
    print("\nSimulating different reneging/jockeying behaviors...")
    
    # Server 1 state subscribers: High reneging, low jockeying
    server_1_state = [req for req in requestObj.state_subscribers if req.server_id == "1"]
    for i, req in enumerate(server_1_state):
        if i % 3 == 0:  # 33% renege
            req.customerid += "_reneged"
        elif i % 10 == 0:  # 10% jockey
            req.customerid += "_jockeyed"
    
    # Server 1 NN subscribers: Low reneging, high jockeying
    server_1_nn = [req for req in requestObj.nn_subscribers if req.server_id == "1"]
    for i, req in enumerate(server_1_nn):
        if i % 10 == 0:  # 10% renege
            req.customerid += "_reneged"
        elif i % 2 == 0:  # 50% jockey
            req.customerid += "_jockeyed"
    
    # Server 2 state subscribers: Moderate reneging and jockeying
    server_2_state = [req for req in requestObj.state_subscribers if req.server_id == "2"]
    for i, req in enumerate(server_2_state):
        if i % 5 == 0:  # 20% renege
            req.customerid += "_reneged"
        elif i % 4 == 0:  # 25% jockey
            req.customerid += "_jockeyed"
    
    # Server 2 NN subscribers: Low reneging, low jockeying
    server_2_nn = [req for req in requestObj.nn_subscribers if req.server_id == "2"]
    for i, req in enumerate(server_2_nn):
        if i % 8 == 0:  # 12.5% renege
            req.customerid += "_reneged"
        elif i % 6 == 0:  # 16.7% jockey
            req.customerid += "_jockeyed"
    
    return requestObj

def demonstrate_new_methods(requestObj):
    """Demonstrate the new per-server rate computation methods."""
    print("\n" + "="*80)
    print("DEMONSTRATING NEW PER-SERVER RATE COMPUTATION METHODS")
    print("="*80)
    
    # Method 1: Compute rates for specific server and information source
    print("\n1. Computing rates for specific server and information source:")
    print("-" * 60)
    
    # Server 1 rates
    server_1_state_reneging = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "1")
    server_1_state_jockeying = requestObj.compute_jockeying_rate_per_server(requestObj.state_subscribers, "1")
    server_1_nn_reneging = requestObj.compute_reneging_rate_per_server(requestObj.nn_subscribers, "1")
    server_1_nn_jockeying = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "1")
    
    print(f"Server 1 - State Subscribers:")
    print(f"  Reneging Rate: {server_1_state_reneging:.3f}")
    print(f"  Jockeying Rate: {server_1_state_jockeying:.3f}")
    print(f"Server 1 - NN Subscribers:")
    print(f"  Reneging Rate: {server_1_nn_reneging:.3f}")
    print(f"  Jockeying Rate: {server_1_nn_jockeying:.3f}")
    
    # Server 2 rates
    server_2_state_reneging = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "2")
    server_2_state_jockeying = requestObj.compute_jockeying_rate_per_server(requestObj.state_subscribers, "2")
    server_2_nn_reneging = requestObj.compute_reneging_rate_per_server(requestObj.nn_subscribers, "2")
    server_2_nn_jockeying = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "2")
    
    print(f"Server 2 - State Subscribers:")
    print(f"  Reneging Rate: {server_2_state_reneging:.3f}")
    print(f"  Jockeying Rate: {server_2_state_jockeying:.3f}")
    print(f"Server 2 - NN Subscribers:")
    print(f"  Reneging Rate: {server_2_nn_reneging:.3f}")
    print(f"  Jockeying Rate: {server_2_nn_jockeying:.3f}")
    
    # Method 2: Get comprehensive summary
    print("\n2. Using the comprehensive summary method:")
    print("-" * 60)
    summary = requestObj.get_rates_summary_per_server_and_source()
    
    for server_name, server_data in summary.items():
        print(f"\n{server_name.replace('_', ' ').title()}:")
        for info_source, rates in server_data.items():
            print(f"  {info_source.replace('_', ' ').title()}:")
            print(f"    Requests: {rates['count']}")
            print(f"    Reneging Rate: {rates['reneging_rate']:.3f}")
            print(f"    Jockeying Rate: {rates['jockeying_rate']:.3f}")

def compare_with_old_methods(requestObj):
    """Compare new methods with old aggregate methods."""
    print("\n" + "="*80)
    print("COMPARISON WITH OLD AGGREGATE METHODS")
    print("="*80)
    
    # Old method - compute for entire queues
    queue_1 = requestObj.dict_queues_obj["1"]
    queue_2 = requestObj.dict_queues_obj["2"]
    
    old_renege_1 = requestObj.compute_reneging_rate(queue_1)
    old_jockey_1 = requestObj.compute_jockeying_rate(queue_1)
    old_renege_2 = requestObj.compute_reneging_rate(queue_2)
    old_jockey_2 = requestObj.compute_jockeying_rate(queue_2)
    
    print("\nOld Aggregate Method (per queue, all information sources combined):")
    print("-" * 70)
    print(f"Server 1 - Reneging: {old_renege_1:.3f}, Jockeying: {old_jockey_1:.3f}")
    print(f"Server 2 - Reneging: {old_renege_2:.3f}, Jockeying: {old_jockey_2:.3f}")
    
    # New method - separate by information source
    summary = requestObj.get_rates_summary_per_server_and_source()
    
    print("\nNew Per-Server and Per-Information-Source Method:")
    print("-" * 70)
    for server in ["server_1", "server_2"]:
        server_num = server.split("_")[1]
        state_data = summary[server]["state_subscribers"]
        nn_data = summary[server]["nn_subscribers"]
        
        print(f"Server {server_num}:")
        print(f"  State Subscribers - Reneging: {state_data['reneging_rate']:.3f}, Jockeying: {state_data['jockeying_rate']:.3f} (n={state_data['count']})")
        print(f"  NN Subscribers    - Reneging: {nn_data['reneging_rate']:.3f}, Jockeying: {nn_data['jockeying_rate']:.3f} (n={nn_data['count']})")
    
    print("\nAdvantages of New Method:")
    print("- Distinguishes between different information sources")
    print("- Allows targeted analysis of subscriber behavior")
    print("- Enables optimization of strategies per information source")
    print("- Provides granular insights for decision making")

def generate_visualization(requestObj):
    """Generate visualization comparing rates."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    
    try:
        # Create the enhanced plot
        summary = requestObj.get_rates_summary_per_server_and_source()
        
        servers = ["server_1", "server_2"]
        info_sources = ["state_subscribers", "nn_subscribers"]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reneging and Jockeying Rates by Server and Information Source', fontsize=16)
        
        for i, server in enumerate(servers):
            # Reneging rates
            reneging_rates = [summary[server][src]["reneging_rate"] for src in info_sources]
            axs[i, 0].bar(info_sources, reneging_rates, color=['skyblue', 'lightcoral'])
            axs[i, 0].set_title(f'{server.replace("_", " ").title()} - Reneging Rates')
            axs[i, 0].set_ylabel('Reneging Rate')
            axs[i, 0].set_ylim(0, max(max(reneging_rates), 0.1) * 1.2)
            
            # Add count labels on bars
            for j, src in enumerate(info_sources):
                count = summary[server][src]["count"]
                rate = summary[server][src]["reneging_rate"]
                if rate > 0:
                    axs[i, 0].text(j, rate + 0.01, f'n={count}', ha='center', va='bottom')
            
            # Jockeying rates  
            jockeying_rates = [summary[server][src]["jockeying_rate"] for src in info_sources]
            axs[i, 1].bar(info_sources, jockeying_rates, color=['lightgreen', 'orange'])
            axs[i, 1].set_title(f'{server.replace("_", " ").title()} - Jockeying Rates')
            axs[i, 1].set_ylabel('Jockeying Rate')
            axs[i, 1].set_ylim(0, max(max(jockeying_rates), 0.1) * 1.2)
            
            # Add count labels on bars
            for j, src in enumerate(info_sources):
                count = summary[server][src]["count"]
                rate = summary[server][src]["jockeying_rate"]
                if rate > 0:
                    axs[i, 1].text(j, rate + 0.01, f'n={count}', ha='center', va='bottom')
        
        # Improve x-axis labels
        for ax in axs.flat:
            ax.set_xticklabels(['State\nSubscribers', 'NN\nSubscribers'])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('per_server_rates_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'per_server_rates_comparison.png'")
        
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")

def main():
    """Main demonstration function."""
    print("="*80)
    print("QUEUING THEORY: PER-SERVER RATE COMPUTATION DEMONSTRATION")
    print("="*80)
    print("This example demonstrates the new functionality for computing")
    print("reneging and jockeying rates per server and per information source.")
    print()
    
    # Create simulation
    requestObj = create_simulation_with_mixed_behaviors()
    
    # Demonstrate new methods
    demonstrate_new_methods(requestObj)
    
    # Compare with old methods
    compare_with_old_methods(requestObj)
    
    # Generate visualization
    generate_visualization(requestObj)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    print("Key takeaways:")
    print("1. New methods provide granular analysis per server and information source")
    print("2. Enable targeted optimization strategies")
    print("3. Maintain backward compatibility with existing code")
    print("4. Support comprehensive visualization and reporting")

if __name__ == "__main__":
    main()