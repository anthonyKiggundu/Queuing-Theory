#!/usr/bin/env python3
"""
Final verification test to ensure all functionality works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from activity import RequestQueue

def final_verification():
    """Run a final verification of all new functionality."""
    print("=" * 60)
    print("FINAL VERIFICATION OF PER-SERVER RATE COMPUTATION")
    print("=" * 60)
    
    # Create RequestQueue
    requestObj = RequestQueue(utility_basic=1.0, discount_coef=0.1)
    
    # Add requests
    print("1. Adding requests...")
    for i in range(8):
        requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
    
    # Verify server_id assignment
    print("2. Verifying server_id assignment...")
    all_requests = requestObj.state_subscribers + requestObj.nn_subscribers
    for req in all_requests:
        assert hasattr(req, 'server_id'), "Request missing server_id"
        assert req.server_id in ["1", "2"], f"Invalid server_id: {req.server_id}"
    print("   ✓ All requests have valid server_id")
    
    # Simulate some behaviors
    print("3. Simulating behaviors...")
    if requestObj.state_subscribers:
        requestObj.state_subscribers[0].customerid += "_reneged"
        requestObj.state_subscribers[1].customerid += "_jockeyed"
    if requestObj.nn_subscribers:
        requestObj.nn_subscribers[0].customerid += "_jockeyed"
    
    # Test new methods
    print("4. Testing new methods...")
    rate1 = requestObj.compute_reneging_rate_per_server(requestObj.state_subscribers, "1")
    rate2 = requestObj.compute_jockeying_rate_per_server(requestObj.nn_subscribers, "2")
    assert 0 <= rate1 <= 1, f"Invalid rate: {rate1}"
    assert 0 <= rate2 <= 1, f"Invalid rate: {rate2}"
    print(f"   ✓ Server 1 state reneging rate: {rate1:.3f}")
    print(f"   ✓ Server 2 NN jockeying rate: {rate2:.3f}")
    
    # Test summary method
    print("5. Testing summary method...")
    summary = requestObj.get_rates_summary_per_server_and_source()
    assert "server_1" in summary, "Missing server_1 in summary"
    assert "server_2" in summary, "Missing server_2 in summary"
    for server in ["server_1", "server_2"]:
        assert "state_subscribers" in summary[server], f"Missing state_subscribers for {server}"
        assert "nn_subscribers" in summary[server], f"Missing nn_subscribers for {server}"
    print("   ✓ Summary method works correctly")
    
    # Test backward compatibility
    print("6. Testing backward compatibility...")
    old_rate1 = requestObj.compute_reneging_rate(requestObj.dict_queues_obj["1"])
    old_rate2 = requestObj.compute_jockeying_rate(requestObj.dict_queues_obj["2"])
    assert 0 <= old_rate1 <= 1, f"Invalid old rate: {old_rate1}"
    assert 0 <= old_rate2 <= 1, f"Invalid old rate: {old_rate2}"
    print("   ✓ Old methods still work")
    
    print("\n" + "=" * 60)
    print("✅ ALL VERIFICATION TESTS PASSED!")
    print("=" * 60)
    print("The implementation successfully provides:")
    print("• Per-server rate computation")
    print("• Information source distinction")
    print("• Backward compatibility")
    print("• Comprehensive testing")
    print("• Enhanced visualization")

if __name__ == "__main__":
    final_verification()