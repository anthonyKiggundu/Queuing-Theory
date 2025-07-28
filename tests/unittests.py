#/usr/bin/python3
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from activity import RequestQueue

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


class TestPerServerRateComputation(unittest.TestCase):
    """Test the new per-server rate computation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.utility_basic = 1.0
        self.discount_coef = 0.1
        self.requestObj = RequestQueue(self.utility_basic, self.discount_coef)
    
    def test_server_id_assignment(self):
        """Test that server_id is properly assigned to requests."""
        # Add a few requests
        for i in range(4):
            self.requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
        
        # Check that all requests have server_id assigned
        all_requests = self.requestObj.state_subscribers + self.requestObj.nn_subscribers
        for req in all_requests:
            self.assertIsNotNone(req.server_id)
            self.assertIn(req.server_id, ["1", "2"])
    
    def test_compute_reneging_rate_per_server(self):
        """Test the compute_reneging_rate_per_server method."""
        # Add requests
        for i in range(6):
            self.requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
        
        # Manually simulate reneging for some requests
        if self.requestObj.state_subscribers:
            self.requestObj.state_subscribers[0].customerid += "_reneged"
        if self.requestObj.nn_subscribers:
            self.requestObj.nn_subscribers[0].customerid += "_reneged"
        
        # Test rate computation
        rate_state_1 = self.requestObj.compute_reneging_rate_per_server(self.requestObj.state_subscribers, "1")
        rate_nn_2 = self.requestObj.compute_reneging_rate_per_server(self.requestObj.nn_subscribers, "2")
        
        # Rates should be between 0 and 1
        self.assertGreaterEqual(rate_state_1, 0.0)
        self.assertLessEqual(rate_state_1, 1.0)
        self.assertGreaterEqual(rate_nn_2, 0.0)
        self.assertLessEqual(rate_nn_2, 1.0)
    
    def test_compute_jockeying_rate_per_server(self):
        """Test the compute_jockeying_rate_per_server method."""
        # Add requests
        for i in range(4):
            self.requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
        
        # Manually simulate jockeying for some requests  
        if self.requestObj.state_subscribers:
            self.requestObj.state_subscribers[0].customerid += "_jockeyed"
        if self.requestObj.nn_subscribers:
            self.requestObj.nn_subscribers[0].customerid += "_jockeyed"
        
        # Test rate computation
        rate_state_1 = self.requestObj.compute_jockeying_rate_per_server(self.requestObj.state_subscribers, "1")
        rate_nn_2 = self.requestObj.compute_jockeying_rate_per_server(self.requestObj.nn_subscribers, "2")
        
        # Rates should be between 0 and 1
        self.assertGreaterEqual(rate_state_1, 0.0)
        self.assertLessEqual(rate_state_1, 1.0)
        self.assertGreaterEqual(rate_nn_2, 0.0)
        self.assertLessEqual(rate_nn_2, 1.0)
    
    def test_empty_subscribers_list(self):
        """Test behavior with empty subscribers list."""
        # Test with empty lists
        rate_renege = self.requestObj.compute_reneging_rate_per_server([], "1")
        rate_jockey = self.requestObj.compute_jockeying_rate_per_server([], "1")
        
        self.assertEqual(rate_renege, 0)
        self.assertEqual(rate_jockey, 0)
    
    def test_rates_summary(self):
        """Test the get_rates_summary_per_server_and_source method."""
        # Add some requests
        for i in range(4):
            self.requestObj.addNewRequest(expected_time_to_service_end=10.0, batchid=i)
        
        summary = self.requestObj.get_rates_summary_per_server_and_source()
        
        # Check structure
        self.assertIn("server_1", summary)
        self.assertIn("server_2", summary)
        
        for server in ["server_1", "server_2"]:
            self.assertIn("state_subscribers", summary[server])
            self.assertIn("nn_subscribers", summary[server])
            
            for subscriber_type in ["state_subscribers", "nn_subscribers"]:
                self.assertIn("reneging_rate", summary[server][subscriber_type])
                self.assertIn("jockeying_rate", summary[server][subscriber_type])
                self.assertIn("count", summary[server][subscriber_type])
                
                # Verify rates are valid
                reneging_rate = summary[server][subscriber_type]["reneging_rate"]
                jockeying_rate = summary[server][subscriber_type]["jockeying_rate"]
                self.assertGreaterEqual(reneging_rate, 0.0)
                self.assertLessEqual(reneging_rate, 1.0)
                self.assertGreaterEqual(jockeying_rate, 0.0)
                self.assertLessEqual(jockeying_rate, 1.0)

