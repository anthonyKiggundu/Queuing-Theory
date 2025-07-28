#/usr/bin/python3
import unittest

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

