import unittest

import functions_module as fm

class Tester(unittest.TestCase):

    def test_merger(self):
        var1 = [1, 3, 5]
        var2 = [2, 4, 6]
        result = fm.merge_two_lists(var1, var2)
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])

    def index_check(self):
        var = ['one', 'two', 'three']
        result = fm.add_ind(var)
        self.assertEqual(result, [(0, 'one'), (1, 'two'), (2, 'three')])

if __name__ == '__main__':
    unittest.main()
