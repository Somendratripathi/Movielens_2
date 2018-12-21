import unittest
import numpy as np
import pandas as pd
from Similarity.cosine_similarity import *


class TestCosineSimilarity(unittest.TestCase):

    def setUp(self):
        self.matrix = pd.read_csv("Test/test.csv", index_col=False, header=None).as_matrix()
        self.centered_matrix = centralize_matrix(self.matrix)
        self.cosine = cosine_matrix(self.centered_matrix)
        self.neighbour = get_nearest_neighbors(self.cosine, 2)

    def test_cosine_similarity_1(self):
        all_output = cosine_matrix(self.centered_matrix)
        output = round(all_output[0,2],3)
        expected_output = 0.912
        self.assertAlmostEquals(output, expected_output)

    def test_cosine_similarity_2(self):
        all_output = cosine_matrix(self.centered_matrix)
        output = round(all_output[0,1],3)
        expected_output = 0.735
        self.assertAlmostEquals(output, expected_output)

    def test_cosine_similarity_3(self):
        all_output = cosine_matrix(self.centered_matrix)
        output = round(all_output[5,3],3)
        expected_output = 0.829
        self.assertAlmostEquals(output, expected_output)

    def test_cosine_similarity_4(self):
        all_output = cosine_matrix(self.centered_matrix)
        output = round(all_output[5,4],3)
        expected_output = 0.730
        self.assertAlmostEquals(output, expected_output)

    def test_predict_1(self):
        output = predict(self.matrix,self.cosine, self.neighbour, 3, 6)
        expected_output = 1
        self.assertAlmostEquals(output, expected_output)

    def test_predict_2(self):
        output = predict(self.matrix,self.cosine, self.neighbour, 3, 1)
        expected_output = 3
        self.assertAlmostEquals(output, expected_output)


if __name__ == '__main__':
    unittest.main()

