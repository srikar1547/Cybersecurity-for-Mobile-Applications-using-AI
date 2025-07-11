import unittest
import os
import pickle
from app import CybersecurityModel

class CybersecurityModelUnitTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "C:\\Charan\\MoneyProjects\\CSAI\\Cyber Security using AI\\enhanced_mobile_app_data_with_source.csv"
        self.model = CybersecurityModel()

    def test_model_training(self):
        accuracy = self.model.train_model(self.dataset_path)
        self.assertGreater(accuracy, 0.5, "Model training accuracy should be greater than 0.5")
        self.assertIsNotNone(self.model.pca, "PCA object should not be None after training")
        self.assertIsNotNone(self.model.model, "Model object should not be None after training")
        self.assertGreater(len(self.model.encoders), 0, "Encoders should not be empty after training")
        self.assertTrue(os.path.exists("pca.pkl"), "PCA file should be saved after training")
        self.assertTrue(os.path.exists("random_forest.pkl"), "Random Forest model file should be saved after training")
        self.assertTrue(os.path.exists("encoders.pkl"), "Encoders file should be saved after training")

if __name__ == "__main__":
    unittest.main()