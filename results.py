import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple


class Results:
    """Class to compute classification results."""

    def __init__(self, labels: Dict[str, int], dataset_name: str = ""):
        """Results initializer.

        Args:
            labels: Dictionary relating textual and numeric labels.
            dataset_name: Name of the dataset. Used to identify the results when printed and saved.

        """
        self._labels = labels
        self._dataset_name = dataset_name

    def compute(self, test_dir: str,dataset: List[str], true_labels: List[int], predicted_labels: List[int]) -> \
            Tuple[float, np.ndarray, List[Tuple[str, str, str]]]:
        """Builds a confusion matrix and computes the classification accuracy.

        Args:
            dataset: Paths to the test images.
            true_labels: Real categories.
            predicted_labels: Predicted categories.

        Returns:
            Classification accuracy.
            Confusion matrix.
            Detailed per image classification results.

        """
        category_count = len(self._labels)
        confusion_matrix = np.zeros((category_count, category_count))
        classification = []

        # Build an inverse lookup dictionary to retrieve label descriptions from indices
        descriptions = {v: k for k, v in self._labels.items()}

        # Format classification results and compute the confusion matrix
        for image, true, predicted in zip(dataset, true_labels, predicted_labels):
            absolute_path = os.path.join(test_dir,image)
            folder_path = os.path.dirname(absolute_path).replace("\\","/") + "/"
            classification.append((os.path.basename(image), descriptions[predicted], folder_path))
            confusion_matrix[true, predicted] += 1

        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        return accuracy, confusion_matrix, classification

    def print(self, accuracy: float, confusion_matrix: np.ndarray):
        """Prints a formatted confusion matrix in the console and the classification accuracy achieved.

        Args:
            confusion_matrix: Confusion matrix.
            accuracy: Classification accuracy.

        """
        # Increase output console line width for Pandas dataframes and NumPy arrays.
        line_width = 400
        pd.set_option('display.max_columns', 15)
        pd.set_option('display.width', line_width)
        np.set_printoptions(linewidth=line_width)

        # Prepare confusion matrix
        labels = [key for key, value in sorted(self._labels.items(), key=lambda x: x[1])]
        confusion_df = pd.DataFrame(confusion_matrix, columns=labels, index=labels)
        confusion_df.columns.name = 'KNOWN/PREDICTED'

        if self._dataset_name:
            print("\n\nCLASSIFICATION RESULTS (", self._dataset_name.upper(), ")", sep='')
        else:
            print("\n\nCLASSIFICATION RESULTS")

        print("\nConfusion matrix\n")
        print(confusion_df)
        print("\nAccuracy: ", accuracy)

    def save(self, confusion_matrix: np.ndarray, classification: List[Tuple[str, str, str]], predictions: List[List[float]], results_folder: str):
        """Save results to an Excel file.

        Every argument is stored in its own sheet.

        Args:
            confusion_matrix: Confusion matrix.
            classification: Detailed per image classification results.
            predictions: Probabilities of every category for each image.

        """
        # Format confusion matrix
        labels = [key for key, value in sorted(self._labels.items(), key=lambda x: x[1])]
        confusion_df = pd.DataFrame(confusion_matrix, columns=labels, index=labels)

        # Format classification results
        classification_df = pd.DataFrame(classification, columns=('Image', 'Predicted', 'Folder_Path'))

        probabilities_df = pd.DataFrame(predictions, columns=["probabilities"])
        probabilities_df["probabilities"] = 1 - probabilities_df["probabilities"]
        classification_df = pd.concat([classification_df, probabilities_df], axis=1)

        # Write to Excel
        workbook = results_folder + self._dataset_name.lower().replace(" ", "_") + '_' +str(datetime.now().today().date()).replace("-","_") + "_"+ datetime.now().time().strftime("%H%M%S")
        workbook += "_results.xlsx"

        with pd.ExcelWriter(workbook) as writer:
            confusion_df.to_excel(writer, sheet_name='Confusion matrix', index_label='KNOWN/PREDICTED')
            classification_df.to_excel(writer, sheet_name='Classification results', index=False, float_format = '%.2f', freeze_panes=(1, 0))
            
