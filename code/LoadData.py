import numpy as np
import random

def load_rating_data(file_path='ml-100k/u.data'):
    """
    Loads user-movie rating data from a file.

    Args:
        file_path (str): Path to the file containing rating data. Default is 'ml-100k/u.data'.

    Returns:
        np.ndarray: A NumPy array containing the rating data with each row as [user_id, movie_id, rating].
    """
    prefer = [
        [int(userid), int(movieid), float(rating)]
        for line in open(file_path, 'r')
        for userid, movieid, rating, ts in [line.split('\t')]
    ]
    return np.array(prefer)

def split_rating_data(data, test_size=0.2):
    """
    Splits rating data into training and testing sets.

    Args:
        data (np.ndarray): The dataset to split, where each row is [user_id, movie_id, rating].
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
        tuple: A tuple containing two NumPy arrays: (train_data, test_data).
    """
    test_mask = np.random.rand(len(data)) < test_size
    train_data = data[~test_mask]
    test_data = data[test_mask]
    return train_data, test_data
