import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- Collaborative Filtering Class --- #

class CollaborativeFiltering:
    """
    Collaborative Filtering using User-Based and Item-Based Similarity
    """
    def __init__(self, k=20):
        self.k = k
        self.user_sim_matrix = None
        self.item_sim_matrix = None
        self.user_mean = None
        self.item_mean = None
        self.train_matrix = None

    def fit(self, train_data):
        """
        Prepares the user-item rating matrix and calculates similarity matrices.
        """
        self.train_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

        self.user_sim_matrix = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim_matrix, 0)
        self.item_sim_matrix = cosine_similarity(self.train_matrix.T)
        np.fill_diagonal(self.item_sim_matrix, 0)

        self.user_mean = self.train_matrix.replace(0, np.nan).mean(axis=1)
        self.item_mean = self.train_matrix.replace(0, np.nan).mean(axis=0)

    def predict_user_based(self, user_id, item_id):
        """
        Predicts the rating for a specific user and item using User-Based Collaborative Filtering.
        """
        if item_id not in self.train_matrix.columns:
            return self.user_mean.get(user_id, 0)
        if user_id not in self.train_matrix.index:
            return self.item_mean.get(item_id, 0)

        user_idx = self.train_matrix.index.get_loc(user_id)
        item_idx = self.train_matrix.columns.get_loc(item_id)

        sim_scores = self.user_sim_matrix[user_idx]
        user_ratings = self.train_matrix.iloc[:, item_idx]

        mask = user_ratings > 0
        sim_scores = sim_scores[mask]
        user_ratings = user_ratings[mask]

        if len(sim_scores) == 0:
            return self.user_mean[user_id]

        top_k_idx = np.argsort(sim_scores)[-self.k:]
        sim_scores_top_k = sim_scores[top_k_idx]
        user_ratings_top_k = user_ratings.iloc[top_k_idx]
        pred = np.dot(sim_scores_top_k, user_ratings_top_k) / np.sum(np.abs(sim_scores_top_k))
        return pred

    def recommend(self, user_id, N=10):
        """
        Recommends N items for a given user based on User-Based Collaborative Filtering.
        """
        if user_id not in self.train_matrix.index:
            return []

        user_idx = self.train_matrix.index.get_loc(user_id)
        user_ratings = self.train_matrix.iloc[user_idx, :]
        rated_items = user_ratings[user_ratings > 0].index.tolist()

        predictions = {}
        for item in self.train_matrix.columns:
            if item not in rated_items:
                pred_rating = self.predict_user_based(user_id, item)
                predictions[item] = pred_rating

        # Sort items by predicted ratings
        recommended_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:N]
        return [item for item, score in recommended_items]

# --- Online Collaborative Filtering Class --- #

class OnlineCollaborativeFiltering:
    """
    Online Collaborative Filtering for dynamic recommendation updates.
    Supports user-based collaborative filtering with dynamic updates to similarity scores.
    """
    def __init__(self, k=10):
        self.k = k
        self.user_item_matrix = None
        self.user_sim_matrix = None

    def initialize(self, initial_data):
        """
        Initializes the user-item interaction matrix and computes the initial similarity matrix.
        """
        self._build_user_item_matrix(initial_data)
        self._update_similarity_matrix()

    def _build_user_item_matrix(self, data):
        users = [d[0] for d in data]
        items = [d[1] for d in data]
        self.user_item_matrix = np.zeros((max(users) + 1, max(items) + 1))
        for user, item, rating in data:
            self.user_item_matrix[user, item] = rating

    def _update_similarity_matrix(self):
        """
        Computes the cosine similarity matrix for users.
        """
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(self.user_sim_matrix, 0)

    def update(self, user_id, item_id, rating):
        """
        Updates the user-item matrix and recalculates similarity dynamically.
        """
        if user_id >= self.user_item_matrix.shape[0] or item_id >= self.user_item_matrix.shape[1]:
            self._expand_user_item_matrix(user_id, item_id)
        self.user_item_matrix[user_id, item_id] = rating
        self._update_similarity_matrix()

    def _expand_user_item_matrix(self, user_id, item_id):
        new_users = max(0, user_id + 1 - self.user_item_matrix.shape[0])
        new_items = max(0, item_id + 1 - self.user_item_matrix.shape[1])
        self.user_item_matrix = np.pad(
            self.user_item_matrix, ((0, new_users), (0, new_items)), mode="constant", constant_values=0
        )

    def predict(self, user_id, item_id):
        if user_id >= self.user_item_matrix.shape[0] or item_id >= self.user_item_matrix.shape[1]:
            return 0

        user_ratings = self.user_item_matrix[:, item_id]
        user_similarities = self.user_sim_matrix[user_id]

        mask = user_ratings > 0
        user_similarities = user_similarities[mask]
        user_ratings = user_ratings[mask]

        if len(user_similarities) == 0:
            return 0

        prediction = np.dot(user_similarities, user_ratings) / np.sum(np.abs(user_similarities))
        return prediction

    def recommend(self, user_id, N=5):
        if user_id >= self.user_item_matrix.shape[0]:
            return []

        user_ratings = self.user_item_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]

        predictions = {}
        for item_id in range(self.user_item_matrix.shape[1]):
            if item_id not in rated_items:
                predictions[item_id] = self.predict(user_id, item_id)

        return sorted(predictions, key=predictions.get, reverse=True)[:N]

# --- Data Loading Function --- #

def load_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=column_names)
    return data

# --- Main Script --- #

def main():
    """
    Main script to load data, train models, and visualize results.
    """
    data = load_data()
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Train Collaborative Filtering Model
    cf_model = CollaborativeFiltering(k=20)
    cf_model.fit(train_data)
    test_users = test_data['user_id'].unique()
    recommendations_cf = {user: cf_model.recommend(user, N=5) for user in test_users}
    predictions_cf = [cf_model.predict_user_based(row['user_id'], row['item_id']) for _, row in test_data.iterrows()]

    # Train Online Collaborative Filtering Model
    initial_data = list(train_data[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
    ocf_model = OnlineCollaborativeFiltering(k=10)
    ocf_model.initialize(initial_data)
    ocf_model.update(user_id=0, item_id=5, rating=4)  # Example update
    recommendations_ocf = {user: ocf_model.recommend(user, N=5) for user in test_users}

    # Print Recommendations for Comparison
    for user in test_users[:5]:  # Limit output for readability
        print(f"User {user} CF Recommendations: {recommendations_cf[user]}")
        print(f"User {user} OCF Recommendations: {recommendations_ocf[user]}")

if __name__ == "__main__":
    main()