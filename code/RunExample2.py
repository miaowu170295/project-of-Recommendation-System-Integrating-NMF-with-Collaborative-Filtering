import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

# --- Hybrid Collaborative Filtering Class --- #
class HybridCollaborativeFiltering:
    """
    Hybrid Collaborative Filtering using a combination of User-Based, Item-Based, and Matrix Factorization
    """
    def __init__(self, k=20, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.user_sim_matrix = None
        self.item_sim_matrix = None
        self.user_mean = None
        self.item_mean = None
        self.train_matrix = None
        self.nmf_model = None

    def fit(self, train_data):
        """
        Prepares the user-item rating matrix and calculates similarity matrices.
        """
        self.train_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        self.train_matrix = csr_matrix(self.train_matrix.values)

        self.user_sim_matrix = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim_matrix, 0)
        self.item_sim_matrix = cosine_similarity(self.train_matrix.T)
        np.fill_diagonal(self.item_sim_matrix, 0)

        self.user_mean = np.array(train_data.groupby('user_id')['rating'].mean())
        self.item_mean = np.array(train_data.groupby('item_id')['rating'].mean())

        # Fit NMF Model for Matrix Factorization
        self.nmf_model = NMF(n_components=10, init='random', random_state=42)
        self.user_factors = self.nmf_model.fit_transform(self.train_matrix)
        self.item_factors = self.nmf_model.components_.T

    def predict_hybrid(self, user_id, item_id):
        """
        Predicts the rating for a specific user and item using a hybrid of User-Based, Item-Based, and Matrix Factorization.
        """
        if item_id >= self.train_matrix.shape[1]:
            return self.user_mean[user_id] if user_id < len(self.user_mean) else 0
        if user_id >= self.train_matrix.shape[0]:
            return self.item_mean[item_id] if item_id < len(self.item_mean) else 0

        user_idx = user_id
        item_idx = item_id

        user_sim_scores = self.user_sim_matrix[user_idx]
        user_ratings = self.train_matrix[:, item_idx].toarray().flatten()

        user_mask = user_ratings > 0
        user_sim_scores = user_sim_scores[user_mask]
        user_ratings = user_ratings[user_mask]

        if len(user_sim_scores) > 0:
            top_k_user_idx = np.argpartition(user_sim_scores, -min(self.k, len(user_sim_scores)))[-min(self.k, len(user_sim_scores)):]
            user_sim_scores_top_k = user_sim_scores[top_k_user_idx]
            user_ratings_top_k = user_ratings[top_k_user_idx]
            user_based_pred = np.dot(user_sim_scores_top_k, user_ratings_top_k) / np.sum(np.abs(user_sim_scores_top_k))
        else:
            user_based_pred = self.user_mean[user_id]

        item_sim_scores = self.item_sim_matrix[item_idx]
        item_ratings = self.train_matrix[user_idx, :].toarray().flatten()

        item_mask = item_ratings > 0
        item_sim_scores = item_sim_scores[item_mask]
        item_ratings = item_ratings[item_mask]

        if len(item_sim_scores) > 0:
            top_k_item_idx = np.argpartition(item_sim_scores, -min(self.k, len(item_sim_scores)))[-min(self.k, len(item_sim_scores)):]
            item_sim_scores_top_k = item_sim_scores[top_k_item_idx]
            item_ratings_top_k = item_ratings[top_k_item_idx]
            item_based_pred = np.dot(item_sim_scores_top_k, item_ratings_top_k) / np.sum(np.abs(item_sim_scores_top_k))
        else:
            item_based_pred = self.item_mean[item_id]

        # Matrix Factorization Prediction
        if user_id < self.user_factors.shape[0] and item_id < self.item_factors.shape[0]:
            mf_pred = np.dot(self.user_factors[user_id], self.item_factors[item_id])
        else:
            mf_pred = 0

        # Hybrid Combination
        hybrid_pred = (self.alpha * user_based_pred + (1 - self.alpha) * item_based_pred + mf_pred) / 2
        return hybrid_pred

    def recommend(self, user_id, N=10):
        """
        Recommends N items for a given user based on Hybrid Collaborative Filtering.
        """
        if user_id >= self.train_matrix.shape[0]:
            return []

        user_idx = user_id
        user_ratings = self.train_matrix[user_idx, :].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]

        predictions = {}
        for item in range(self.train_matrix.shape[1]):
            if item not in rated_items:
                pred_rating = self.predict_hybrid(user_id, item)
                predictions[item] = pred_rating

        recommended_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:N]
        return [item for item, score in recommended_items]

# --- Data Loading Function --- #

def load_data():
    """
    Loads the dataset and returns it as a DataFrame.
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    try:
        data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=column_names)
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return pd.DataFrame()
    return data

# --- Visualization Functions --- #
def plot_recommendation_overlap(recommendations, test_users):
    """
    Plots the overlap matrix of recommendations among test users.
    """
    overlap_matrix = np.zeros((len(test_users), len(test_users)))
    for i, user_a in enumerate(test_users):
        for j, user_b in enumerate(test_users):
            overlap_matrix[i, j] = len(set(recommendations[user_a]) & set(recommendations[user_b]))
    plt.figure(figsize=(10, 8))
    plt.imshow(overlap_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Overlap Count')
    plt.title('Recommendation Overlap Matrix')
    plt.xlabel('User Index')
    plt.ylabel('User Index')
    plt.show()


def plot_item_coverage(recommendations, all_items):
    """
    Plots the coverage of recommended items across the dataset.
    """
    recommended_items = set(item for recs in recommendations.values() for item in recs)
    coverage = len(recommended_items) / len(all_items) * 100
    plt.figure(figsize=(8, 5))
    plt.bar(['Recommended', 'Not Recommended'], [coverage, 100 - coverage], color=['green', 'red'])
    plt.title('Item Coverage')
    plt.ylabel('Percentage')
    plt.grid(axis='y')
    plt.show()


def plot_user_bias(train_data, cf_model):
    """
    Visualizes user bias by comparing average ratings to predicted ratings.
    """
    user_avg_ratings = train_data.groupby('user_id')['rating'].mean()
    user_predictions = [cf_model.user_mean[user] if user < len(cf_model.user_mean) else 0 for user in user_avg_ratings.index]
    plt.figure(figsize=(10, 6))
    plt.scatter(user_avg_ratings, user_predictions, alpha=0.6)
    plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Ideal Predictions (y=x)')
    plt.xlabel('Average User Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('User Bias Analysis: Predictions vs. Average Ratings')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction_vs_ground_truth(test_data, predictions):
    """
    Plots predicted ratings versus ground truth ratings for evaluation.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data['rating'], predictions, alpha=0.5, label='Predictions')
    plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Ideal Predictions (y=x)')
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Prediction vs. Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_popularity_vs_personalization(recommendations, train_data):
    """
    Analyzes personalization by comparing recommendation popularity.
    """
    item_popularity = train_data['item_id'].value_counts()
    personalization_scores = [
        np.mean([item_popularity.get(item, 0) for item in recs])
        for recs in recommendations.values()
    ]
    plt.figure(figsize=(10, 6))
    plt.hist(personalization_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Popularity vs. Personalization in Recommendations')
    plt.xlabel('Average Item Popularity (Recommended)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.show()

# --- Main Script --- #

def main():
    """
    Main script to load data, train the model, and visualize results.
    """
    data = load_data()
    if data.empty:
        return
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Hybrid Collaborative Filtering
    hybrid_model = HybridCollaborativeFiltering(k=20, alpha=0.5)
    hybrid_model.fit(train_data)
    test_users = test_data['user_id'].unique()
    recommendations_hybrid = {user: hybrid_model.recommend(user, N=5) for user in test_users}
    predictions_hybrid = [hybrid_model.predict_hybrid(row['user_id'], row['item_id']) for _, row in test_data.iterrows()]

    # Visualization
    plot_recommendation_overlap(recommendations_hybrid, test_users)
    plot_item_coverage(recommendations_hybrid, train_data['item_id'].unique())
    plot_user_bias(train_data, hybrid_model)
    plot_prediction_vs_ground_truth(test_data, predictions_hybrid)
    plot_popularity_vs_personalization(recommendations_hybrid, train_data)

if __name__ == "__main__":
    main()