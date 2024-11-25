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


# --- Data Loading Function --- #

def load_data():
    """
    Loads the dataset and returns it as a DataFrame.
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=column_names)
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
    user_predictions = [cf_model.user_mean[user] if user in cf_model.user_mean else 0 for user in user_avg_ratings.index]
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
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    cf_model = CollaborativeFiltering(k=20)
    cf_model.fit(train_data)

    test_users = test_data['user_id'].unique()
    recommendations = {user: cf_model.recommend(user, N=5) for user in test_users}

    predictions = [cf_model.predict_user_based(row['user_id'], row['item_id']) for _, row in test_data.iterrows()]

    plot_recommendation_overlap(recommendations, test_users)
    plot_item_coverage(recommendations, train_data['item_id'].unique())
    plot_user_bias(train_data, cf_model)
    plot_prediction_vs_ground_truth(test_data, predictions)
    plot_popularity_vs_personalization(recommendations, train_data)


if __name__ == "__main__":
    main()