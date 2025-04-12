
from src.data_preparation import load_data, clean_ratings, filter_active_users_and_books
from src.helper import split_data_by_user
from src.model import create_ratings_matrix, compute_user_similarity, predict_for_test
import numpy as np
import matplotlib.pyplot as plt

# Experiment B: Varying train/test split ratios with a fixed k

users_path = 'D:/vscode/collaborative_filtering/data/Users.csv'
books_path = 'D:/vscode/collaborative_filtering/data/Books.csv'
ratings_path = 'D:/vscode/collaborative_filtering/data/Ratings.csv'

# Load and prepare data
users, books, ratings = load_data(users_path, books_path, ratings_path)
ratings_clean = clean_ratings(ratings)
ratings_filtered = filter_active_users_and_books(ratings_clean, min_user_ratings=10, min_book_ratings=10)

# Define train ratios to test and fixed k
train_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
mad_train_split_results = {}
k_fixed = 5  

for ratio in train_ratios:
    train_set, test_set = split_data_by_user(ratings_filtered, train_ratio=ratio, random_state=42)
    ratings_matrix = create_ratings_matrix(train_set)
    similarity_df = compute_user_similarity(ratings_matrix)
    predictions = predict_for_test(test_set, ratings_matrix, similarity_df, k=k_fixed)
    actual_ratings = test_set['Book-Rating'].values
    mad = np.mean(np.abs(actual_ratings - predictions))
    mad_train_split_results[ratio] = mad
    print(f"MAD (Train Ratio = {ratio}, k={k_fixed}): {mad}")


plt.figure(figsize=(8, 5))
plt.plot(list(mad_train_split_results.keys()), list(mad_train_split_results.values()), marker='o', color='green')
plt.xlabel("Training Set Ratio")
plt.ylabel("Mean Absolute Difference (MAD)")
plt.title(f"MAD vs. Training Set Ratio (k={k_fixed})")
plt.grid(True)
plt.show()
