

from src.data_preparation import load_data, clean_ratings, filter_active_users_and_books
from src.helper import split_data_by_user
from src.model import create_ratings_matrix, compute_user_similarity, predict_for_test
import numpy as np
import matplotlib.pyplot as plt


# File paths
users_path = 'D:/vscode/collaborative_filtering/data/Users.csv'  
books_path = 'D:/vscode/collaborative_filtering/data/Books.csv'
ratings_path = 'D:/vscode/collaborative_filtering/data/Ratings.csv'

# Load and prepare data
users, books, ratings = load_data(users_path, books_path, ratings_path)
ratings_clean = clean_ratings(ratings)
ratings_filtered = filter_active_users_and_books(ratings_clean, min_user_ratings=10, min_book_ratings=10)

# Split data
#train_set, test_set = split_data(ratings_filtered, train_size=0.75)
train_set, test_set = split_data_by_user(ratings_filtered, train_ratio=0.75, random_state=42)
# Build model
ratings_matrix = create_ratings_matrix(train_set)
similarity_df = compute_user_similarity(ratings_matrix)


# predictions = predict_for_test(test_set, ratings_matrix, similarity_df, k=5)

# # Evaluate the predictions using Mean Absolute Difference (MAD)
# actual_ratings = test_set['Book-Rating'].values
# mad = np.mean(np.abs(actual_ratings - predictions))
# print("MAD (k=5):", mad)
# List of k values to try
k_values = [5, 10, 15, 20, 50, 100]
mad_results = {}

for k in k_values:
    predictions = predict_for_test(test_set, ratings_matrix, similarity_df, k=k)
    actual_ratings = test_set['Book-Rating'].values
    mad = np.mean(np.abs(actual_ratings - predictions))
    mad_results[k] = mad
    print(f"MAD (k={k}): {mad}")



plt.figure(figsize=(8, 5))
plt.plot(list(mad_results.keys()), list(mad_results.values()), marker='o')
plt.xlabel("Neighborhood Size (k)")
plt.ylabel("Mean Absolute Difference (MAD)")
plt.title("MAD vs. Neighborhood Size")
plt.grid(True)
plt.show()