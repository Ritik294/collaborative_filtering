# src/model.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_ratings_matrix(train_data: pd.DataFrame) -> pd.DataFrame:
    ratings_matrix = train_data.pivot(index='User-ID', columns='ISBN', values='Book-Rating')
    return ratings_matrix

def compute_user_similarity(ratings_matrix: pd.DataFrame) -> pd.DataFrame:

    filled_matrix = ratings_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)
    
    # Create DataFrame for easier manipulation and indexing
    similarity_df = pd.DataFrame(similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
    return similarity_df

def predict_rating(user_id: int, book_isbn: str, ratings_matrix: pd.DataFrame, 
                   similarity_df: pd.DataFrame, k: int = 5) -> float:

    # If the book is not in the columns, return global average rating
    if book_isbn not in ratings_matrix.columns:
        return ratings_matrix.stack().mean()
    
    # Get similarities for the target user
    user_similarities = similarity_df[user_id]
    
    # Identify users who have rated the book
    rated_by_users = ratings_matrix[book_isbn].dropna().index
    rated_by_users = rated_by_users[rated_by_users != user_id]  # Exclude the target user itself
    
    if len(rated_by_users) == 0:
        # No one has rated this book, return the global mean rating
        return ratings_matrix.stack().mean()
    
    # Get similarity scores for these users
    sim_scores = user_similarities.loc[rated_by_users]
    
    # Sort the users by similarity score (descending) and select the top k
    top_users = sim_scores.sort_values(ascending=False).head(k).index
    top_sim_scores = sim_scores.loc[top_users]
    top_ratings = ratings_matrix.loc[top_users, book_isbn]
    
    # Weighted average of the ratings from similar users
    if top_sim_scores.sum() > 0:
        prediction = np.dot(top_ratings, top_sim_scores) / top_sim_scores.sum()
    else:
        prediction = ratings_matrix.stack().mean()  # fallback to global mean
        
    return prediction

def predict_for_test(test_data: pd.DataFrame, ratings_matrix: pd.DataFrame, 
                     similarity_df: pd.DataFrame, k: int = 5) -> np.ndarray:

    predictions = []
    for idx, row in test_data.iterrows():
        user = row['User-ID']
        book = row['ISBN']
        pred = predict_rating(user, book, ratings_matrix, similarity_df, k)
        predictions.append(pred)
    return np.array(predictions)


# if __name__ == '__main__':
#     from data_preparation import load_data, clean_ratings
#     from helper import split_data
    
#     users_path = 'D:/vscode/collaborative_filtering/data/Users.csv'  
#     books_path = 'D:/vscode/collaborative_filtering/data/Books.csv'
#     ratings_path = 'D:/vscode/collaborative_filtering/data/Ratings.csv'
    
#     # Load and prepare data
#     _, _, ratings = load_data(users_path, books_path, ratings_path)
#     ratings_clean = clean_ratings(ratings)
#     train_set, test_set = split_data(ratings_clean, train_size=0.75)
    
#     # Build ratings matrix from training data and compute similarity
#     ratings_matrix = create_ratings_matrix(train_set)
#     similarity_df = compute_user_similarity(ratings_matrix)
    
#     # Predict rating for a sample (first row of test set)
#     sample = test_set.iloc[0]
#     pred_rating = predict_rating(sample['User-ID'], sample['ISBN'], ratings_matrix, similarity_df, k=5)
#     print("Sample Test Row:")
#     print(sample)
#     print("Predicted Rating for the sample:", pred_rating)
