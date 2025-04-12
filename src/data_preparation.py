import pandas as pd

def load_data(users_path: str, books_path: str, ratings_path: str):

    users = pd.read_csv(users_path)
    books_dtype = {
        "ISBN": str,
        "Year-Of-Publication": str
    }
    books = pd.read_csv(books_path, low_memory=False, dtype=books_dtype)
    ratings = pd.read_csv(ratings_path)
    
    return users, books, ratings

def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:

    # Filter out rows where 'Book-Rating' is 0
    ratings_clean = ratings[ratings['Book-Rating'] != 0]
    return ratings_clean

def merge_data(ratings: pd.DataFrame, users: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:

    # Merge ratings with users (on "User-ID")
    ratings_merged = ratings.merge(users, on="User-ID", how="left")
    
    # Merge with books (on "ISBN")
    ratings_merged = ratings_merged.merge(books, on="ISBN", how="left")
    
    return ratings_merged

def filter_active_users_and_books(ratings: pd.DataFrame, min_user_ratings=10, min_book_ratings=10):
    """Filter users and books with too few ratings."""
    user_counts = ratings['User-ID'].value_counts()
    book_counts = ratings['ISBN'].value_counts()
    
    ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= min_user_ratings].index)]
    ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= min_book_ratings].index)]
    return ratings



