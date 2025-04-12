# import pandas as pd
# from sklearn.model_selection import train_test_split

# def split_data(ratings: pd.DataFrame, train_size: float = 0.75, random_state: int = 42):
#     train, test = train_test_split(ratings, train_size=train_size, random_state=random_state)
#     return train, test

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_by_user(ratings: pd.DataFrame, train_ratio: float = 0.75, random_state: int = 42):

    train_list = []
    test_list = []
    
    # Group ratings by user
    grouped = ratings.groupby('User-ID')
    for user, group in grouped:
        if len(group) == 1:
            # If the user has only one rating, add it only to training set
            train_list.append(group)
        else:
            # For users with more than one rating, split them
            train_group, test_group = train_test_split(
                group, train_size=train_ratio, random_state=random_state
            )
            train_list.append(train_group)
            test_list.append(test_group)
    
    # Combine the groups back into full DataFrames
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    return train_df, test_df