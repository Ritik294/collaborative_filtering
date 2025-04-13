# Collaborative Filtering for Book Rating Prediction

## Overview

This project implements a collaborative filtering algorithm to predict book ratings. Using user-user collaborative filtering, the model leverages historical ratings to predict ratings for books that a user has not yet rated. The goal is to minimize the Mean Absolute Difference (MAD) between predicted and actual ratings, thereby enhancing the recommendations.

## Dataset Description

The project uses three main CSV files:

- **Users.csv:**  
  Contains anonymized user data along with demographic details.
  - *User-ID:* Unique identifier for each user.
  - *Location:* The location of the user.
  - *Age:* The age of the user (may contain NULL values).

- **Books.csv:**  
  Contains detailed information about books.
  - *ISBN:* Serves as the unique identifier (loaded as a string to preserve leading zeros).
  - *Book Title:* The title of the book.
  - *Book Author:* The author of the book.
  - *Year Of Publication:* The year the book was published (loaded as a string to manage mixed data types).
  - *Publisher:* The book's publisher.
  - *Image URLs:* Includes URLs for the small, medium, and large cover images.

- **Ratings.csv:**  
  Contains ratings submitted by users for various books.
  - *User-ID:* The identifier of the user who provided the rating.
  - *ISBN:* The unique identifier for the book.
  - *Book Rating:* The rating given by the user (ranges from 1 to 10; 0 indicates that the book was not rated).
   
## Methodology

### Data Preparation and Cleaning

1. **Loading the Data:**  
   The data is read from the CSV files using pandas. For Books.csv, explicit data types are set for key columns (ISBN and Year-Of-Publication) to avoid mixed type issues. This ensures that identifiers like ISBN (which may have leading zeros) and publication years are treated consistently as strings.

2. **Cleaning the Ratings:**  
   Rows where the rating is 0—indicating that the book was not rated—are removed. In addition, we filter the dataset to include only users and books with a minimum number of ratings (at least 10 each). This step helps maintain data quality and reduces the memory footprint when building the user-item matrix.

3. **Splitting the Dataset:**  
   The data is split on a per-user basis. For users with multiple ratings, their ratings are divided between training and test sets (commonly 75% training and 25% testing), ensuring that every user in the test set is also represented in the training set. For users with only one rating, that single rating is automatically assigned to the training set.

### Collaborative Filtering Model

1. **Building the Ratings Matrix:**  
   A pivot table is created where rows represent users and columns represent books (identified by ISBN), with cell values corresponding to the ratings. This matrix forms the basis for comparing user preferences.

2. **Computing User Similarity:**  
   Cosine similarity is computed between users based on their rating vectors. The resulting similarity matrix is used to identify the most similar users for a given target user.

3. **Predicting Ratings:**  
   For each missing rating in the test set, the model predicts a value using a weighted average of ratings from the top k similar users. Several values of k (5, 10, 15, 20, 50, 100) are tested in order to assess the impact on performance.

4. **Evaluation:**  
   Model performance is measured using the Mean Absolute Difference (MAD) between the predicted ratings and the actual ratings. Additional experiments vary the training set ratio (from 60% to 90% training data) to further analyze how the amount of training data affects prediction accuracy.

## Experiments

### Experiment A: Varying Neighborhood Size (k)

**Setup:**  
With a fixed train/test split (75% training), the model was evaluated using different neighborhood sizes:  
- k = 5, 10, 15, 20, 50, and 100.

**Observations:**  
- The MAD decreased noticeably from k = 5 to k = 10.
- Beyond k ≈ 20, the MAD plateaued, suggesting that after a moderate number of similar users are considered, additional users provide little extra predictive power.

**Conclusion:**  
This experiment indicates that a neighborhood size in the range of **15–20** is sufficient to achieve near-optimal prediction performance.


### Experiment B: Varying the Train/Test Split Ratio

**Setup:**  
Using a fixed neighborhood size (k = 5), the training set was varied from 60% to 90% of the data (in increments of 5%).

**Observations:**  
- The MAD steadily decreased as the training ratio increased, indicating better predictions with more training data.
- However, improvements became less pronounced at higher training ratios, suggesting diminishing returns after a certain point.

**Conclusion:**  
The model benefits from more training data, but after reaching roughly **80–90%** training data, the gains in accuracy become minimal.

## How to Run

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ritik294/collaborative_filtering.git
   cd collaborative_filtering

2. **Create and Activate the Virtual Environment:**
   ```bash
   python -m venv env
   # On Windows (using PowerShell):
   .\env\Scripts\Activate.ps1

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

### Running the Experiments

1. **Experiment A (Varying k):**
   ```bash
   python main.py

2. **Experiment B (Varying Train/Test Split Ratio):**
   ```bash
   python main_2.py

### Dependencies

- pandas

- numpy

- scikit-learn

- matplotlib
