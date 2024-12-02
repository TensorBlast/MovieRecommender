import pandas as pd
import numpy as np
import pickle
import requests

try:
    # Load the similarity matrix for the top 100 movies
    with open('S100.pkl', 'wb') as f:
        resp = requests.get('https://codeberg.org/apasi/cs598-psl-proj4artifacts/raw/branch/main/S100.pkl')
        f.write(resp.content)
except Exception as e:
    print(f"Error loading Movie Similarity Matrix: {e}")

with open('S100.pkl', 'rb') as f:
    S100 = pickle.load(f)

try:
    # Load the top 100 movies
    top100 = pd.read_csv('https://codeberg.org/apasi/cs598-psl-proj4artifacts/raw/branch/main/top100.csv')
except Exception as e:
    print(f"Error loading Top 100 Movies: {e}") 

genres = set(g for sublist in top100['genres'].str.split('|') for g in sublist)

def get_displayed_movies():
    return top100

def get_popular_movies(genre):
    if genre is None:
        # return top 100 movies sorted by avg_rating
        return top100.sort_values(by='avg_rating', ascending=False)
    else:
        # return top 100 movies of the given genre sorted by avg_rating
        return top100[top100['genres'].str.contains(genre)].sort_values(by='avg_rating', ascending=False)   
    
def get_recommended_movies(user_ratings):
    recommendations, _ = myIBCF(user_ratings, S100)
    recommended_movies = top100[top100['movie_id'].isin(recommendations)]
    return recommended_movies



def myIBCF(newuser, S, k=10):
    """
    Compute IBCF recommendations for a new user.
    
    Args:
        newuser: dict with movie_ids as keys and ratings (1-5) as values, or DataFrame/Series
        S: similarity matrix (pandas DataFrame with movie_ids as index/columns)
        k: number of recommendations to return (default 10)
    """
    # Handle dict input
    if isinstance(newuser, dict):
        newuser = pd.Series(newuser)
    # Handle DataFrame input
    elif isinstance(newuser, pd.DataFrame):
        newuser = newuser.iloc[0]
    
    # Align ratings with similarity matrix
    aligned_ratings = pd.Series(np.nan, index=S.columns)
    aligned_ratings[newuser.index] = newuser
    newuser = aligned_ratings.to_numpy()
        
    # Initialize recommendations list
    recommendations = []
    
    # Get indices of rated movies
    rated_indices = ~np.isnan(newuser)
    
    # Initialize predictions array
    predictions = np.zeros(len(newuser))
    movie_ids = S.columns.tolist()
    
    # For each movie i
    for i, movie_id in enumerate(movie_ids):
        # Skip if already rated
        if rated_indices[i]:
            predictions[i] = np.nan
            continue
            
        # Get valid similarity scores for movie i
        valid_sims = ~S[movie_id].isna()
        
        # Find overlap between valid similarities and rated movies
        overlap = valid_sims & rated_indices
        
        if not np.any(overlap):
            predictions[i] = np.nan
            continue
            
        # Get relevant similarities and ratings
        sims = S.loc[movie_id, overlap].values
        ratings = newuser[overlap]
        
        # Calculate weighted average
        predictions[i] = np.sum(sims * ratings) / (1e-8 + np.sum(sims))
    
    # Get number of valid predictions
    valid_mask = ~np.isnan(predictions)
    n_valid = np.sum(valid_mask)
    
    if n_valid >= k:
        # Get indices of top k predictions
        print("We have enough valid predictions")

        valid_predictions = predictions[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        # Sort the valid predictions in descending order
        sorted_indices = np.argsort(valid_predictions)[::-1][:k]
        top_preds = valid_predictions[sorted_indices]
        top_indices = valid_indices[sorted_indices]
        recommendations = [movie_ids[i] for i in top_indices]
        return recommendations, top_preds
    else:
        # Get valid predictions first
        print("We don't have enough valid predictions so adding popular movies")

        valid_predictions = predictions[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Sort the valid predictions in descending order
        sorted_indices = np.argsort(valid_predictions)[::-1][:n_valid]
        top_preds = valid_predictions[sorted_indices]
        top_indices = valid_indices[sorted_indices]
        recommendations = [movie_ids[i] for i in top_indices]
        
        # # Get popularity scores for remaining slots
        # ratings = pd.read_csv('https://codeberg.org/apasi/cs598-psl-proj4artifacts/raw/branch/main/Rpivot.csv')
        # popular_movies = ratings.groupby('movie_id').agg(
        #     avg_rating=('rating', 'mean'),
        #     num_ratings=('rating', 'count')
        # ).reset_index()
        
        # # Filter and sort popular movies
        # popular_movies = popular_movies[popular_movies['num_ratings'] >= 1000]
        # popular_movies = popular_movies.sort_values('avg_rating', ascending=False)

        # We already loaded the popular movies in the beginning
        popular_movies = top100.copy().sort_values(by='avg_rating', ascending=False)
        
        # Add popular movies until we have k recommendations
        for _, row in popular_movies.iterrows():
            movie_id = row['movie_id']
            if len(recommendations) >= k:
                break
            # Check if movie is not rated using S columns for indexing
            movie_idx = S.columns.get_loc(movie_id)
            if movie_id not in recommendations and np.isnan(newuser[movie_idx]):
                recommendations.append(movie_id)
                
    return recommendations, top_preds