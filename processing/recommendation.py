import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from pathlib import Path
import os
import json

class RecommendationEngine:
    def __init__(self):
        self.ratings_file = Path('Files/user_ratings.json')
        self.preferences_file = Path('Files/user_preferences.json')
        self.new_df = None
        self.movies = None
        self.movies2 = None
        self._load_dataframes()
        
    def _load_dataframes(self):
        # Load the dataframes from pickle files
        pickle_file_path = r'Files/new_df_dict.pkl'
        
        if os.path.exists(pickle_file_path):
            # Load new_df
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)
            self.new_df = pd.DataFrame.from_dict(loaded_dict)
            
            # Load movies
            pickle_file_path = r'Files/movies_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)
            self.movies = pd.DataFrame.from_dict(loaded_dict)
            
            # Load movies2
            pickle_file_path = r'Files/movies2_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)
            self.movies2 = pd.DataFrame.from_dict(loaded_dict)
    
    def _load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_content_based_recommendations(self, username, n=10):
        """Get content-based recommendations based on user preferences"""
        # Load user preferences
        user_prefs = self._load_json(self.preferences_file).get(username, {})
        
        if not user_prefs:
            # Return popular movies if no preferences
            return self._get_popular_movies(n)
        
        # Filter by favorite genres
        favorite_genres = user_prefs.get('favorite_genres', [])
        filtered_movies = self.new_df
        
        if favorite_genres:
            # Create a score for each movie based on genre match
            genre_scores = []
            
            for _, movie in filtered_movies.iterrows():
                score = 0
                movie_genres = movie['genres'].lower().split()
                for genre in favorite_genres:
                    if genre.lower() in movie_genres:
                        score += 1
                genre_scores.append(score)
            
            # Add scores to dataframe
            temp_df = filtered_movies.copy()
            temp_df['genre_score'] = genre_scores
            
            # Filter by minimum rating if specified
            min_rating = user_prefs.get('min_rating', 0)
            if min_rating > 0:
                # Join with movies2 to get vote_average
                temp_df = temp_df.merge(self.movies2[['movie_id', 'vote_average']], on='movie_id')
                temp_df = temp_df[temp_df['vote_average'] >= min_rating]
            
            # Sort by genre score (descending)
            temp_df = temp_df.sort_values(by='genre_score', ascending=False)
            
            # Get top N movie IDs
            top_movies = temp_df.head(n)
            
            return [(movie['title'], movie['movie_id']) for _, movie in top_movies.iterrows()]
        else:
            # If no genre preferences, return popular movies
            return self._get_popular_movies(n)
    
    def get_collaborative_recommendations(self, username, n=10):
        """Get collaborative filtering recommendations based on user ratings"""
        # Load all user ratings
        all_ratings = self._load_json(self.ratings_file)
        
        # If user has no ratings or there are too few users, fall back to content-based
        if username not in all_ratings or len(all_ratings) < 3:
            return self.get_content_based_recommendations(username, n)
        
        # Convert ratings to dataframe
        ratings_data = []
        for user, movie_ratings in all_ratings.items():
            for movie_id, rating in movie_ratings.items():
                ratings_data.append({
                    'user': user,
                    'movie_id': int(movie_id),
                    'rating': rating
                })
        
        if not ratings_data:
            return self._get_popular_movies(n)
            
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot(
            index='user', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # Get user index
        if username not in user_item_matrix.index:
            return self.get_content_based_recommendations(username, n)
            
        user_idx = user_item_matrix.index.get_loc(username)
        
        # Perform SVD
        U, sigma, Vt = svds(user_item_matrix.values, k=min(user_item_matrix.shape[0]-1, 10))
        
        # Convert sigma to diagonal matrix
        sigma = np.diag(sigma)
        
        # Predict ratings for all movies
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Convert to dataframe
        preds_df = pd.DataFrame(
            all_user_predicted_ratings, 
            columns=user_item_matrix.columns,
            index=user_item_matrix.index
        )
        
        # Get user's rated movie IDs
        user_rated_movies = ratings_df[ratings_df['user'] == username]['movie_id'].tolist()
        
        # Get user's predictions and sort
        user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
        
        # Filter out already rated movies
        user_predictions = user_predictions[~user_predictions.index.isin(user_rated_movies)]
        
        # Get top N movie IDs
        top_movie_ids = user_predictions.head(n).index.tolist()
        
        # Get movie titles
        top_movies = []
        for movie_id in top_movie_ids:
            movie = self.new_df[self.new_df['movie_id'] == movie_id]
            if not movie.empty:
                top_movies.append((movie.iloc[0]['title'], movie_id))
        
        # If we don't have enough recommendations, fill with content-based
        if len(top_movies) < n:
            content_recs = self.get_content_based_recommendations(username, n - len(top_movies))
            # Avoid duplicates
            existing_ids = [movie_id for _, movie_id in top_movies]
            for title, movie_id in content_recs:
                if movie_id not in existing_ids and len(top_movies) < n:
                    top_movies.append((title, movie_id))
        
        return top_movies
    
    def _get_popular_movies(self, n=10):
        """Get popular movies based on vote average and vote count"""
        # Join with movies2 to get vote information
        popular_df = self.new_df.merge(self.movies2[['movie_id', 'vote_average', 'vote_count']], on='movie_id')
        
        # Calculate popularity score (weighted average)
        popular_df['popularity_score'] = popular_df['vote_average'] * popular_df['vote_count']
        
        # Sort by popularity score
        popular_df = popular_df.sort_values(by='popularity_score', ascending=False)
        
        # Get top N movies
        top_movies = popular_df.head(n)
        
        return [(movie['title'], movie['movie_id']) for _, movie in top_movies.iterrows()]
    
    def get_recommendations(self, username, model_type='hybrid', n=10):
        """Get recommendations based on specified model type"""
        if model_type == 'content':
            return self.get_content_based_recommendations(username, n)
        elif model_type == 'collaborative':
            return self.get_collaborative_recommendations(username, n)
        else:  # hybrid approach
            # Get recommendations from both models
            content_recs = self.get_content_based_recommendations(username, n)
            collab_recs = self.get_collaborative_recommendations(username, n)
            
            # Combine recommendations
            hybrid_recs = []
            existing_ids = set()
            
            # Strictly alternate between models to ensure diversity
            i = 0
            j = 0
            while len(hybrid_recs) < n and (i < len(collab_recs) or j < len(content_recs)):
                # Add collaborative recommendation if available
                if i < len(collab_recs):
                    if collab_recs[i][1] not in existing_ids:
                        hybrid_recs.append(collab_recs[i])
                        existing_ids.add(collab_recs[i][1])
                    i += 1
                
                # Break if we have enough recommendations
                if len(hybrid_recs) >= n:
                    break
                    
                # Add content recommendation if available
                if j < len(content_recs):
                    if content_recs[j][1] not in existing_ids:
                        hybrid_recs.append(content_recs[j])
                        existing_ids.add(content_recs[j][1])
                    j += 1
            
            # If we still don't have enough, add popular movies
            if len(hybrid_recs) < n:
                popular_recs = self._get_popular_movies(n - len(hybrid_recs))
                for rec in popular_recs:
                    if rec[1] not in existing_ids and len(hybrid_recs) < n:
                        hybrid_recs.append(rec)
                        existing_ids.add(rec[1])
            
            return hybrid_recs[:n]