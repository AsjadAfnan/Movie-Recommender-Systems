import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from pathlib import Path
import os
import json
import time
from processing.metrics import RecommendationMetrics

class RecommendationEngine:
    def __init__(self):
        self.ratings_file = Path('Files/user_ratings.json')
        self.preferences_file = Path('Files/user_preferences.json')
        self.new_df = None
        self.movies = None
        self.movies2 = None
        self.metrics = RecommendationMetrics()
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
    
    def get_content_based_recommendations(self, username, n=10, model_seed=None):
        """Get content-based recommendations based on user preferences"""
        # Load user preferences
        user_prefs = self._load_json(self.preferences_file).get(username, {})
        
        # Add randomization to ensure fresh results each time
        # Use provided seed if available, otherwise generate a new one
        random_seed = model_seed if model_seed is not None else np.random.randint(1000)
        np.random.seed(random_seed)
        
        if not user_prefs:
            # Return popular movies if no preferences
            return self._get_popular_movies(n, model_seed=random_seed)
        
        # Start with all movies
        filtered_movies = self.new_df.copy()
        
        # Filter by language preference if specified
        preferred_languages = user_prefs.get('preferred_languages', [])
        if preferred_languages:
            # Join with movies2 to get language information
            filtered_movies = filtered_movies.merge(self.movies2[['movie_id', 'spoken_languages']], on='movie_id')
            # Convert spoken_languages from string to list
            filtered_movies['spoken_languages'] = filtered_movies['spoken_languages'].apply(lambda x: [
                lang['name'].lower() if isinstance(lang, dict) and 'name' in lang
                else lang.lower() if isinstance(lang, str)
                else ''
                for lang in ast.literal_eval(x)
            ])
            # Filter movies by preferred languages
            filtered_movies = filtered_movies[filtered_movies['spoken_languages'].apply(
                lambda x: any(lang.lower() in [pref.lower() for pref in preferred_languages] for lang in x)
            )]
        
        # Calculate content-based scores
        scores = pd.Series(0.0, index=filtered_movies.index)
        
        # Genre score (0-3 points)
        favorite_genres = user_prefs.get('favorite_genres', [])
        if favorite_genres:
            for idx, movie in filtered_movies.iterrows():
                movie_genres = movie['genres'].lower().split()
                genre_score = sum(genre.lower() in movie_genres for genre in favorite_genres)
                # Add slight randomization to genre score
                scores[idx] += genre_score + (np.random.random() * 0.5)
        
        # Era preference score (0-1 point)
        era_pref = user_prefs.get('preferred_era', '')
        if era_pref:
            for idx, movie in filtered_movies.iterrows():
                release_year = pd.to_datetime(self.movies2[self.movies2['movie_id'] == movie['movie_id']]['release_date'].iloc[0]).year
                if era_pref == 'Classic (Pre-1970)' and release_year < 1970:
                    scores[idx] += 1 + (np.random.random() * 0.3)
                elif era_pref == '1970s-1990s' and 1970 <= release_year < 1990:
                    scores[idx] += 1 + (np.random.random() * 0.3)
                elif era_pref == '1990s-2010s' and 1990 <= release_year < 2010:
                    scores[idx] += 1 + (np.random.random() * 0.3)
                elif era_pref == 'Modern (2010+)' and release_year >= 2010:
                    scores[idx] += 1 + (np.random.random() * 0.3)
        
        # Director preference score (0-2 points)
        preferred_directors = user_prefs.get('preferred_directors', '').lower().split(',')
        if preferred_directors and preferred_directors[0]:
            for idx, movie in filtered_movies.iterrows():
                movie_director = movie['tcrew'].lower()
                if any(director.strip() in movie_director for director in preferred_directors):
                    scores[idx] += 2 + (np.random.random() * 0.4)
        
        # Mood preference score (0-1 point)
        mood_pref = user_prefs.get('preferred_mood', '')
        if mood_pref:
            # Add mood score based on keywords and genres
            mood_keywords = {
                'Light & Funny': ['comedy', 'family', 'animation'],
                'Thought-provoking': ['drama', 'documentary', 'mystery'],
                'Intense & Thrilling': ['thriller', 'horror', 'crime'],
                'Emotional & Dramatic': ['drama', 'romance', 'family'],
                'Escapist & Fantasy': ['fantasy', 'adventure', 'sci-fi']
            }
            for idx, movie in filtered_movies.iterrows():
                movie_keywords = movie['keywords'].lower()
                if any(keyword in movie_keywords for keyword in mood_keywords.get(mood_pref, [])):
                    scores[idx] += 1 + (np.random.random() * 0.3)
        
        # Add a larger random factor to all scores to ensure variety between model toggles
        scores = scores + (np.random.random(len(scores)) * 1.0)
        
        # Normalize scores
        if not scores.empty:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Add vote average as a factor (0-1 point)
            vote_averages = self.movies2[['movie_id', 'vote_average']].set_index('movie_id')
            for idx, movie in filtered_movies.iterrows():
                try:
                    vote_avg = vote_averages.loc[movie['movie_id'], 'vote_average']
                    # Handle case where vote_avg might be a Series instead of a scalar
                    if hasattr(vote_avg, 'iloc'):
                        vote_avg = vote_avg.iloc[0] if not vote_avg.empty else 0
                    scores[idx] += (vote_avg / 10.0) * (1 + np.random.random() * 0.2)
                except (KeyError, ValueError, TypeError):
                    # If movie_id not found or other error, skip adding vote score
                    continue
            
            # Get top N recommendations
            filtered_movies['final_score'] = scores
            filtered_movies = filtered_movies.sort_values('final_score', ascending=False)
            top_movies = filtered_movies.head(n)
            
            return [(movie['title'], movie['movie_id']) for _, movie in top_movies.iterrows()]
        
        # Fallback to popular movies if no recommendations found
        return self._get_popular_movies(n, model_seed=random_seed)
    
    def get_collaborative_recommendations(self, username, n=10, model_seed=None):
        """Get collaborative filtering recommendations based on user ratings"""
        # Load all user ratings
        all_ratings = self._load_json(self.ratings_file)
        
        # Add randomization to ensure fresh results each time
        # Use provided seed if available, otherwise generate a new one
        random_seed = model_seed if model_seed is not None else np.random.randint(1000)
        np.random.seed(random_seed)
        
        # If user has no ratings or there are too few users, fall back to content-based
        if username not in all_ratings or len(all_ratings) < 3:
            return self.get_content_based_recommendations(username, n, model_seed=random_seed)
        
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
            return self._get_popular_movies(n, model_seed=random_seed)
            
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot(
            index='user', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # Get user index
        if username not in user_item_matrix.index:
            return self.get_content_based_recommendations(username, n, model_seed=random_seed)
            
        user_idx = user_item_matrix.index.get_loc(username)
        
        # Perform SVD with more latent factors for better accuracy
        # Use min(n_users-1, 20) instead of 10 to capture more patterns
        k_factors = min(user_item_matrix.shape[0]-1, 20)
        U, sigma, Vt = svds(user_item_matrix.values, k=k_factors)
        
        # Convert sigma to diagonal matrix
        sigma = np.diag(sigma)
        
        # Predict ratings for all movies
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Add small random noise to predictions to ensure variety
        # Use a larger random factor for collaborative filtering to make it more distinct
        all_user_predicted_ratings += np.random.random(all_user_predicted_ratings.shape) * 1.0
        
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
        
        # Apply language filtering if user has language preferences
        user_prefs = self._load_json(self.preferences_file).get(username, {})
        preferred_languages = user_prefs.get('preferred_languages', [])
        
        # Get top movie IDs with potential language filtering
        if preferred_languages:
            # Get more candidates than needed for filtering
            candidate_movie_ids = user_predictions.head(n*3).index.tolist()
            
            # Filter by language
            filtered_movies = []
            for movie_id in candidate_movie_ids:
                movie_data = self.movies2[self.movies2['movie_id'] == movie_id]
                if not movie_data.empty:
                    languages = ast.literal_eval(movie_data.iloc[0]['spoken_languages'])
                    language_names = []
                    for lang in languages:
                        if isinstance(lang, dict) and 'name' in lang:
                            language_names.append(lang['name'].lower())
                        elif isinstance(lang, str):
                            language_names.append(lang.lower())
                    if any(lang.lower() in language_names for lang in preferred_languages):
                        movie = self.new_df[self.new_df['movie_id'] == movie_id]
                        if not movie.empty:
                            # Add a larger random factor to ensure variety when toggling models
                            score = user_predictions[movie_id] + (np.random.random() * 0.5)
                            filtered_movies.append((movie.iloc[0]['title'], movie_id, score))
                            if len(filtered_movies) >= n:
                                break
            
            # Sort by score and extract recommendations
            filtered_movies.sort(key=lambda x: x[2], reverse=True)
            top_movies = [(title, movie_id) for title, movie_id, _ in filtered_movies[:n]]
            
            # If we have enough filtered movies, return them
            if len(top_movies) >= n:
                return top_movies
        
        # Get top N movie IDs without language filtering or if not enough language-filtered movies
        top_movie_ids = user_predictions.head(n).index.tolist()
        top_movies = []
        for movie_id in top_movie_ids:
            movie = self.new_df[self.new_df['movie_id'] == movie_id]
            if not movie.empty:
                top_movies.append((movie.iloc[0]['title'], movie_id))
        
        # If we don't have enough recommendations, fill with content-based
        if len(top_movies) < n:
            content_recs = self.get_content_based_recommendations(username, n - len(top_movies), model_seed=random_seed)
            # Avoid duplicates
            existing_ids = [movie_id for _, movie_id in top_movies]
            for title, movie_id in content_recs:
                if movie_id not in existing_ids and len(top_movies) < n:
                    top_movies.append((title, movie_id))
        
        return top_movies[:n]
    
    def _get_popular_movies(self, n=10, model_seed=None):
        """Get popular movies based on vote average and vote count"""
        # Add randomization to ensure fresh results each time
        # Use provided seed if available, otherwise generate a new one
        random_seed = model_seed if model_seed is not None else np.random.randint(1000)
        np.random.seed(random_seed)
        
        # Join with movies2 to get vote information
        popular_df = self.new_df.merge(self.movies2[['movie_id', 'vote_average', 'vote_count']], on='movie_id')
        
        # Calculate popularity score with randomization
        popular_df['popularity_score'] = popular_df['vote_average'] * popular_df['vote_count'] + (np.random.random(len(popular_df)) * 1000)
        
        # Add variety by sometimes prioritizing newer movies
        if np.random.random() > 0.5:
            # Join with release date information
            popular_df = popular_df.merge(self.movies2[['movie_id', 'release_date']], on='movie_id')
            # Convert release_date to datetime
            popular_df['release_date'] = pd.to_datetime(popular_df['release_date'], errors='coerce')
            # Extract year
            popular_df['year'] = popular_df['release_date'].dt.year
            # Add recency bonus (more recent = higher score)
            current_year = pd.Timestamp.now().year
            popular_df['recency_bonus'] = popular_df['year'].apply(lambda x: max(0, min(1, (x - 2000) / 20)) if pd.notnull(x) else 0)
            # Add recency bonus to popularity score
            popular_df['popularity_score'] = popular_df['popularity_score'] * (1 + popular_df['recency_bonus'] * 0.5)
        
        # Sort by popularity score
        popular_df = popular_df.sort_values(by='popularity_score', ascending=False)
        
        # Get top N movies
        top_movies = popular_df.head(n*2)  # Get more than needed for random sampling
        
        # Randomly sample from top movies to add variety
        if len(top_movies) > n:
            top_movies = top_movies.sample(n, random_state=random_seed)
        
        return [(movie['title'], movie['movie_id']) for _, movie in top_movies.iterrows()]
    
    def get_recommendations(self, username, model_type='hybrid', n=10):
        """Get recommendations based on specified model type"""
        start_time = time.time()
        recommendations = []
        
        # Use a simpler seed generation for faster model switching
        # This still ensures different models produce different results
        model_type_seed = {'hybrid': 1000, 'content': 2000, 'collaborative': 3000}[model_type]
        # Use a simpler time seed that changes less frequently
        current_time_seed = int(time.time() / 10) % 100
        random_seed = model_type_seed + current_time_seed
        
        if model_type == 'content':
            # Use a specific seed for content-based recommendations
            recommendations = self.get_content_based_recommendations(username, n, model_seed=random_seed)
        elif model_type == 'collaborative':
            # Use a specific seed for collaborative recommendations
            recommendations = self.get_collaborative_recommendations(username, n, model_seed=random_seed)
        else:  # hybrid approach
            # Get recommendations from both models with different weights and different seeds
            content_seed = random_seed + 500  # Different seed for content
            collab_seed = random_seed + 1000  # Different seed for collaborative
            
            content_recs = self.get_content_based_recommendations(username, n, model_seed=content_seed)
            collab_recs = self.get_collaborative_recommendations(username, n, model_seed=collab_seed)
            
            # Get user activity metrics
            user_ratings = self._load_json(self.ratings_file).get(username, {})
            user_prefs = self._load_json(self.preferences_file).get(username, {})
            
            # Simplified weighting for faster model switching
            rating_weight = min(0.7, len(user_ratings) / 20)
            pref_weight = 0.6 if user_prefs and any(user_prefs.values()) else 0.4
            
            # Normalize weights
            total_weight = rating_weight + pref_weight
            collab_weight = rating_weight / total_weight
            content_weight = pref_weight / total_weight
            
            # Combine recommendations with weights and diversity
            hybrid_recs = []
            seen_ids = set()
            
            # Add collaborative recommendations
            for i, rec in enumerate(collab_recs):
                if rec[1] not in seen_ids:
                    # Add more randomization to ensure variety between model types
                    # Increased randomization factor for more distinct results when toggling
                    score = collab_weight * (1 - i/n) + 0.4 * np.random.random()
                    hybrid_recs.append((rec, score))
                    seen_ids.add(rec[1])
            
            # Add content recommendations
            for i, rec in enumerate(content_recs):
                if rec[1] not in seen_ids:
                    # Add more randomization to ensure variety between model types
                    # Increased randomization factor for more distinct results when toggling
                    score = content_weight * (1 - i/n) + 0.4 * np.random.random()
                    hybrid_recs.append((rec, score))
                    seen_ids.add(rec[1])
            
            # Sort by score and get top N
            hybrid_recs.sort(key=lambda x: x[1], reverse=True)
            recommendations = [rec for rec, _ in hybrid_recs[:n]]
        
        # Record metrics
        response_time = time.time() - start_time
        metrics_model_type = 'content_based' if model_type == 'content' else model_type
        
        self.metrics.record_metrics(
            metrics_model_type,
            username,
            recommendations,
            response_time
        )
        
        return recommendations
        
    def get_model_performance(self, model_type):
        """Get performance metrics for a recommendation model"""
        return self.metrics.get_model_performance(model_type.replace('-', '_'))
    
    def record_user_satisfaction(self, model_type, username, rating):
        """Record user satisfaction with recommendations"""
        self.metrics.record_user_satisfaction(model_type.replace('-', '_'), username, rating)
        
    def get_user_satisfaction(self, model_type):
        """Get average user satisfaction for a model"""
        return self.metrics.get_user_satisfaction(model_type.replace('-', '_'))