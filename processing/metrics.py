import numpy as np
import pandas as pd
from collections import defaultdict
import time
from pathlib import Path
import json

class RecommendationMetrics:
    def __init__(self):
        self.ratings_file = Path('Files/user_ratings.json')
        self.preferences_file = Path('Files/user_preferences.json')
        self.metrics_file = Path('Files/recommendation_metrics.json')
        self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        """Initialize metrics file if it doesn't exist"""
        Path('Files').mkdir(exist_ok=True)
        if not self.metrics_file.exists():
            initial_metrics = {
                'content_based': {
                    'genre_diversity': [],
                    'response_time': [],
                    'user_satisfaction': defaultdict(list)
                },
                'collaborative': {
                    'prediction_accuracy': [],
                    'response_time': [],
                    'user_satisfaction': defaultdict(list)
                },
                'hybrid': {
                    'genre_diversity': [],
                    'prediction_accuracy': [],
                    'response_time': [],
                    'user_satisfaction': defaultdict(list)
                }
            }
            self._save_metrics(initial_metrics)
    
    def _load_metrics(self):
        """Load metrics from file"""
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def _save_metrics(self, metrics):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f)
    
    def calculate_genre_diversity(self, recommendations):
        """Calculate genre diversity score for recommendations"""
        genres = set()
        total_genres = 0
        
        # Load movie data from pickle files if needed
        try:
            import pickle
            with open('Files/new_df_dict.pkl', 'rb') as f:
                new_df_dict = pickle.load(f)
                new_df = pd.DataFrame.from_dict(new_df_dict)
                
            for _, movie_id in recommendations:
                movie = new_df[new_df['movie_id'] == movie_id]
                if not movie.empty:
                    movie_genres = movie.iloc[0]['genres'].lower().split()
                    genres.update(movie_genres)
                    total_genres += len(movie_genres)
        except Exception as e:
            # Fallback to placeholder if data loading fails
            print(f"Error loading genre data: {e}")
            return 0.5  # Default medium diversity
        
        # Calculate diversity as ratio of unique genres to total genres
        if total_genres == 0:
            return 0
            
        # Normalize to 0-1 scale
        return min(1.0, len(genres) / min(total_genres, 10))
    
    def calculate_prediction_accuracy(self, username, recommendations):
        """Calculate prediction accuracy using user ratings"""
        try:
            with open(self.ratings_file, 'r') as f:
                all_ratings = json.load(f)
                user_ratings = all_ratings.get(username, {})
            
            if not user_ratings:
                return None
                
            # Get all users for collaborative filtering
            if len(all_ratings) < 3:  # Need at least 3 users for meaningful CF
                return None
                
            # Convert ratings to dataframe for matrix factorization
            ratings_data = []
            for user, movie_ratings in all_ratings.items():
                for movie_id, rating in movie_ratings.items():
                    ratings_data.append({
                        'user': user,
                        'movie_id': int(movie_id),
                        'rating': rating
                    })
            
            if not ratings_data:
                return None
                
            ratings_df = pd.DataFrame(ratings_data)
            
            # Create user-item matrix
            user_item_matrix = ratings_df.pivot(
                index='user', 
                columns='movie_id', 
                values='rating'
            ).fillna(0)
            
            # Check if user exists in matrix
            if username not in user_item_matrix.index:
                return None
                
            # Perform SVD for prediction
            try:
                from scipy.sparse.linalg import svds
                k_factors = min(user_item_matrix.shape[0]-1, 10)
                U, sigma, Vt = svds(user_item_matrix.values, k=k_factors)
                sigma = np.diag(sigma)
                predicted_ratings = np.dot(np.dot(U, sigma), Vt)
                
                # Convert to dataframe
                preds_df = pd.DataFrame(
                    predicted_ratings, 
                    columns=user_item_matrix.columns,
                    index=user_item_matrix.index
                )
                
                # Get user's predictions
                user_idx = user_item_matrix.index.get_loc(username)
                user_predictions = preds_df.iloc[user_idx]
                
                # Calculate RMSE for available ratings
                squared_errors = []
                for movie_id_str, actual_rating in user_ratings.items():
                    try:
                        movie_id = int(movie_id_str)
                        if movie_id in user_predictions.index:
                            predicted_rating = user_predictions[movie_id]
                            squared_errors.append((actual_rating - predicted_rating) ** 2)
                    except (ValueError, KeyError):
                        continue
                
                if not squared_errors:
                    return None
                
                # Calculate RMSE and normalize to 0-1 scale (higher is better)
                rmse = np.sqrt(np.mean(squared_errors))
                return max(0, 1 - (rmse / 5))
            except Exception as e:
                print(f"Error in SVD calculation: {e}")
                return None
        except Exception as e:
            print(f"Error in prediction accuracy calculation: {e}")
            return None
    
    def record_metrics(self, model_type, username, recommendations, response_time):
        """Record metrics for a recommendation session"""
        metrics = self._load_metrics()
        
        # Record response time
        metrics[model_type]['response_time'].append(response_time)
        
        # Record genre diversity if applicable
        if model_type in ['content_based', 'hybrid']:
            diversity = self.calculate_genre_diversity(recommendations)
            metrics[model_type]['genre_diversity'].append(diversity)
        
        # Record prediction accuracy if applicable
        if model_type in ['collaborative', 'hybrid']:
            accuracy = self.calculate_prediction_accuracy(username, recommendations)
            if accuracy is not None:
                metrics[model_type]['prediction_accuracy'].append(accuracy)
        
        self._save_metrics(metrics)
    
    def get_model_performance(self, model_type):
        """Get aggregated performance metrics for a model"""
        metrics = self._load_metrics()
        model_metrics = metrics[model_type]
        
        performance = {
            'average_response_time': np.mean(model_metrics['response_time']) if model_metrics['response_time'] else 0
        }
        
        if 'genre_diversity' in model_metrics:
            performance['average_genre_diversity'] = np.mean(model_metrics['genre_diversity']) if model_metrics['genre_diversity'] else 0
        
        if 'prediction_accuracy' in model_metrics:
            performance['average_prediction_accuracy'] = np.mean(model_metrics['prediction_accuracy']) if model_metrics['prediction_accuracy'] else 0
        
        return performance
    
    def record_user_satisfaction(self, model_type, username, rating):
        """Record user satisfaction rating for recommendations"""
        metrics = self._load_metrics()
        # Check if username exists in user_satisfaction dict, if not initialize it
        if username not in metrics[model_type]['user_satisfaction']:
            metrics[model_type]['user_satisfaction'][username] = []
        metrics[model_type]['user_satisfaction'][username].append(rating)
        self._save_metrics(metrics)
    
    def get_user_satisfaction(self, model_type):
        """Get average user satisfaction for a model"""
        metrics = self._load_metrics()
        satisfaction_ratings = []
        for user_ratings in metrics[model_type]['user_satisfaction'].values():
            satisfaction_ratings.extend(user_ratings)
        return np.mean(satisfaction_ratings) if satisfaction_ratings else 0
        
    def record_rating_change(self, model_type, username, movie_id, old_rating, new_rating):
        """Record a rating change to improve model performance"""
        metrics = self._load_metrics()
        
        # Initialize rating_changes if it doesn't exist
        if 'rating_changes' not in metrics[model_type]:
            metrics[model_type]['rating_changes'] = []
            
        # Record the rating change
        metrics[model_type]['rating_changes'].append({
            'username': username,
            'movie_id': str(movie_id),
            'old_rating': old_rating,
            'new_rating': new_rating,
            'timestamp': time.time()
        })
        
        # Keep only the last 100 rating changes to avoid file size growth
        if len(metrics[model_type]['rating_changes']) > 100:
            metrics[model_type]['rating_changes'] = metrics[model_type]['rating_changes'][-100:]
            
        self._save_metrics(metrics)