import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

class UserAuth:
    def __init__(self):
        self.users_file = Path('Files/users.json')
        self.ratings_file = Path('Files/user_ratings.json')
        self.preferences_file = Path('Files/user_preferences.json')
        self._initialize_files()

    def _initialize_files(self):
        # Create Files directory if it doesn't exist
        Path('Files').mkdir(exist_ok=True)
        
        # Initialize users file
        if not self.users_file.exists():
            self._save_json(self.users_file, {})
        
        # Initialize ratings file
        if not self.ratings_file.exists():
            self._save_json(self.ratings_file, {})
            
        # Initialize preferences file
        if not self.preferences_file.exists():
            self._save_json(self.preferences_file, {})

    def _save_json(self, file_path, data):
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def register_user(self, username, password):
        users = self._load_json(self.users_file)
        if username in users:
            return False, 'Username already exists'
        
        users[username] = {
            'password': password,
            'completed_onboarding': False
        }
        self._save_json(self.users_file, users)
        return True, 'Registration successful'

    def login_user(self, username, password):
        users = self._load_json(self.users_file)
        if username not in users or users[username]['password'] != password:
            return False, 'Invalid username or password'
        return True, 'Login successful'

    def save_user_preferences(self, username, preferences):
        user_prefs = self._load_json(self.preferences_file)
        user_prefs[username] = preferences
        self._save_json(self.preferences_file, user_prefs)
        
        # Mark onboarding as completed
        users = self._load_json(self.users_file)
        users[username]['completed_onboarding'] = True
        self._save_json(self.users_file, users)

    def get_user_preferences(self, username):
        user_prefs = self._load_json(self.preferences_file)
        return user_prefs.get(username, {})

    def save_user_rating(self, username, movie_id, rating):
        ratings = self._load_json(self.ratings_file)
        if username not in ratings:
            ratings[username] = {}
            
        # Check if this is a new rating or an update
        is_new_rating = str(movie_id) not in ratings[username]
        old_rating = ratings[username].get(str(movie_id), 0)
        
        # Save the new rating
        ratings[username][str(movie_id)] = rating
        self._save_json(self.ratings_file, ratings)
        
        # Trigger recommendation model update
        # This ensures real-time model retraining based on user feedback
        try:
            from processing.recommendation import RecommendationEngine
            rec_engine = RecommendationEngine()
            
            # Force refresh recommendations in session state
            import streamlit as st
            st.session_state.refresh_recommendations = True
            
            # Reset loaded movies count to show fresh recommendations
            if 'loaded_movies_count' in st.session_state:
                st.session_state.loaded_movies_count = 10
                
            # Clear any cached recommendations to ensure fresh results
            if 'cached_recommendations' in st.session_state:
                del st.session_state.cached_recommendations
                
            # Also clear cached model type to force a complete refresh
            if 'cached_model_type' in st.session_state:
                del st.session_state.cached_model_type
                
            # Clear cached timestamp to ensure fresh recommendations
            if 'cached_timestamp' in st.session_state:
                del st.session_state.cached_timestamp
            
            # Record the rating change in metrics for model improvement
            # For both new ratings and updates, as both affect recommendations
            rating_change = abs(rating - old_rating) if not is_new_rating else rating
            if rating_change > 0:
                from processing.metrics import RecommendationMetrics
                metrics = RecommendationMetrics()
                # Record the rating change for all model types to improve them
                for model_type in ['content_based', 'collaborative', 'hybrid']:
                    metrics.record_rating_change(model_type, username, movie_id, old_rating, rating)
                    
            # Log the rating activity
            print(f"User {username} rated movie {movie_id} with {rating} stars. Recommendations will refresh.")
        except Exception as e:
            print(f"Error updating recommendation models: {e}")
            # Even if update fails, the rating is still saved

    def get_user_ratings(self, username):
        ratings = self._load_json(self.ratings_file)
        return ratings.get(username, {})

    def needs_onboarding(self, username):
        users = self._load_json(self.users_file)
        return not users[username]['completed_onboarding']

def render_auth_page():
    if 'user' not in st.session_state:
        st.session_state.user = None

    auth = UserAuth()

    if not st.session_state.user:
        tab1, tab2 = st.tabs(['Login', 'Register'])
        
        with tab1:
            st.subheader('Login')
            login_username = st.text_input('Username', key='login_username')
            login_password = st.text_input('Password', type='password', key='login_password')
            
            if st.button('Login'):
                success, message = auth.login_user(login_username, login_password)
                if success:
                    st.session_state.user = login_username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with tab2:
            st.subheader('Register')
            reg_username = st.text_input('Username', key='reg_username')
            reg_password = st.text_input('Password', type='password', key='reg_password')
            
            if st.button('Register'):
                success, message = auth.register_user(reg_username, reg_password)
                if success:
                    st.success(message)
                    st.session_state.user = reg_username
                    st.rerun()
                else:
                    st.error(message)

    return auth

def render_onboarding(suffix=''):
    # Generate a truly unique suffix for widget keys
    import time
    import random
    
    # Create a unique identifier based on timestamp and random number
    if '_onboarding_key' not in st.session_state:
        st.session_state['_onboarding_key'] = f"{suffix}_{time.time()}_{random.randint(1000, 9999)}"
    
    # Use the stored unique key
    unique_suffix = st.session_state['_onboarding_key']
    
    st.title('Welcome! Let\'s Get to Know Your Movie Preferences')
    
    # Collect user preferences through 5 questions
    preferences = {}
    
    # Question 1: Favorite Genres
    preferences['favorite_genres'] = st.multiselect(
        'What are your favorite movie genres? (Select up to 3)',
        ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'],
        max_selections=3,
        key=f'onboarding_genres{unique_suffix}'
    )
    
    # Question 2: Movie Era Preference
    preferences['preferred_era'] = st.select_slider(
        'Which era of movies do you prefer?',
        options=['Classic (Pre-1970)', '1970s-1990s', '1990s-2010s', 'Modern (2010+)'],
        value='1990s-2010s',
        key=f'onboarding_era{unique_suffix}'
    )
    
    # Question 3: Preferred Directors
    preferences['preferred_directors'] = st.text_input(
        'Who are your favorite directors? (Separate names with commas)',
        placeholder='e.g. Christopher Nolan, Steven Spielberg, Quentin Tarantino',
        key=f'onboarding_directors{unique_suffix}'
    )
    
    # Question 4: Mood Preference
    preferences['preferred_mood'] = st.select_slider(
        'What type of movie mood do you typically enjoy?',
        options=['Light & Funny', 'Thought-provoking', 'Intense & Thrilling', 'Emotional & Dramatic', 'Escapist & Fantasy'],
        value='Thought-provoking',
        key=f'onboarding_mood{unique_suffix}'
    )
    
    # Question 5: Content Language
    preferences['preferred_languages'] = st.multiselect(
        'What languages do you prefer for movies? (Select all that apply)',
        ['English', 'Spanish', 'French', 'Hindi', 'Japanese', 'Korean', 'Chinese'],
        default=['English'],
        key=f'onboarding_languages{unique_suffix}'
    )
    
    return preferences