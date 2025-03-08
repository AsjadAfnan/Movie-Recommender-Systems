import streamlit as st
import pymongo
from pathlib import Path
import bcrypt

class UserAuth:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="movie_recommender"):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]  # Access the database
        self.users_collection = self.db["users"]
        self.ratings_collection = self.db["user_ratings"]
        self.preferences_collection = self.db["user_preferences"]
        self._initialize_collections()

    def _initialize_collections(self):
        # Check if indexes are needed (e.g., for usernames)
        if "username_index" not in self.users_collection.index_information():
            self.users_collection.create_index("username", name="username_index", unique=True)

    def register_user(self, username, password):
        if self.users_collection.find_one({"username": username}):
            return False, "Username already exists"

        # Hash the password using bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        user_data = {
            "username": username,
            "password": hashed_password,  # Store the hashed password
            "completed_onboarding": False,
        }
        self.users_collection.insert_one(user_data)
        return True, "Registration successful"

    def login_user(self, username, password):
        user = self.users_collection.find_one({"username": username})
        if not user:
            return False, "Invalid username or password"

        # Verify the password using bcrypt
        hashed_password = user["password"]  # Get the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return True, "Login successful"
        else:
            return False, "Invalid username or password"

    def save_user_preferences(self, username, preferences):
        # Update preferences if they exist, otherwise insert
        self.preferences_collection.update_one(
            {"username": username},
            {"$set": {"preferences": preferences}},
            upsert=True
        )

        # Mark onboarding as completed
        self.users_collection.update_one(
            {"username": username},
            {"$set": {"completed_onboarding": True}}
        )

    def save_user_preferences(self, username, preferences):
        # Update preferences if they exist, otherwise insert
        self.preferences_collection.update_one(
            {"username": username},
            {"$set": {"preferences": preferences}},
            upsert=True
        )

        # Mark onboarding as completed
        self.users_collection.update_one(
            {"username": username},
            {"$set": {"completed_onboarding": True}}
        )

    def get_user_preferences(self, username):
        user_doc = self.preferences_collection.find_one({"username": username})
        if user_doc and "preferences" in user_doc:  # Changed this line
            return user_doc["preferences"]
        return {}
    
    def save_user_rating(self, username, movie_id, rating):
        # Use update_one with $set to add or update the nested field
        self.ratings_collection.update_one(
            {"username": username},
            {"$set": {f"ratings.{movie_id}": rating}},
            upsert=True
        )

    def get_user_ratings(self, username):
        user_doc = self.ratings_collection.find_one({"username": username})
        if user_doc and "ratings" in user_doc:
            return user_doc["ratings"]
        return {}

    def needs_onboarding(self, username):
        user = self.users_collection.find_one({"username": username})
        return not (user and user.get("completed_onboarding", False))

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

def render_onboarding():
    st.title('Welcome! Let\'s Get to Know Your Movie Preferences')

    # Collect user preferences through 5 questions
    preferences = {}

    # Question 1: Favorite Genres
    preferences['favorite_genres'] = st.multiselect(
        'What are your favorite movie genres? (Select up to 3)',
        ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'],
        max_selections=3
    )

    # Question 2: Movie Era Preference
    preferences['preferred_era'] = st.select_slider(
        'Which era of movies do you prefer?',
        options=['Classic (Pre-1970)', '1970s-1990s', '1990s-2010s', 'Modern (2010+)'],
        value='1990s-2010s'
    )

    # Question 3: Preferred Directors
    preferences['preferred_directors'] = st.text_input(
        'Who are your favorite directors? (Separate names with commas)',
        placeholder='e.g. Christopher Nolan, Steven Spielberg, Quentin Tarantino'
    )

    # Question 4: Mood Preference (replacing movie duration)
    preferences['preferred_mood'] = st.select_slider(
        'What type of movie mood do you typically enjoy?',
        options=['Light & Funny', 'Thought-provoking', 'Intense & Thrilling', 'Emotional & Dramatic', 'Escapist & Fantasy'],
        value='Thought-provoking'
    )

    # Question 5: Content Language
    preferences['preferred_languages'] = st.multiselect(
        'What languages do you prefer for movies? (Select all that apply)',
        ['English', 'Spanish', 'French', 'Hindi', 'Japanese', 'Korean', 'Chinese'],
        default=['English']
    )

    return preferences