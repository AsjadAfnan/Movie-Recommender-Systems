import streamlit as st
import streamlit_option_menu
from streamlit_extras.stoggle import stoggle
import pandas as pd
import time
import numpy as np
from processing import preprocess
from processing.display import Main
from processing.auth import render_auth_page, render_onboarding, UserAuth
from processing.recommendation import RecommendationEngine
from processing.metrics_display import render_metrics_page
import nltk

# Setting the wide mode as default
st.set_page_config(layout="wide")

displayed = []

# Initialize session state variables
if 'movie_number' not in st.session_state:
    st.session_state['movie_number'] = 0

if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = ""

if 'user_menu' not in st.session_state:
    st.session_state['user_menu'] = ""
    
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = "hybrid"
    
if 'user' not in st.session_state:
    st.session_state['user'] = None
    
if 'onboarding_complete' not in st.session_state:
    st.session_state['onboarding_complete'] = False
    
if 'viewing_movie_details' not in st.session_state:
    st.session_state['viewing_movie_details'] = False
    
if 'refresh_recommendations' not in st.session_state:
    st.session_state['refresh_recommendations'] = False


# Define initial_options function at the module level, outside of any other function
def initial_options(new_df, movies, movies2):
    # Check if we're viewing movie details
    if st.session_state.viewing_movie_details:
        display_movie_details(new_df, movies, movies2)
        return
        
    # To display menu
    st.session_state.user_menu = streamlit_option_menu.option_menu(
        menu_title='What are you looking for? 👀',
        options=['Personalized Recommendations', 'Recommend me a similar movie', 'Check all Movies', 'Performance Metrics'],
        icons=['star', 'film', 'film', 'graph-up'],
        menu_icon='list',
        orientation="horizontal",
    )

    if st.session_state.user_menu == 'Personalized Recommendations':
        show_personalized_recommendations(new_df, movies, movies2)
        
    elif st.session_state.user_menu == 'Recommend me a similar movie':
        recommend_display(new_df, movies, movies2)

    elif st.session_state.user_menu == 'Check all Movies':
        paging_movies(new_df, movies, movies2)
        
    elif st.session_state.user_menu == 'Rate Movies':
        rate_movies()
        
    elif st.session_state.user_menu == 'Performance Metrics':
        render_metrics_page()

def show_personalized_recommendations(new_df, movies, movies2):
    st.title('Your Personalized Movie Recommendations')
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine()
    
    # Add model selection in the main content area
    st.subheader("Choose Your Recommendation Style")
    model_options = {
        "hybrid": "✨ Best of Both Worlds - Combines personal preferences with popular choices",
        "content": "🎯 Vibe Match - Based on your favorite genres, directors, and movie styles",
        "collaborative": "👥 Crowd Favorites - Movies enjoyed by people with similar taste"
    }
    
    selected_model = st.radio(
        "Select recommendation style:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x].split(' - ')[0],
        index=list(model_options.keys()).index(st.session_state.model_type),
        horizontal=True,
        key="model_selector_static"
    )
    
    # Show description of selected model
    st.caption(model_options[selected_model].split(' - ')[1])
    
    # Update model type if changed
    if selected_model != st.session_state.model_type:
        # Store the old model type for comparison
        old_model_type = st.session_state.model_type
        # Update the model type
        st.session_state.model_type = selected_model
        # Set flag to force refresh recommendations
        st.session_state.refresh_recommendations = True
        # Reset loaded movies count
        st.session_state.loaded_movies_count = 10
        # Clear only essential cache data
        st.session_state.cached_recommendations = None
        st.session_state.cached_model_type = selected_model
        st.session_state.cached_timestamp = pd.Timestamp.now()
        # Log the model switch for debugging
        print(f"Switching recommendation model from {old_model_type} to {selected_model}")
        # Force rerun to immediately show new recommendations
        st.rerun()
    
    # Force refresh recommendations if coming from a rating
    if 'refresh_recommendations' in st.session_state and st.session_state.refresh_recommendations:
        st.session_state.refresh_recommendations = False
        if 'loaded_movies_count' in st.session_state:
            st.session_state.loaded_movies_count = 10
        # Clear any cached recommendations to ensure fresh results
        if 'cached_recommendations' in st.session_state:
            del st.session_state.cached_recommendations
        # Clear cached model type to force a complete refresh
        if 'cached_model_type' in st.session_state:
            del st.session_state.cached_model_type
        # Clear cached timestamp to ensure fresh recommendations
        if 'cached_timestamp' in st.session_state:
            del st.session_state.cached_timestamp
    
    # Get more recommendations for infinite scrolling (30 instead of 10)
    with st.spinner(f"Generating your personalized recommendations using {model_options[st.session_state.model_type].split(' - ')[0]}..."):
        # Add a timestamp to force refresh when needed
        current_timestamp = pd.Timestamp.now()
        
        # Add a unique identifier to prevent caching
        cache_buster = np.random.randint(10000)
        
        # Simplified cache refresh logic
        if not st.session_state.get('cached_recommendations') or \
           st.session_state.cached_model_type != st.session_state.model_type or \
           st.session_state.refresh_recommendations or \
           (current_timestamp - st.session_state.get('cached_timestamp', pd.Timestamp.now())).total_seconds() > 300:  # Cache expires after 5 minutes
           
            # Add cache buster to URL parameters to prevent browser caching
            st.query_params.update(cache_buster=str(cache_buster))
            
            # Get fresh recommendations
            recommendations = rec_engine.get_recommendations(
                st.session_state.user, 
                model_type=st.session_state.model_type,
                n=30
            )
            # Cache the recommendations, model type, and timestamp
            st.session_state.cached_recommendations = recommendations
            st.session_state.cached_model_type = st.session_state.model_type
            st.session_state.cached_timestamp = current_timestamp
            st.session_state.refresh_recommendations = False
        else:
            # Use cached recommendations
            recommendations = st.session_state.cached_recommendations
    
    if not recommendations:
        st.info("We don't have enough data to make personalized recommendations yet. Please rate some movies first.")
        return
        
    # Display recommendations in a grid with infinite scrolling
    # Initialize state for loaded movies if not exists
    if 'loaded_movies_count' not in st.session_state:
        st.session_state.loaded_movies_count = 10
    
    # Display the currently loaded movies
    cols = st.columns(5)
    for i, (title, movie_id) in enumerate(recommendations[:st.session_state.loaded_movies_count]):
        col_idx = i % 5
        with cols[col_idx]:
            poster = preprocess.fetch_posters(movie_id)
            st.image(poster)
            st.write(title)
            
            # Add button to view details with a more stable key that doesn't depend on the position
            # This ensures the button works correctly even after loading more movies
            if st.button(f"View Details", key=f"view_movie_{movie_id}"):
                st.session_state.selected_movie_name = title
                st.session_state.viewing_movie_details = True
                st.rerun()
                return
    
    # Add a "Load More" button if there are more movies to show
    if st.session_state.loaded_movies_count < len(recommendations):
        if st.button("Load More Movies"):
            # Increase the number of loaded movies
            st.session_state.loaded_movies_count += 10
            st.rerun()

def rate_movies():
    st.title('Rate Movies')
    st.write("Your ratings help us provide better recommendations!")
    
    # Initialize recommendation engine to get popular movies
    rec_engine = RecommendationEngine()
    popular_movies = rec_engine._get_popular_movies(20)
    
    # Get user's existing ratings
    user_auth = UserAuth()
    user_ratings = user_auth.get_user_ratings(st.session_state.user)
    
    # Display movies to rate in a grid
    cols = st.columns(5)
    for i, (title, movie_id) in enumerate(popular_movies):
        col_idx = i % 5
        with cols[col_idx]:
            poster = preprocess.fetch_posters(movie_id)
            st.image(poster)
            st.write(title)
            
            # Show current rating if exists
            current_rating = user_ratings.get(str(movie_id), 0)
            
            # Rating input with star system
            rating_css = """
            <style>
                div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
                    font-size: 20px;
                    font-family: Arial;
                }
                div[data-testid="stRadio"] > div[role="radiogroup"] > label {
                    background-color: transparent !important;
                    padding: 0px !important;
                    margin-right: 5px !important;
                }
            </style>
            """
            st.markdown(rating_css, unsafe_allow_html=True)
            # Define star rating options
            star_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
            new_rating = st.radio(
                f"Rate",
                options=star_options,
                horizontal=True,
                key=f"rate_{movie_id}",
                index=max(0, min(int(current_rating)-1, 4)) if current_rating > 0 else 0
            )
            
            # Convert star rating to numeric (1-5 based on index + 1)
            new_rating_value = star_options.index(new_rating) + 1 if new_rating else 0
            
            # Save rating if changed
            if new_rating_value != current_rating and new_rating_value > 0:
                user_auth.save_user_rating(st.session_state.user, movie_id, new_rating_value)
                st.success(f"Rating saved for {title}! Your recommendations will be updated.")
                
                # Set flag to refresh recommendations
                st.session_state.refresh_recommendations = True
                
                # Reset loaded_movies_count to refresh recommendations
                if 'loaded_movies_count' in st.session_state:
                    st.session_state.loaded_movies_count = 10
                    
                # Clear any cached recommendations
                if 'cached_recommendations' in st.session_state:
                    del st.session_state.cached_recommendations
                    
                # Also clear cached model type to force a complete refresh
                if 'cached_model_type' in st.session_state:
                    del st.session_state.cached_model_type
                    
                # Clear cached timestamp to ensure fresh recommendations
                if 'cached_timestamp' in st.session_state:
                    del st.session_state.cached_timestamp

def recommend_display(new_df, movies, movies2):
    st.title('Movie Recommender System')

    selected_movie_name = st.selectbox(
        'Select a Movie...', new_df['title'].values
    )

    rec_button = st.button('Recommend')
    if rec_button:
        st.session_state.selected_movie_name = selected_movie_name
        recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tags.pkl',"are")
        recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_genres.pkl',"on the basis of genres are")
        recommendation_tags(new_df, selected_movie_name,
                            r'Files/similarity_tags_tprduction_comp.pkl',"from the same production company are")
        recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_keywords.pkl',"on the basis of keywords are")
        recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tcast.pkl',"on the basis of cast are")

def recommendation_tags(new_df, selected_movie_name, pickle_file_path,str):

    movies, posters = preprocess.recommend(new_df, selected_movie_name, pickle_file_path)
    st.subheader(f'Best Recommendations {str}...')

    rec_movies = []
    rec_posters = []
    cnt = 0
    # Adding only 5 uniques recommendations
    for i, j in enumerate(movies):
        if cnt == 5:
            break
        if j not in displayed:
            rec_movies.append(j)
            rec_posters.append(posters[i])
            displayed.append(j)
            cnt += 1

    # Columns to display informations of movies i.e. movie title and movie poster
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(rec_movies[0])
        st.image(rec_posters[0])
    with col2:
        st.text(rec_movies[1])
        st.image(rec_posters[1])
    with col3:
        st.text(rec_movies[2])
        st.image(rec_posters[2])
    with col4:
        st.text(rec_movies[3])
        st.image(rec_posters[3])
    with col5:
        st.text(rec_movies[4])
        st.image(rec_posters[4])

def display_movie_details(new_df, movies, movies2):
    # Make sure user is in session state
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.error("Please login to view movie details")
        st.session_state.viewing_movie_details = False
        st.rerun()
        return
        
    # Add a back button at the top
    if st.button("← Back to Recommendations"):
        st.session_state.viewing_movie_details = False
        st.rerun()
        return

    selected_movie_name = st.session_state['selected_movie_name']
    info = preprocess.get_details(selected_movie_name)

    # Create a new container for the movie details
    st.title(selected_movie_name)
    
    # Use two columns for the main layout
    image_col, text_col = st.columns((1, 2))
    
    with image_col:
        st.image(info[0])
        
        # Add rating system
        user_auth = UserAuth()
        movie_id = new_df[new_df['title'] == selected_movie_name]['movie_id'].iloc[0]
        current_rating = user_auth.get_user_ratings(st.session_state.user).get(str(movie_id), 0) if st.session_state.user else 0
        
        st.write("Rate this movie:")
        # Create star rating using radio buttons with custom styling
        rating_css = """
        <style>
            div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
                font-size: 24px;
                font-family: Arial;
            }
            div[data-testid="stRadio"] > div[role="radiogroup"] > label {
                background-color: transparent !important;
                padding: 0px !important;
                margin-right: 10px !important;
            }
        </style>
        """
        st.markdown(rating_css, unsafe_allow_html=True)
        # Ensure index is always valid (between 0 and 4)
        try:
            rating_index = max(0, min(int(current_rating)-1, 4)) if current_rating else 0
        except:
            rating_index = 0
            
        new_rating = st.radio(
            "Select your rating",
            options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            horizontal=True,
            key=f"rate_detail_{movie_id}",
            index=rating_index
        )
        # Convert star rating back to numeric
        new_rating_value = len(new_rating) // 2 if new_rating else 0
        
        # Save rating if changed
        if new_rating_value != current_rating and new_rating_value > 0:
            user_auth.save_user_rating(st.session_state.user, movie_id, new_rating_value)
            st.success("Rating saved! Your recommendations will be updated.")
            # Force refresh recommendations
            st.session_state.refresh_recommendations = True
            # Reset loaded movies count to show fresh recommendations
            if 'loaded_movies_count' in st.session_state:
                st.session_state.loaded_movies_count = 10
            # Return to recommendations page
            st.session_state.viewing_movie_details = False
            st.rerun()

    with text_col:
        # Display basic info
        st.subheader("Movie Information")
        st.write(f"**Average Rating:** {info[8]}")
        st.write(f"**Number of ratings:** {info[9]}")
        st.write(f"**Runtime:** {info[6]}")
        
        # Overview
        st.subheader("Overview")
        st.write(info[3])
        
        # Additional details
        st.subheader("Additional Details")
        st.write(f"**Release Date:** {info[4]}")
        st.write(f"**Budget:** {info[1]}")
        st.write(f"**Revenue:** {info[5]}")
        
        # Genres
        genres_str = " . ".join(info[2])
        st.write(f"**Genres:** {genres_str}")
        
        # Available in
        available_str = " . ".join(info[13])
        st.write(f"**Available in:** {available_str}")
        
        # Director
        st.write(f"**Directed by:** {info[12][0]}")

    # Displaying information of casts in a separate section
    st.header('Cast')
    cnt = 0
    urls = []
    bio = []
    for i in info[14]:
        if cnt == 5:
            break
        url, biography= preprocess.fetch_person_details(i)
        urls.append(url)
        bio.append(biography)
        cnt += 1

    # Cast columns
    cast_cols = st.columns(5)
    for i in range(5):
        with cast_cols[i]:
            st.image(urls[i])
            stoggle("Show More", bio[i])

def paging_movies(new_df, movies, movies2):
    # To create pages functionality using session state.
    max_pages = movies.shape[0] / 10
    max_pages = int(max_pages) - 1

    col1, col2, col3 = st.columns([1, 9, 1])

    with col1:
        st.text("Previous page")
        prev_btn = st.button("Prev")
        if prev_btn:
            if st.session_state['movie_number'] >= 10:
                st.session_state['movie_number'] -= 10

    with col2:
        new_page_number = st.slider("Jump to page number", 0, max_pages, st.session_state['movie_number'] // 10)
        st.session_state['movie_number'] = new_page_number * 10

    with col3:
        st.text("Next page")
        next_btn = st.button("Next")
        if next_btn:
            if st.session_state['movie_number'] + 10 < len(movies):
                st.session_state['movie_number'] += 10

    display_all_movies(movies, st.session_state['movie_number'])

def display_all_movies(movies, start):

    i = start
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            id = movies.iloc[i]['movie_id']
            link = preprocess.fetch_posters(id)
            st.image(link, caption=movies['title'][i])
            i = i + 1

        with col2:
            id = movies.iloc[i]['movie_id']
            link = preprocess.fetch_posters(id)
            st.image(link, caption=movies['title'][i])
            i = i + 1

        with col3:
            id = movies.iloc[i]['movie_id']
            link = preprocess.fetch_posters(id)
            st.image(link, caption=movies['title'][i])
            i = i + 1

        with col4:
            id = movies.iloc[i]['movie_id']
            link = preprocess.fetch_posters(id)
            st.image(link, caption=movies['title'][i])
            i = i + 1

        with col5:
            id = movies.iloc[i]['movie_id']
            link = preprocess.fetch_posters(id)
            st.image(link, caption=movies['title'][i])
            i = i + 1

    st.session_state['page_number'] = i

def main():
    # Initialize session state variables if they don't exist
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if 'onboarding_complete' not in st.session_state:
        st.session_state['onboarding_complete'] = False
    
    # Handle authentication first
    auth = render_auth_page()
    
    # If user is logged in
    if st.session_state.get('user'):
        # Show user info in sidebar
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state['user']}**")
            
            # Logout button
            if st.button("Logout"):
                st.session_state['user'] = None
                st.session_state['onboarding_complete'] = False
                st.rerun()

    with Main() as bot:
        bot.main_()
        new_df, movies, movies2 = bot.getter()

        # Show main app interface if user is logged in
        if st.session_state.get('user'):
            user_auth = UserAuth()
            # Check if user needs onboarding
            if user_auth.needs_onboarding(st.session_state['user']) and not st.session_state['onboarding_complete']:
                preferences = render_onboarding('onboarding')
                if st.button('Save Preferences', key="save_preferences"):
                    user_auth.save_user_preferences(st.session_state['user'], preferences)
                    st.session_state['onboarding_complete'] = True
                    st.success("Preferences saved! Now showing your personalized recommendations.")
                    st.rerun()
            else:
                # Use the dataframes already loaded by the Main class
                # No need to reload them
                initial_options(new_df, movies, movies2)

    return new_df, movies, movies2


if __name__ == '__main__':
    main()
