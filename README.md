# Movie Recommender System

Unlock Your Next Favorite Film! Our NLP-powered Movie Recommendation Web App delivers tailored suggestions based on cast, genres, and production companies. Explore a seamless Streamlit interface with personalized recommendations, movie details, and a comprehensive movie catalog.

## Project Overview

Our Movie Recommender System, built using Python and Natural Language Processing (NLP), offers a user-friendly way to discover your next favorite movie. The system now features multiple recommendation engines:

- **Hybrid Recommendations**: Combines personal preferences with popular choices for the best of both worlds
- **Content-Based Filtering**: Suggests movies based on your favorite genres, directors, and movie styles
- **Collaborative Filtering**: Recommends movies enjoyed by users with similar taste

Key Features:
- Personalized movie recommendations based on your preferences and ratings
- Detailed movie information including cast, crew, and production details
- Similar movie suggestions based on various factors (tags, genres, production companies)
- Performance metrics to track recommendation accuracy
- User preference management and rating system
- Comprehensive movie catalog with easy navigation

## Sample Application Screenshots

### Recommendation Interface
![Home Screen](images/Screenshot%202025-03-08%20at%2019.32.16.png)
![Personalized Recommendations](images/Screenshot%202025-03-08%20at%2019.32.42.png)

**Recommendation Page:** Discover personalized movie suggestions based on your preferences and ratings.

### Movie Details
![Movie Information](images/Screenshot%202025-03-08%20at%2019.33.14.png)
![Cast Information](images/Screenshot%202025-03-08%20at%2019.33.34.png)

**Description Page:** Explore essential movie details and information about the cast.

### Performance Dashboard
![Metrics Dashboard](images/Screenshot%202025-03-08%20at%2019.41.57.png)

**Analytics Page:** Track recommendation performance and user engagement metrics.

## Installation Guide

Follow these steps to set up and run the application:

1. **Clone the Repository:** 
    ```bash
    git clone https://github.com/AsjadAfnan/Movie-Recommender-System.git
    ```

2. **Create a Virtual Environment:** 
   Make sure you have a virtual environment set up for your project.

3. **Install Dependencies:**
   Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   To start the app, execute the following command in your terminal:
   ```bash
   streamlit run main.py
   ```

**Note**: When running the application for the first time, it may take some time as it creates necessary files and initializes the environment.

## Features in Detail

### Recommendation Engines

1. **Hybrid Recommendations**
   - Combines collaborative and content-based filtering
   - Balances personal preferences with community favorites
   - Provides diverse and relevant suggestions

2. **Content-Based Filtering**
   - Analyzes movie attributes (genres, directors, cast)
   - Considers user-specified preferences
   - Matches movies based on similarity scores

3. **Collaborative Filtering**
   - Uses SVD (Singular Value Decomposition) for user-item interactions
   - Identifies similar user patterns
   - Suggests movies based on community ratings

### Performance Metrics

- Track recommendation accuracy
- Monitor user engagement
- Analyze rating patterns
- Measure system effectiveness

### User Features

- Personalized preference settings
- Movie rating system
- Watch history tracking
- Custom recommendation styles

Discover the joy of finding your next favorite movie with our Movie Recommender System!
