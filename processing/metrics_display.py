import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from processing.recommendation import RecommendationEngine

class MetricsDisplay:
    def __init__(self):
        self.rec_engine = RecommendationEngine()
        
    def display_metrics_dashboard(self):
        """Display a dashboard of recommendation engine performance metrics"""
        st.title("Recommendation Engine Performance Metrics")
        
        # Get metrics for each model type
        content_metrics = self.rec_engine.get_model_performance('content_based')
        collab_metrics = self.rec_engine.get_model_performance('collaborative')
        hybrid_metrics = self.rec_engine.get_model_performance('hybrid')
        
        # Create tabs for different metric views
        tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Metrics", "Model Comparison"])
        
        with tab1:
            self._display_overview(content_metrics, collab_metrics, hybrid_metrics)
            
        with tab2:
            self._display_detailed_metrics(content_metrics, collab_metrics, hybrid_metrics)
            
        with tab3:
            self._display_model_comparison(content_metrics, collab_metrics, hybrid_metrics)
    
    def _display_overview(self, content_metrics, collab_metrics, hybrid_metrics):
        """Display overview of metrics"""
        st.header("Recommendation Engines Overview")
        
        # Create columns for each model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Content-Based")
            self._display_model_card(content_metrics, 'content_based')
            
        with col2:
            st.subheader("Collaborative")
            self._display_model_card(collab_metrics, 'collaborative')
            
        with col3:
            st.subheader("Hybrid")
            self._display_model_card(hybrid_metrics, 'hybrid')
    
    def _display_model_card(self, metrics, model_type):
        """Display a card with key metrics for a model"""
        # Format response time
        response_time = metrics.get('average_response_time', 0)
        response_time_str = f"{response_time:.2f} seconds"
        
        # Get user satisfaction
        # Convert 'content' to 'content_based' for consistency
        model_key = 'content_based' if model_type == 'content' else model_type
        satisfaction = self.rec_engine.get_user_satisfaction(model_key)
        satisfaction_str = f"{satisfaction:.1f}/5.0" if satisfaction > 0 else "No ratings yet"
        
        # Display metrics
        st.metric("Avg. Response Time", response_time_str)
        st.metric("User Satisfaction", satisfaction_str)
        
        # Display model-specific metrics
        if 'average_genre_diversity' in metrics:
            diversity = metrics['average_genre_diversity']
            st.metric("Genre Diversity", f"{diversity:.2f}")
            
        if 'average_prediction_accuracy' in metrics:
            accuracy = metrics['average_prediction_accuracy']
            st.metric("Prediction Accuracy", f"{accuracy:.2f}")
    
    def _display_detailed_metrics(self, content_metrics, collab_metrics, hybrid_metrics):
        """Display detailed metrics for each model"""
        st.header("Detailed Performance Metrics")
        
        # Select model to view detailed metrics
        model_type = st.selectbox(
            "Select recommendation model",
            ["Content-Based", "Collaborative", "Hybrid"],
            format_func=lambda x: x
        )
        
        if model_type == "Content-Based":
            metrics = content_metrics
            model_key = 'content_based'
        elif model_type == "Collaborative":
            metrics = collab_metrics
            model_key = 'collaborative'
        else:
            metrics = hybrid_metrics
            model_key = 'hybrid'
        
        # Display metrics in expandable sections
        with st.expander("Response Time", expanded=True):
            response_time = metrics.get('average_response_time', 0)
            st.metric("Average Response Time", f"{response_time:.3f} seconds")
            st.info("Response time measures how quickly recommendations are generated.")
        
        # Show genre diversity for content-based and hybrid
        if model_type in ["Content-Based", "Hybrid"]:
            with st.expander("Genre Diversity", expanded=True):
                diversity = metrics.get('average_genre_diversity', 0)
                st.metric("Genre Diversity Score", f"{diversity:.2f}")
                st.progress(min(diversity, 1.0))
                st.info("Genre diversity measures how varied the recommended movies are in terms of genres.")
        
        # Show prediction accuracy for collaborative and hybrid
        if model_type in ["Collaborative", "Hybrid"]:
            with st.expander("Prediction Accuracy", expanded=True):
                accuracy = metrics.get('average_prediction_accuracy', 0)
                st.metric("Prediction Accuracy", f"{accuracy:.2f}")
                st.progress(min(accuracy, 1.0))
                st.info("Prediction accuracy measures how well the model predicts your ratings.")
        
        # User satisfaction
        with st.expander("User Satisfaction", expanded=True):
            satisfaction = self.rec_engine.get_user_satisfaction(model_key)
            if satisfaction > 0:
                st.metric("Average User Satisfaction", f"{satisfaction:.1f}/5.0")
                # Display stars
                st.write("★" * int(round(satisfaction)) + "☆" * (5 - int(round(satisfaction))))
            else:
                st.info("No user satisfaction ratings yet.")
            
            # Allow users to rate the recommendations
            st.subheader("Rate these recommendations")
            user_rating = st.slider("How would you rate the recommendations?", 1, 5, 3)
            if st.button("Submit Rating"):
                if st.session_state.user:
                    self.rec_engine.record_user_satisfaction(model_key, st.session_state.user, user_rating)
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please log in to submit ratings.")
    
    def _display_model_comparison(self, content_metrics, collab_metrics, hybrid_metrics):
        """Display comparison between models"""
        st.header("Model Comparison")
        
        # Prepare data for charts
        data = {
            'Model': ['Content-Based', 'Collaborative', 'Hybrid'],
            'Response Time (s)': [
                content_metrics.get('average_response_time', 0),
                collab_metrics.get('average_response_time', 0),
                hybrid_metrics.get('average_response_time', 0)
            ],
            'User Satisfaction': [
                self.rec_engine.get_user_satisfaction('content_based'),
                self.rec_engine.get_user_satisfaction('collaborative'),
                self.rec_engine.get_user_satisfaction('hybrid')
            ]
        }
        
        # Add genre diversity
        data['Genre Diversity'] = [
            content_metrics.get('average_genre_diversity', 0),
            0,  # Collaborative doesn't have genre diversity
            hybrid_metrics.get('average_genre_diversity', 0)
        ]
        
        # Add prediction accuracy
        data['Prediction Accuracy'] = [
            0,  # Content-based doesn't have prediction accuracy
            collab_metrics.get('average_prediction_accuracy', 0),
            hybrid_metrics.get('average_prediction_accuracy', 0)
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Display as table
        st.subheader("Metrics Comparison")
        st.dataframe(df.set_index('Model'))
        
        # Create bar chart for response time
        st.subheader("Response Time Comparison")
        chart_data = pd.DataFrame({
            'Model': data['Model'],
            'Response Time (s)': data['Response Time (s)']
        })
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Model:N', sort=None),
            y='Response Time (s):Q',
            color=alt.Color('Model:N', legend=None)
        ).properties(
            width=600,
            height=300
        )
        st.altair_chart(chart)
        
        # Create radar chart for overall comparison
        st.subheader("Overall Performance Comparison")
        st.info("This chart shows relative performance across different metrics. Higher is better for all metrics except response time.")
        
        # Since Streamlit doesn't have built-in radar charts, we'll use a table view instead
        comparison_data = pd.DataFrame({
            'Metric': ['Response Time (lower is better)', 'User Satisfaction', 'Genre Diversity', 'Prediction Accuracy'],
            'Content-Based': [
                1 / (content_metrics.get('average_response_time', 0.1) + 0.1),  # Inverse for response time
                self.rec_engine.get_user_satisfaction('content_based'),
                content_metrics.get('average_genre_diversity', 0),
                0  # Content-based doesn't have prediction accuracy
            ],
            'Collaborative': [
                1 / (collab_metrics.get('average_response_time', 0.1) + 0.1),  # Inverse for response time
                self.rec_engine.get_user_satisfaction('collaborative'),
                0,  # Collaborative doesn't have genre diversity
                collab_metrics.get('average_prediction_accuracy', 0)
            ],
            'Hybrid': [
                1 / (hybrid_metrics.get('average_response_time', 0.1) + 0.1),  # Inverse for response time
                self.rec_engine.get_user_satisfaction('hybrid'),
                hybrid_metrics.get('average_genre_diversity', 0),
                hybrid_metrics.get('average_prediction_accuracy', 0)
            ]
        })
        
        st.dataframe(comparison_data.set_index('Metric'))
        
        # Recommendation
        st.subheader("Recommendation")
        best_model = self._determine_best_model(content_metrics, collab_metrics, hybrid_metrics)
        st.info(f"Based on current metrics, the **{best_model}** model appears to perform best overall.")
    
    def _determine_best_model(self, content_metrics, collab_metrics, hybrid_metrics):
        """Determine which model performs best overall"""
        # This is a simplified heuristic - in reality you'd want a more sophisticated approach
        scores = {
            'Content-Based': 0,
            'Collaborative': 0,
            'Hybrid': 0
        }
        
        # Response time (lower is better)
        response_times = [
            content_metrics.get('average_response_time', float('inf')),
            collab_metrics.get('average_response_time', float('inf')),
            hybrid_metrics.get('average_response_time', float('inf'))
        ]
        min_time = min(response_times)
        if min_time == response_times[0]:
            scores['Content-Based'] += 1
        elif min_time == response_times[1]:
            scores['Collaborative'] += 1
        else:
            scores['Hybrid'] += 1
        
        # User satisfaction
        satisfactions = [
            self.rec_engine.get_user_satisfaction('content_based'),
            self.rec_engine.get_user_satisfaction('collaborative'),
            self.rec_engine.get_user_satisfaction('hybrid')
        ]
        max_satisfaction = max(satisfactions)
        if max_satisfaction > 0:
            if max_satisfaction == satisfactions[0]:
                scores['Content-Based'] += 2
            elif max_satisfaction == satisfactions[1]:
                scores['Collaborative'] += 2
            else:
                scores['Hybrid'] += 2
        
        # Genre diversity
        content_diversity = content_metrics.get('average_genre_diversity', 0)
        hybrid_diversity = hybrid_metrics.get('average_genre_diversity', 0)
        if content_diversity > hybrid_diversity:
            scores['Content-Based'] += 1
        else:
            scores['Hybrid'] += 1
        
        # Prediction accuracy
        collab_accuracy = collab_metrics.get('average_prediction_accuracy', 0)
        hybrid_accuracy = hybrid_metrics.get('average_prediction_accuracy', 0)
        if collab_accuracy > hybrid_accuracy:
            scores['Collaborative'] += 1
        else:
            scores['Hybrid'] += 1
        
        # Hybrid gets a bonus for being balanced
        scores['Hybrid'] += 0.5
        
        # Return the model with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

def render_metrics_page():
    """Render the metrics dashboard page"""
    metrics_display = MetricsDisplay()
    metrics_display.display_metrics_dashboard()