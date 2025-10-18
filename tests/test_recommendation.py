"""
Tests for Movie Recommendation System (Example 02)

Validates:
- Recommendation generation
- Preference inference
- Movie similarity handling
- Genre-based recommendations

Author: Gianluca Mazza
"""

import pytest
import sys
from pathlib import Path

# Import the recommendation system
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "02_recommendation"))
from movie_recommender import MovieRecommendationSystem
from lnn import Fact


class TestMovieRecommendationSystem:
    """Test suite for movie recommendation system"""

    @pytest.fixture
    def system(self):
        """Create a fresh recommendation system with sample data"""
        sys = MovieRecommendationSystem()

        # Add movies
        sys.add_movie('Inception', ['SciFi', 'Thriller'])
        sys.add_movie('Interstellar', ['SciFi', 'Drama'])
        sys.add_movie('The Matrix', ['SciFi', 'Action'])
        sys.add_movie('Tenet', ['SciFi', 'Thriller'])

        sys.add_movie('Titanic', ['Romance', 'Drama'])
        sys.add_movie('The Notebook', ['Romance', 'Drama'])
        sys.add_movie('La La Land', ['Romance', 'Musical'])

        # Add similarities
        sys.add_movie_similarity('Inception', 'Tenet', 0.9)
        sys.add_movie_similarity('Inception', 'Interstellar', 0.7)
        sys.add_movie_similarity('Titanic', 'The Notebook', 0.85)

        return sys

    def test_system_creation(self, system):
        """Test that system initializes correctly"""
        assert system.model is not None
        assert len(system.movies) > 0
        assert len(system.genres) > 0

    def test_explicit_preference_recommendation(self, system, tolerance):
        """Test recommendations based on explicit genre preference"""
        # User explicitly likes SciFi
        system.add_user_preference('Alice', 'SciFi', strength=1.0)

        recommendations = system.get_recommendations('Alice', top_n=3)

        # Should recommend SciFi movies
        assert len(recommendations) > 0

        scifi_movies = {'Inception', 'Interstellar', 'The Matrix', 'Tenet'}
        for movie, score in recommendations:
            assert movie in scifi_movies, f"Expected SciFi movie, got {movie}"
            assert score > 0.3, f"Score too low: {score}"

    def test_viewing_history_inference(self, system, tolerance):
        """Test that preferences are inferred from viewing history"""
        # User watched 2 SciFi movies → should infer SciFi preference
        system.add_viewing_history('Bob', ['Inception', 'Interstellar'])

        # Get recommendations (this triggers preference inference)
        recommendations = system.get_recommendations('Bob', top_n=5)

        # Should recommend other SciFi movies
        recommended_movies = [movie for movie, score in recommendations]

        # Should include other SciFi movies not yet watched
        assert 'The Matrix' in recommended_movies or 'Tenet' in recommended_movies

    def test_similarity_based_recommendation(self, system):
        """Test recommendations based on movie similarity"""
        # User watched Inception
        system.add_viewing_history('Charlie', ['Inception'])

        # Manually add SciFi preference to ensure recommendation
        system.add_user_preference('Charlie', 'SciFi', strength=0.8)

        recommendations = system.get_recommendations('Charlie', top_n=5)
        recommended_movies = [movie for movie, score in recommendations]

        # Tenet should be recommended (highly similar to Inception)
        assert 'Tenet' in recommended_movies, "Tenet should be recommended (similar to Inception)"

    def test_multiple_genre_preference(self, system):
        """Test user with multiple genre preferences"""
        system.add_user_preference('Diana', 'SciFi', strength=0.8)
        system.add_user_preference('Diana', 'Romance', strength=0.6)

        recommendations = system.get_recommendations('Diana', top_n=10)

        # Should get mix of SciFi and Romance
        recommended_movies = [movie for movie, score in recommendations]

        has_scifi = any(m in {'Inception', 'Interstellar', 'The Matrix', 'Tenet'}
                       for m in recommended_movies)
        has_romance = any(m in {'Titanic', 'The Notebook', 'La La Land'}
                         for m in recommended_movies)

        assert has_scifi or has_romance, "Should recommend from preferred genres"

    def test_no_duplicate_recommendations(self, system):
        """Test that already-watched movies are not recommended"""
        watched_movies = ['Inception', 'Interstellar']
        system.add_viewing_history('Eve', watched_movies)
        system.add_user_preference('Eve', 'SciFi', strength=1.0)

        recommendations = system.get_recommendations('Eve', top_n=10)
        recommended_movies = [movie for movie, score in recommendations]

        # Should NOT recommend already-watched movies
        for watched in watched_movies:
            assert watched not in recommended_movies, f"{watched} should not be recommended (already watched)"

    def test_cold_start_user(self, system):
        """Test recommendation for new user with no history"""
        # User with only explicit preference
        system.add_user_preference('NewUser', 'Romance', strength=1.0)

        recommendations = system.get_recommendations('NewUser', top_n=3)

        # Should still get recommendations
        assert len(recommendations) > 0

        romance_movies = {'Titanic', 'The Notebook', 'La La Land'}
        for movie, score in recommendations:
            assert movie in romance_movies

    def test_recommendation_scores_ordered(self, system):
        """Test that recommendations are ordered by score"""
        system.add_user_preference('Frank', 'SciFi', strength=1.0)

        recommendations = system.get_recommendations('Frank', top_n=5)

        # Scores should be in descending order
        scores = [score for movie, score in recommendations]
        assert scores == sorted(scores, reverse=True), "Recommendations should be ordered by score"

    def test_explanation_generation(self, system):
        """Test that explanations are generated for recommendations"""
        system.add_user_preference('George', 'SciFi', strength=1.0)
        system.add_viewing_history('George', ['Inception'])

        # Get recommendation
        recommendations = system.get_recommendations('George', top_n=1)

        if recommendations:
            movie, score = recommendations[0]

            # Get explanation
            explanation = system.explain_recommendation('George', movie)

            # Should have some explanation
            assert len(explanation) > 0
            assert isinstance(explanation, str)

    def test_preference_strength_propagation(self, system, high_tolerance):
        """Test that preference strength affects recommendation scores"""
        # Strong preference
        system.add_user_preference('Strong', 'SciFi', strength=1.0)

        # Weak preference
        system.add_user_preference('Weak', 'SciFi', strength=0.5)

        strong_recs = system.get_recommendations('Strong', top_n=5)
        weak_recs = system.get_recommendations('Weak', top_n=5)

        # Strong preference should generally have higher scores
        if strong_recs and weak_recs:
            strong_avg = sum(score for _, score in strong_recs) / len(strong_recs)
            weak_avg = sum(score for _, score in weak_recs) / len(weak_recs)

            # Allow for some tolerance
            assert strong_avg >= weak_avg - high_tolerance, \
                "Stronger preference should yield higher average scores"


def test_recommendation_integration():
    """Integration test: full recommendation workflow"""
    system = MovieRecommendationSystem()

    # Build catalog
    system.add_movie('Inception', ['SciFi', 'Thriller'])
    system.add_movie('Interstellar', ['SciFi', 'Drama'])
    system.add_movie('Tenet', ['SciFi', 'Thriller'])
    system.add_movie('Titanic', ['Romance', 'Drama'])

    system.add_movie_similarity('Inception', 'Tenet', 0.9)

    # User behavior
    system.add_viewing_history('TestUser', ['Inception', 'Interstellar'])

    # Get recommendations
    recommendations = system.get_recommendations('TestUser', top_n=3)

    # Should have recommendations
    assert len(recommendations) > 0

    # Should not include already-watched movies
    recommended_movies = [m for m, s in recommendations]
    assert 'Inception' not in recommended_movies
    assert 'Interstellar' not in recommended_movies

    # Should include similar/same-genre movies
    assert 'Tenet' in recommended_movies or 'Titanic' in recommended_movies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
