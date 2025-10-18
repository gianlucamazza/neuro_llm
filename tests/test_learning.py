"""
Tests for Learning with LNN (Example 04)

Validates:
- Training convergence
- Loss reduction
- Weight learning
- Prediction after training

Author: Gianluca Mazza
"""

import pytest
import sys
from pathlib import Path

# Import the learning system
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "04_learning"))
from learning_weights import SocialNetworkLearner, generate_synthetic_data
from lnn import Fact, Predicate


class TestSocialNetworkLearner:
    """Test suite for LNN learning system"""

    @pytest.fixture
    def learner(self):
        """Create a fresh learner"""
        return SocialNetworkLearner()

    @pytest.fixture
    def training_data(self):
        """Generate training data"""
        return generate_synthetic_data()

    def test_learner_creation(self, learner):
        """Test that learner initializes correctly"""
        assert learner.model is not None
        assert learner.Amico is not None
        assert learner.Simile is not None

    def test_data_loading(self, learner, training_data):
        """Test that training data loads correctly"""
        learner.add_training_data(training_data)

        # Check that data was added
        # Should have some friendships
        friendship_data = learner.model[learner.Amico].state()
        assert len(friendship_data) > 0

    def test_training_convergence(self, learner, training_data):
        """Test that training reduces loss"""
        learner.add_training_data(training_data)

        # Train for few epochs
        losses = learner.train(epochs=20, learning_rate=0.01)

        # Should have loss for each epoch
        assert len(losses) == 20

        # Loss should generally decrease
        initial_loss = losses[0]
        final_loss = losses[-1]

        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_loss_reduction_significant(self, learner, training_data):
        """Test that loss reduction is significant"""
        learner.add_training_data(training_data)

        losses = learner.train(epochs=50, learning_rate=0.01)

        initial_loss = losses[0]
        final_loss = losses[-1]

        reduction_pct = (1 - final_loss / initial_loss) * 100

        # Should reduce loss by at least 20%
        assert reduction_pct >= 20, \
            f"Loss should reduce by at least 20%, got {reduction_pct:.1f}%"

    def test_prediction_bounds_valid(self, learner, training_data):
        """Test that predictions have valid bounds"""
        learner.add_training_data(training_data)
        learner.train(epochs=30, learning_rate=0.01)

        # Make predictions
        bounds = learner.predict('Alice', 'Bob', 'Simile')

        # Bounds should be valid
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2

        lower, upper = bounds
        assert 0 <= lower <= 1, f"Lower bound should be in [0,1], got {lower}"
        assert 0 <= upper <= 1, f"Upper bound should be in [0,1], got {upper}"
        assert lower <= upper, f"Lower should be <= upper, got [{lower}, {upper}]"

    def test_exception_learning(self, learner, training_data):
        """Test that system learns from exceptions in data"""
        learner.add_training_data(training_data)
        learner.train(epochs=50, learning_rate=0.01)

        # Frank and George are friends but NOT similar (exception)
        # After training, prediction should reflect this
        simile_bounds = learner.predict('Frank', 'George', 'Simile')

        # Should have low similarity (learning from exception)
        avg_similarity = (simile_bounds[0] + simile_bounds[1]) / 2

        # This is an exception case, so similarity should be relatively low
        assert avg_similarity < 0.7, \
            f"Should learn that Frank-George are not very similar (exception case)"

    def test_normal_case_learning(self, learner, training_data):
        """Test that normal cases are learned correctly"""
        learner.add_training_data(training_data)
        learner.train(epochs=50, learning_rate=0.01)

        # Alice and Bob are friends AND similar (normal case)
        simile_bounds = learner.predict('Alice', 'Bob', 'Simile')

        avg_similarity = (simile_bounds[0] + simile_bounds[1]) / 2

        # Should have high similarity
        assert avg_similarity > 0.5, \
            f"Alice-Bob should be similar (normal case), got {avg_similarity:.2f}"

    def test_multiple_training_runs(self, learner, training_data):
        """Test that multiple training runs continue to improve"""
        learner.add_training_data(training_data)

        # First training
        losses1 = learner.train(epochs=20, learning_rate=0.01)

        # Second training (continue)
        losses2 = learner.train(epochs=20, learning_rate=0.01)

        # Both should complete successfully
        assert len(losses1) == 20
        assert len(losses2) == 20

        # Final loss should be reasonable
        assert losses2[-1] < losses1[0]


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    data = generate_synthetic_data()

    # Should have data for all predicates
    assert Predicate('Amico', arity=2) in data
    assert Predicate('Simile', arity=2) in data
    assert Predicate('Interagisce', arity=2) in data

    # Should have multiple entries
    amico_data = data[Predicate('Amico', arity=2)]
    assert len(amico_data) > 5, "Should have multiple friendships"

    # Should include both TRUE and bounds (for exceptions)
    has_true = any(v == Fact.TRUE for v in amico_data.values())
    has_bounds = any(isinstance(v, list) for v in amico_data.values())

    assert has_true or has_bounds, "Should have fact values"


def test_learning_integration():
    """Integration test: full learning workflow"""
    # Create system
    learner = SocialNetworkLearner()

    # Generate data
    data = generate_synthetic_data()

    # Add data
    learner.add_training_data(data)

    # Train
    losses = learner.train(epochs=50, learning_rate=0.01)

    # Verify training
    assert len(losses) == 50
    assert losses[-1] < losses[0]

    # Make predictions
    test_cases = [
        ('Alice', 'Bob', 'Simile'),
        ('Frank', 'George', 'Simile'),
        ('Alice', 'Charlie', 'Amico'),
    ]

    for p1, p2, pred in test_cases:
        bounds = learner.predict(p1, p2, pred)

        # All predictions should be valid
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert 0 <= bounds[0] <= bounds[1] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
