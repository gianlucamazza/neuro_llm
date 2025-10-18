"""
Tests for Hybrid LNN + LLM System (Example 03)

NOTE: These tests run WITHOUT API calls (demo mode)
They validate the structure and logic, not LLM integration

Author: Gianluca Mazza
"""

import pytest
import sys
from pathlib import Path

# Import the hybrid system
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "03_hybrid_llm"))
from hybrid_nlu import HybridNLUSystem
from lnn import Fact


class TestHybridNLUSystem:
    """Test suite for hybrid LNN + LLM system"""

    @pytest.fixture
    def system(self):
        """Create system without API key (demo mode)"""
        return HybridNLUSystem(api_key=None)

    def test_system_creation(self, system):
        """Test that system initializes correctly"""
        assert system.lnn_model is not None
        assert system.IsPerson is not None
        assert system.BornIn is not None
        assert system.CitizenOf is not None

    def test_demo_extraction(self, system):
        """Test demo fact extraction (without API)"""
        text = "Leonardo da Vinci nacque a Vinci in Toscana."

        facts = system.extract_facts_with_llm(text)

        # Should return structured dict
        assert 'entities' in facts
        assert 'facts' in facts

        assert 'people' in facts['entities']
        assert 'locations' in facts['entities']

        # Should extract some facts
        assert len(facts['facts']) > 0

    def test_fact_loading_to_lnn(self, system):
        """Test loading extracted facts into LNN"""
        facts_dict = {
            'entities': {
                'people': ['Leonardo'],
                'locations': ['Vinci', 'Italia'],
                'organizations': []
            },
            'facts': [
                {'predicate': 'BornIn', 'args': ['Leonardo', 'Vinci']},
                {'predicate': 'LocatedIn', 'args': ['Vinci', 'Italia']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)

        # Check data was loaded
        person_state = system.lnn_model[system.IsPerson].state()
        assert 'Leonardo' in person_state

        location_state = system.lnn_model[system.IsLocation].state()
        assert 'Vinci' in location_state
        assert 'Italia' in location_state

    def test_citizenship_inference(self, system):
        """Test that citizenship is inferred from birth location"""
        # Add facts
        facts_dict = {
            'entities': {
                'people': ['Mario'],
                'locations': ['Roma'],
                'organizations': []
            },
            'facts': [
                {'predicate': 'BornIn', 'args': ['Mario', 'Roma']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)

        # Run inference
        system.infer()

        # Should infer citizenship
        citizenship_bounds = system.lnn_model[system.CitizenOf].get(('Mario', 'Roma'))

        assert citizenship_bounds is not None, "Should infer citizenship from birth"
        assert citizenship_bounds[0] > 0.5, "Citizenship should have high probability"

    def test_location_transitivity(self, system):
        """Test transitive location inference"""
        facts_dict = {
            'entities': {
                'people': [],
                'locations': ['Vinci', 'Toscana', 'Italia'],
                'organizations': []
            },
            'facts': [
                {'predicate': 'LocatedIn', 'args': ['Vinci', 'Toscana']},
                {'predicate': 'LocatedIn', 'args': ['Toscana', 'Italia']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)
        system.infer()

        # Should infer: Vinci is in Italia (transitive)
        vinci_italia_bounds = system.lnn_model[system.LocatedIn].get(('Vinci', 'Italia'))

        assert vinci_italia_bounds is not None, "Should infer transitive location"
        assert vinci_italia_bounds[0] > 0.5, "Transitive location should have high probability"

    def test_inferred_facts_extraction(self, system):
        """Test extraction of inferred facts"""
        # Setup scenario
        facts_dict = {
            'entities': {
                'people': ['Alice'],
                'locations': ['Paris', 'France'],
                'organizations': []
            },
            'facts': [
                {'predicate': 'BornIn', 'args': ['Alice', 'Paris']},
                {'predicate': 'LocatedIn', 'args': ['Paris', 'France']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)
        system.infer()

        # Extract inferred facts
        inferred = system.extract_inferred_facts()

        # Should have some inferred facts
        assert len(inferred) > 0

        # Check structure
        for fact in inferred:
            assert 'type' in fact
            assert 'confidence' in fact
            assert 'explanation' in fact

    def test_demo_answer_generation(self, system):
        """Test demo answer generation (without API)"""
        context_facts = [
            {
                'type': 'citizenship',
                'person': 'Leonardo',
                'location': 'Italia',
                'confidence': (0.9, 1.0),
                'explanation': 'Leonardo è cittadino di Italia'
            }
        ]

        question = "Di dove è Leonardo?"
        answer = system.answer_question(question, context_facts)

        # Should return some answer
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_residence_citizenship_rule(self, system):
        """Test weaker citizenship inference from residence"""
        facts_dict = {
            'entities': {
                'people': ['Bob'],
                'locations': ['London'],
                'organizations': []
            },
            'facts': [
                {'predicate': 'LivesIn', 'args': ['Bob', 'London']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)
        system.infer()

        # Should infer citizenship but with lower confidence than birth
        citizenship_bounds = system.lnn_model[system.CitizenOf].get(('Bob', 'London'))

        assert citizenship_bounds is not None
        # Confidence might be lower than birth-based citizenship
        assert citizenship_bounds[0] >= 0.3  # Some inference, but not as strong

    def test_multiple_facts_integration(self, system):
        """Test integration of multiple related facts"""
        facts_dict = {
            'entities': {
                'people': ['Anna', 'Bob'],
                'locations': ['Milano', 'Italia'],
                'organizations': ['Google']
            },
            'facts': [
                {'predicate': 'BornIn', 'args': ['Anna', 'Milano']},
                {'predicate': 'LocatedIn', 'args': ['Milano', 'Italia']},
                {'predicate': 'WorksFor', 'args': ['Bob', 'Google']},
            ]
        }

        system.add_facts_to_lnn(facts_dict)
        system.infer()

        # Should handle multiple diverse facts
        anna_citizen = system.lnn_model[system.CitizenOf].get(('Anna', 'Milano'))
        assert anna_citizen is not None

        bob_works = system.lnn_model[system.WorksFor].get(('Bob', 'Google'))
        assert bob_works is not None


def test_hybrid_integration_demo():
    """Integration test: full pipeline in demo mode"""
    system = HybridNLUSystem(api_key=None)

    # Simulate text processing
    text = "Leonardo da Vinci nacque a Vinci in Toscana, Italia."

    # Extract facts (demo mode)
    facts = system.extract_facts_with_llm(text)

    # Should extract something
    assert len(facts['facts']) > 0

    # Load to LNN
    system.add_facts_to_lnn(facts)

    # Infer
    system.infer()

    # Extract inferred
    inferred = system.extract_inferred_facts()

    # Should have inferences
    assert len(inferred) > 0

    # Answer question (demo)
    question = "Di dove è Leonardo?"
    answer = system.answer_question(question, inferred)

    # Should generate answer
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_empty_facts_handling(system):
    """Test that system handles empty facts gracefully"""
    empty_facts = {
        'entities': {
            'people': [],
            'locations': [],
            'organizations': []
        },
        'facts': []
    }

    # Should not crash
    system.add_facts_to_lnn(empty_facts)
    system.infer()

    inferred = system.extract_inferred_facts()
    # Might be empty
    assert isinstance(inferred, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
