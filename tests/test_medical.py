"""
Tests for Medical Diagnosis System (Example 01)

Validates:
- Correct diagnosis inference
- Bounds computation
- Rule application
- Multiple patients handling

Author: Gianluca Mazza
"""

import pytest
import sys
from pathlib import Path

# Import the medical system
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "01_medical_expert"))
from medical_diagnosis import MedicalDiagnosisSystem
from lnn import Fact


class TestMedicalDiagnosisSystem:
    """Test suite for medical diagnosis system"""

    @pytest.fixture
    def system(self):
        """Create a fresh medical diagnosis system"""
        return MedicalDiagnosisSystem()

    def test_system_creation(self, system):
        """Test that system initializes correctly"""
        assert system.model is not None
        assert system.Ha_Febbre is not None
        assert system.Influenza is not None

    def test_clear_influenza_case(self, system, tolerance):
        """Test diagnosis of clear influenza case"""
        # Patient with fever and muscle pain → should have high influenza probability
        system.add_patient('TestPatient', {
            'febbre': Fact.TRUE,
            'dolori_muscolari': Fact.TRUE,
        })

        results = system.diagnose()

        assert 'TestPatient' in results
        diagnosis = results['TestPatient']

        # Check influenza is diagnosed
        assert 'Influenza' in diagnosis
        influenza_bounds = diagnosis['Influenza']

        # Should have high probability
        assert influenza_bounds[0] >= 0.7, f"Expected influenza lower bound >= 0.7, got {influenza_bounds[0]}"

    def test_clear_cold_case(self, system, tolerance):
        """Test diagnosis of clear cold case"""
        # Patient with cough, sore throat, NO fever → cold
        system.add_patient('ColdPatient', {
            'tosse': Fact.TRUE,
            'mal_di_gola': Fact.TRUE,
            'febbre': Fact.FALSE,
        })

        results = system.diagnose()
        diagnosis = results['ColdPatient']

        # Check cold is diagnosed
        assert 'Raffreddore' in diagnosis
        cold_bounds = diagnosis['Raffreddore']

        # Should have reasonable probability
        assert cold_bounds[0] >= 0.5, f"Expected cold lower bound >= 0.5, got {cold_bounds[0]}"

    def test_covid_symptoms(self, system, tolerance):
        """Test COVID-19 symptom pattern"""
        system.add_patient('CovidPatient', {
            'febbre': Fact.TRUE,
            'tosse': Fact.TRUE,
            'dolori_muscolari': Fact.TRUE,
        })

        results = system.diagnose()
        diagnosis = results['CovidPatient']

        # Should diagnose both Covid and Influenza (overlapping symptoms)
        assert 'Covid' in diagnosis
        assert 'Influenza' in diagnosis

        # Both should have reasonable probability
        covid_bounds = diagnosis['Covid']
        assert covid_bounds[0] >= 0.5

    def test_uncertainty_handling(self, system):
        """Test that system handles uncertain symptoms"""
        # Moderate symptoms with uncertainty
        system.add_patient('UncertainPatient', {
            'febbre': [0.6, 0.8],  # Uncertain fever
            'tosse': [0.5, 0.7],   # Uncertain cough
        })

        results = system.diagnose()

        # Should still produce diagnosis
        assert 'UncertainPatient' in results
        diagnosis = results['UncertainPatient']

        # Should have diagnoses with bounds reflecting uncertainty
        for disease, bounds in diagnosis.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert 0 <= bounds[0] <= bounds[1] <= 1

    def test_multiple_patients(self, system):
        """Test system handles multiple patients correctly"""
        system.add_patient('Patient1', {
            'febbre': Fact.TRUE,
            'dolori_muscolari': Fact.TRUE,
        })

        system.add_patient('Patient2', {
            'tosse': Fact.TRUE,
            'mal_di_gola': Fact.TRUE,
            'febbre': Fact.FALSE,
        })

        results = system.diagnose()

        # Both patients should have diagnoses
        assert 'Patient1' in results
        assert 'Patient2' in results

        # Patient 1 should have influenza
        assert results['Patient1']['Influenza'][0] >= 0.5

        # Patient 2 should have cold
        assert results['Patient2']['Raffreddore'][0] >= 0.5

    def test_no_symptoms(self, system):
        """Test patient with no significant symptoms"""
        system.add_patient('HealthyPatient', {
            'febbre': Fact.FALSE,
            'tosse': Fact.FALSE,
            'dolori_muscolari': Fact.FALSE,
        })

        results = system.diagnose()
        diagnosis = results['HealthyPatient']

        # All diagnoses should have low probability
        for disease, bounds in diagnosis.items():
            assert bounds[1] < 0.5, f"{disease} should have low probability for healthy patient"

    def test_bronchitis_pattern(self, system, tolerance):
        """Test bronchitis symptom pattern"""
        system.add_patient('BronchitisPatient', {
            'tosse': Fact.TRUE,
            'respiro_corto': Fact.TRUE,
            'dolori_muscolari': Fact.TRUE,
        })

        results = system.diagnose()
        diagnosis = results['BronchitisPatient']

        # Should diagnose bronchitis
        assert 'Bronchite' in diagnosis
        bronchitis_bounds = diagnosis['Bronchite']
        assert bronchitis_bounds[0] >= 0.5


def test_system_integration():
    """Integration test: full workflow"""
    system = MedicalDiagnosisSystem()

    # Add multiple patients with different conditions
    patients_data = {
        'Influenza_Case': {
            'febbre': Fact.TRUE,
            'dolori_muscolari': Fact.TRUE,
        },
        'Cold_Case': {
            'tosse': Fact.TRUE,
            'mal_di_gola': Fact.TRUE,
            'febbre': Fact.FALSE,
        },
        'Covid_Case': {
            'febbre': Fact.TRUE,
            'tosse': Fact.TRUE,
            'congestione': Fact.TRUE,
        }
    }

    for patient, symptoms in patients_data.items():
        system.add_patient(patient, symptoms)

    # Run diagnosis
    results = system.diagnose()

    # Verify all patients have results
    assert len(results) == 3

    # Verify specific diagnoses
    assert results['Influenza_Case']['Influenza'][0] >= 0.7
    assert results['Cold_Case']['Raffreddore'][0] >= 0.5
    assert results['Covid_Case']['Covid'][0] >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
