"""
Sistema Esperto per Diagnosi Medica usando IBM Logical Neural Networks (LNN)

Questo modulo fornisce la classe MedicalDiagnosisSystem che implementa un sistema
esperto per diagnosi medica basato su regole logiche.

Caratteristiche:
- Definisce regole logiche per inferire malattie da sintomi
- Gestisce incertezza con bounds (lower, upper)
- Esegue inferenza bidirezionale

Utilizzo:
    from medical_diagnosis import MedicalDiagnosisSystem

    system = MedicalDiagnosisSystem()
    system.add_patient('Mario', {'febbre': Fact.TRUE, ...})
    results = system.diagnose()

Per demo interattiva completa, vedi il notebook medical_diagnosis.ipynb

Autore: Gianluca Mazza
"""

from lnn import Predicate, Variable, Implies, And, Or, Not, Model, Fact, World


class MedicalDiagnosisSystem:
    """Sistema esperto per diagnosi medica basato su regole logiche"""

    def __init__(self):
        self.model = Model()
        self.patient_var = Variable('paziente')
        self._setup_predicates()
        self._setup_rules()

    def _setup_predicates(self):
        """Definisce i predicati (neuroni logici) per sintomi e diagnosi"""
        # Sintomi
        self.Ha_Febbre = Predicate('Ha_Febbre')
        self.Ha_Tosse = Predicate('Ha_Tosse')
        self.Ha_Mal_Di_Gola = Predicate('Ha_Mal_Di_Gola')
        self.Ha_Dolori_Muscolari = Predicate('Ha_Dolori_Muscolari')
        self.Ha_Congestione = Predicate('Ha_Congestione')
        self.Ha_Respiro_Corto = Predicate('Ha_Respiro_Corto')

        # Diagnosi
        self.Influenza = Predicate('Influenza')
        self.Raffreddore = Predicate('Raffreddore')
        self.Covid = Predicate('Covid')
        self.Bronchite = Predicate('Bronchite')

    def _setup_rules(self):
        """Definisce le regole logiche per la diagnosi"""
        x = self.patient_var

        # Regola 1: Febbre ∧ Dolori Muscolari → Influenza
        self.model.add_knowledge(
            Implies(
                And(self.Ha_Febbre(x), self.Ha_Dolori_Muscolari(x)),
                self.Influenza(x)
            )
        )

        # Regola 2: Tosse ∧ Mal_Di_Gola ∧ ¬Febbre → Raffreddore
        self.model.add_knowledge(
            Implies(
                And(
                    self.Ha_Tosse(x),
                    self.Ha_Mal_Di_Gola(x),
                    Not(self.Ha_Febbre(x))
                ),
                self.Raffreddore(x)
            )
        )

        # Regola 3: Febbre ∧ Tosse ∧ (Dolori ∨ Congestione) → Covid
        self.model.add_knowledge(
            Implies(
                And(
                    self.Ha_Febbre(x),
                    self.Ha_Tosse(x),
                    Or(self.Ha_Dolori_Muscolari(x), self.Ha_Congestione(x))
                ),
                self.Covid(x)
            )
        )

        # Regola 4: Tosse ∧ Respiro_Corto ∧ Dolori_Muscolari → Bronchite
        self.model.add_knowledge(
            Implies(
                And(
                    self.Ha_Tosse(x),
                    self.Ha_Respiro_Corto(x),
                    self.Ha_Dolori_Muscolari(x)
                ),
                self.Bronchite(x)
            )
        )

    def add_patient(self, name: str, symptoms: dict):
        """
        Aggiunge un paziente con i suoi sintomi

        Args:
            name: Nome del paziente
            symptoms: Dizionario {sintomo: valore}
                     valore può essere:
                     - Fact.TRUE / Fact.FALSE: certezza
                     - [lower, upper]: bounds per incertezza (es. [0.7, 0.9])
        """
        data = {}

        # Mappa sintomi a predicati
        symptom_map = {
            'febbre': self.Ha_Febbre,
            'tosse': self.Ha_Tosse,
            'mal_di_gola': self.Ha_Mal_Di_Gola,
            'dolori_muscolari': self.Ha_Dolori_Muscolari,
            'congestione': self.Ha_Congestione,
            'respiro_corto': self.Ha_Respiro_Corto,
        }

        for symptom, value in symptoms.items():
            if symptom in symptom_map:
                predicate = symptom_map[symptom]
                if predicate not in data:
                    data[predicate] = {}
                data[predicate][name] = value

        self.model.add_data(data)

    def diagnose(self) -> dict:
        """
        Esegue l'inferenza e restituisce le diagnosi

        Returns:
            Dizionario {paziente: {diagnosi: bounds}}
        """
        # Inferenza bidirezionale
        self.model.infer()

        # Estrai risultati
        results = {}

        # Predicati diagnosi da controllare
        diagnoses = {
            'Influenza': self.Influenza,
            'Raffreddore': self.Raffreddore,
            'Covid': self.Covid,
            'Bronchite': self.Bronchite,
        }

        # Per ogni paziente nel modello
        all_patients = set()
        for predicate in [self.Ha_Febbre, self.Ha_Tosse, self.Ha_Mal_Di_Gola]:
            all_patients.update(predicate.state())

        for patient in all_patients:
            if patient not in results:
                results[patient] = {}

            for diag_name, diag_predicate in diagnoses.items():
                state = diag_predicate.state()
                if patient in state:
                    bounds = state[patient]
                    results[patient][diag_name] = bounds

        return results

    def print_diagnosis(self, patient_name: str, diagnosis: dict):
        """Stampa la diagnosi in formato leggibile"""
        print(f"\n{'='*60}")
        print(f"DIAGNOSI PER: {patient_name}")
        print(f"{'='*60}")

        # Ordina per probabilità (lower bound)
        sorted_diag = sorted(
            diagnosis.items(),
            key=lambda x: x[1][0] if isinstance(x[1], tuple) else 0,
            reverse=True
        )

        for disease, bounds in sorted_diag:
            if isinstance(bounds, tuple):
                lower, upper = bounds
                confidence = (lower + upper) / 2 * 100

                # Classificazione
                if lower >= 0.7:
                    status = "PROBABILE"
                    symbol = "⚠️"
                elif lower >= 0.4:
                    status = "POSSIBILE"
                    symbol = "❓"
                else:
                    status = "IMPROBABILE"
                    symbol = "✓"

                print(f"{symbol} {disease:15s}: {status:12s} "
                      f"[{lower:.2f}, {upper:.2f}] ~{confidence:.1f}%")


# Verifica modulo quando eseguito direttamente
if __name__ == "__main__":
    print("=" * 60)
    print("Medical Diagnosis System - Module Check")
    print("=" * 60)

    # Test creazione sistema
    system = MedicalDiagnosisSystem()
    print("\n✓ MedicalDiagnosisSystem creato correttamente")

    # Quick test
    system.add_patient('Test', {'febbre': Fact.TRUE})
    results = system.diagnose()
    print("✓ Inferenza funzionante")

    print("\n" + "=" * 60)
    print("Per la demo completa, usa il notebook:")
    print("  medical_diagnosis.ipynb")
    print("=" * 60)
