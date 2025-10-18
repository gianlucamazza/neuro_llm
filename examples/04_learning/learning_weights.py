"""
Learning con IBM Logical Neural Networks (LNN)

Questo modulo dimostra training end-to-end di sistemi LNN per apprendere
pesi logici da dati con eccezioni.

Per demo interattiva, vedi learning_weights.ipynb
"""

from lnn import (
    Predicate, Variable, Implies, And, Or, Forall,
    Model, Fact, World, Loss, Direction
)
import torch
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


class SocialNetworkLearner:
    """
    Sistema che apprende regole sociali da dati

    Scenario: Vogliamo imparare quanto le seguenti regole sono affidabili:
    1. "Gli amici di solito hanno gusti simili"
    2. "Le persone simili di solito diventano amiche"
    3. "Se A è amico di B e B è amico di C, allora A e C potrebbero essere amici"
    """

    def __init__(self):
        self.model = Model()
        self._setup_predicates()
        self._setup_rules()

    def _setup_predicates(self):
        """Definisce i predicati per il social network"""
        # Predicati binari
        self.Amico = Predicate('Amico', arity=2)
        self.Simile = Predicate('Simile', arity=2)
        self.Interagisce = Predicate('Interagisce', arity=2)

    def _setup_rules(self):
        """
        Definisce regole con World.AXIOM (possono essere violate)
        LNN apprenderà i pesi ottimali
        """
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        # Regola 1: Amici → Simili
        # (Ma ci sono eccezioni! "Gli opposti si attraggono")
        self.model.add_knowledge(
            Forall(
                x, y,
                Implies(
                    self.Amico(x, y),
                    self.Simile(x, y)
                ),
                world=World.AXIOM  # Può essere violata
            ),
        )

        # Regola 2: Simili → Amici
        # (Non sempre! Possono essere simili ma non conoscersi)
        self.model.add_knowledge(
            Forall(
                x, y,
                Implies(
                    And(
                        self.Simile(x, y),
                        self.Interagisce(x, y)  # Devono anche interagire
                    ),
                    self.Amico(x, y)
                ),
                world=World.AXIOM
            ),
        )

        # Regola 3: Transitività amicizia (debole)
        # "L'amico del mio amico è mio amico" (non sempre vero!)
        self.model.add_knowledge(
            Forall(
                x, y, z,
                Implies(
                    And(
                        self.Amico(x, y),
                        self.Amico(y, z)
                    ),
                    self.Amico(x, z)
                ),
                world=World.AXIOM
            ),
        )

    def add_training_data(self, data: Dict):
        """
        Aggiunge dati di training

        Args:
            data: Dizionario {predicate: {args: value}}
        """
        self.model.add_data(data)

    def train(self, epochs: int = 100, learning_rate: float = 0.01) -> List[float]:
        """
        Training per apprendere pesi ottimali

        NOTA: Il training end-to-end con gradient descent su LNN richiede
        una versione specifica dell'API IBM LNN che supporta il calcolo
        della loss function. La versione attuale di questo progetto usa
        LNN principalmente per inferenza logica deterministica.

        Per simulazione educativa, questo metodo esegue solo inferenza
        senza training dei pesi.

        Args:
            epochs: Numero di epoche (non usato nella simulazione)
            learning_rate: Learning rate (non usato nella simulazione)

        Returns:
            Lista vuota (placeholder per compatibilità)
        """
        print(f"\n{'='*60}")
        print("SOCIAL NETWORK LEARNING SYSTEM - INFERENCE MODE")
        print(f"{'='*60}")
        print()
        print("⚠️  NOTA: Il training end-to-end con gradient descent richiede")
        print("   una versione specifica dell'API LNN non disponibile.")
        print()
        print("✓ Eseguendo inferenza logica sui dati caricati...")
        print()

        # Esegui inferenza LNN sui dati
        self.model.infer(direction=Direction.UPWARD)

        print("✓ Inferenza completata!")
        print()
        print("Il sistema ha applicato le regole logiche ai dati:")
        print("  1. Amici → Simili (con eccezioni)")
        print("  2. Simili + Interagiscono → Amici")
        print("  3. Transitività amicizia (debole)")
        print()
        print("Per predizioni, usa il metodo predict(person1, person2, predicate)")
        print(f"{'='*60}")

        return []  # Lista vuota per compatibilità

    def evaluate_rule_strength(self) -> Dict[str, float]:
        """
        Valuta quanto ogni regola è "affidabile" dopo training

        Returns:
            Dizionario {nome_regola: forza}
        """
        strengths = {}

        # Estrai pesi appresi per ogni regola
        for name, formula in self.model.formulae.items():
            # I pesi sono nei neuroni
            if hasattr(formula, 'weights'):
                # Media dei pesi come "forza" della regola
                weight_values = [w.item() for w in formula.weights.values()]
                if weight_values:
                    strengths[name] = sum(weight_values) / len(weight_values)

        return strengths

    def predict(self, person1: str, person2: str, predicate_name: str) -> Tuple[float, float]:
        """
        Predice relazione tra due persone

        Args:
            person1, person2: Nomi persone
            predicate_name: Nome predicato ('Amico', 'Simile', etc)

        Returns:
            Bounds [lower, upper]
        """
        predicate = getattr(self, predicate_name)
        state = predicate.state()
        bounds = state.get((person1, person2))

        if bounds:
            return bounds
        else:
            return (0.0, 0.0)


def generate_synthetic_data() -> Dict:
    """
    Genera dataset sintetico per social network

    Include:
    - Casi "normali" dove regole sono rispettate
    - Eccezioni dove regole sono violate
    """
    # Crea Predicate objects UNA VOLTA SOLA
    # Ogni Predicate('Amico', arity=2) crea un NUOVO oggetto con ID diverso
    # Dobbiamo riusare gli STESSI oggetti come chiavi del dizionario
    Amico = Predicate('Amico', arity=2)
    Simile = Predicate('Simile', arity=2)
    Interagisce = Predicate('Interagisce', arity=2)

    data = {
        Amico: {},
        Simile: {},
        Interagisce: {},
    }

    # Cluster 1: Tech enthusiasts
    # Alice, Bob, Charlie - amici e simili (regola rispettata)
    tech_people = ['Alice', 'Bob', 'Charlie']
    for i, p1 in enumerate(tech_people):
        for p2 in tech_people[i+1:]:
            data[Amico][(p1, p2)] = Fact.TRUE
            data[Amico][(p2, p1)] = Fact.TRUE
            data[Simile][(p1, p2)] = Fact.TRUE
            data[Simile][(p2, p1)] = Fact.TRUE
            data[Interagisce][(p1, p2)] = Fact.TRUE
            data[Interagisce][(p2, p1)] = Fact.TRUE

    # Cluster 2: Artists
    artists = ['Diana', 'Eve']
    data[Amico][('Diana', 'Eve')] = Fact.TRUE
    data[Amico][('Eve', 'Diana')] = Fact.TRUE
    data[Simile][('Diana', 'Eve')] = Fact.TRUE
    data[Simile][('Eve', 'Diana')] = Fact.TRUE
    data[Interagisce][('Diana', 'Eve')] = Fact.TRUE
    data[Interagisce][('Eve', 'Diana')] = Fact.TRUE

    # ECCEZIONE 1: Frank e George sono amici ma NON simili
    # "Gli opposti si attraggono"
    data[Amico][('Frank', 'George')] = Fact.TRUE
    data[Amico][('George', 'Frank')] = Fact.TRUE
    data[Simile][('Frank', 'George')] = (0.1, 0.3)  # Poco simili
    data[Simile][('George', 'Frank')] = (0.1, 0.3)
    data[Interagisce][('Frank', 'George')] = Fact.TRUE
    data[Interagisce][('George', 'Frank')] = Fact.TRUE

    # ECCEZIONE 2: Helen e Igor sono simili ma NON amici
    # (Simili ma non si conoscono)
    data[Simile][('Helen', 'Igor')] = Fact.TRUE
    data[Simile][('Igor', 'Helen')] = Fact.TRUE
    data[Interagisce][('Helen', 'Igor')] = Fact.FALSE
    data[Interagisce][('Igor', 'Helen')] = Fact.FALSE
    data[Amico][('Helen', 'Igor')] = Fact.FALSE
    data[Amico][('Igor', 'Helen')] = Fact.FALSE

    # ECCEZIONE 3: Transitività non sempre vera
    # Jack → Kevin → Lisa, ma Jack NON è amico di Lisa
    data[Amico][('Jack', 'Kevin')] = Fact.TRUE
    data[Amico][('Kevin', 'Lisa')] = Fact.TRUE
    data[Amico][('Jack', 'Lisa')] = (0.2, 0.4)  # Conoscenti, non amici
    data[Interagisce][('Jack', 'Kevin')] = Fact.TRUE
    data[Interagisce][('Kevin', 'Lisa')] = Fact.TRUE
    data[Interagisce][('Jack', 'Lisa')] = Fact.TRUE

    return data


def plot_learning_curve(losses: List[float], output_path: str = "learning_curve.png"):
    """Plotta curva di apprendimento"""
    if not losses:
        print("\n⚠️  Nessun dato di loss disponibile per il grafico.")
        print("   Il training end-to-end non è attualmente implementato.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Logical Contradiction)', fontsize=12)
    plt.title('LNN Training: Loss Reduction', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nGrafico salvato: {output_path}")


# Verifica modulo quando eseguito direttamente
if __name__ == "__main__":
    print("=" * 70)
    print("LNN Learning System - Module Check")
    print("=" * 70)

    # Test creazione sistema
    learner = SocialNetworkLearner()
    print("\n✓ SocialNetworkLearner creato correttamente")
    print("✓ Metodi di training disponibili")

    print("\n" + "=" * 70)
    print("Per la demo completa con training e dati, usa il notebook:")
    print("  learning_weights.ipynb")
    print("=" * 70)
