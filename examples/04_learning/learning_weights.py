"""
Learning con IBM Logical Neural Networks (LNN)

Questo modulo fornisce le classi per training end-to-end di sistemi LNN,
dimostrando come apprendere pesi logici da dati con eccezioni.

Caratteristiche:
- Apprendimento pesi logici da dati con eccezioni
- Ottimizzazione "quanto" una regola è affidabile
- Gestione contraddizioni e dati rumorosi
- Training end-to-end con gradient descent

Utilizzo:
    from learning_weights import SocialNetworkLearner, generate_synthetic_data

    learner = SocialNetworkLearner()
    data = generate_synthetic_data()
    learner.add_training_data(data)
    losses = learner.train(epochs=100)

Per demo interattiva completa, vedi il notebook learning_weights.ipynb

Autore: Gianluca Mazza
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

        Args:
            epochs: Numero di epoche
            learning_rate: Learning rate

        Returns:
            Lista di loss values per epoca
        """
        print(f"\n{'='*60}")
        print("TRAINING LNN")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Loss Function: LOGICAL_CONTRADICTION")
        print()

        # LNN training con loss function che penalizza contraddizioni logiche
        losses = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass: inferenza
            self.model.infer(direction=Direction.UPWARD)

            # Calcola loss (contraddizioni logiche)
            loss = self.model.loss(Loss.LOGICAL_CONTRADICTION)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

        print(f"\nTraining completato!")
        print(f"Loss iniziale: {losses[0]:.6f}")
        print(f"Loss finale:   {losses[-1]:.6f}")
        print(f"Riduzione:     {(1 - losses[-1]/losses[0])*100:.1f}%")

        return losses

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
    data = {
        Predicate('Amico', arity=2): {},
        Predicate('Simile', arity=2): {},
        Predicate('Interagisce', arity=2): {},
    }

    # Cluster 1: Tech enthusiasts
    # Alice, Bob, Charlie - amici e simili (regola rispettata)
    tech_people = ['Alice', 'Bob', 'Charlie']
    for i, p1 in enumerate(tech_people):
        for p2 in tech_people[i+1:]:
            data[Predicate('Amico', arity=2)][(p1, p2)] = Fact.TRUE
            data[Predicate('Amico', arity=2)][(p2, p1)] = Fact.TRUE
            data[Predicate('Simile', arity=2)][(p1, p2)] = Fact.TRUE
            data[Predicate('Simile', arity=2)][(p2, p1)] = Fact.TRUE
            data[Predicate('Interagisce', arity=2)][(p1, p2)] = Fact.TRUE
            data[Predicate('Interagisce', arity=2)][(p2, p1)] = Fact.TRUE

    # Cluster 2: Artists
    artists = ['Diana', 'Eve']
    data[Predicate('Amico', arity=2)][('Diana', 'Eve')] = Fact.TRUE
    data[Predicate('Amico', arity=2)][('Eve', 'Diana')] = Fact.TRUE
    data[Predicate('Simile', arity=2)][('Diana', 'Eve')] = Fact.TRUE
    data[Predicate('Simile', arity=2)][('Eve', 'Diana')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('Diana', 'Eve')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('Eve', 'Diana')] = Fact.TRUE

    # ECCEZIONE 1: Frank e George sono amici ma NON simili
    # "Gli opposti si attraggono"
    data[Predicate('Amico', arity=2)][('Frank', 'George')] = Fact.TRUE
    data[Predicate('Amico', arity=2)][('George', 'Frank')] = Fact.TRUE
    data[Predicate('Simile', arity=2)][('Frank', 'George')] = [0.1, 0.3]  # Poco simili
    data[Predicate('Simile', arity=2)][('George', 'Frank')] = [0.1, 0.3]
    data[Predicate('Interagisce', arity=2)][('Frank', 'George')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('George', 'Frank')] = Fact.TRUE

    # ECCEZIONE 2: Helen e Igor sono simili ma NON amici
    # (Simili ma non si conoscono)
    data[Predicate('Simile', arity=2)][('Helen', 'Igor')] = Fact.TRUE
    data[Predicate('Simile', arity=2)][('Igor', 'Helen')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('Helen', 'Igor')] = Fact.FALSE
    data[Predicate('Interagisce', arity=2)][('Igor', 'Helen')] = Fact.FALSE
    data[Predicate('Amico', arity=2)][('Helen', 'Igor')] = Fact.FALSE
    data[Predicate('Amico', arity=2)][('Igor', 'Helen')] = Fact.FALSE

    # ECCEZIONE 3: Transitività non sempre vera
    # Jack → Kevin → Lisa, ma Jack NON è amico di Lisa
    data[Predicate('Amico', arity=2)][('Jack', 'Kevin')] = Fact.TRUE
    data[Predicate('Amico', arity=2)][('Kevin', 'Lisa')] = Fact.TRUE
    data[Predicate('Amico', arity=2)][('Jack', 'Lisa')] = [0.2, 0.4]  # Conoscenti, non amici
    data[Predicate('Interagisce', arity=2)][('Jack', 'Kevin')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('Kevin', 'Lisa')] = Fact.TRUE
    data[Predicate('Interagisce', arity=2)][('Jack', 'Lisa')] = Fact.TRUE

    return data


def plot_learning_curve(losses: List[float], output_path: str = "learning_curve.png"):
    """Plotta curva di apprendimento"""
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
