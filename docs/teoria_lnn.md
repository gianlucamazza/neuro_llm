# Teoria: Logical Neural Networks (LNN)

## Indice

1. [Introduzione](#introduzione)
2. [Fondamenti Matematici](#fondamenti-matematici)
3. [Architettura](#architettura)
4. [Operatori Logici](#operatori-logici)
5. [Inferenza Bidirezionale](#inferenza-bidirezionale)
6. [Learning](#learning)
7. [Confronti con Altri Approcci](#confronti)

## Introduzione

**Logical Neural Networks (LNN)** è un framework sviluppato da IBM Research che fonde completamente **logica simbolica** e **reti neurali**, creando un nuovo paradigma di Neuro-Symbolic AI.

### Perché LNN è Rivoluzionario

Tradizionalmente, AI simbolica e AI sub-simbolica (neural networks) sono state approcci separati:

| Aspetto | AI Simbolica (Prolog) | Neural Networks | **LNN** |
|---------|----------------------|-----------------|---------|
| Ragionamento | ✅ Rigoroso | ❌ Approssimato | ✅ Rigoroso + Apprendibile |
| Apprendimento | ❌ Manuale | ✅ Automatico | ✅ Automatico |
| Interpretabilità | ✅ Trasparente | ❌ Black box | ✅ Trasparente |
| Incertezza | ❌ Binaria | ✅ Probabilistica | ✅ Bounds [L, U] |

**LNN unisce il meglio di entrambi i mondi.**

## Fondamenti Matematici

### Real-Valued Logic

LNN usa **logica a valori reali** invece di booleana (True/False):

```
Valore ∈ [0, 1]

0 = FALSE
1 = TRUE
0.5 = UNKNOWN
[0.3, 0.7] = UNCERTAIN (range di valori possibili)
```

### Bounds Semantics

Ogni proposizione ha due valori: **lower bound (L)** e **upper bound (U)**:

```
[L, U] dove 0 ≤ L ≤ U ≤ 1

Esempio:
- [1.0, 1.0] = Certamente vero
- [0.0, 0.0] = Certamente falso
- [0.0, 1.0] = Completamente sconosciuto
- [0.6, 0.8] = Probabilmente vero con incertezza
```

**Interpretazione:**
- L = "Almeno quanto è vero"
- U = "Al massimo quanto è vero"
- [L, U] = "Range di verità possibile"

### Open World Assumption

A differenza di Prolog (Closed World), LNN assume **Open World**:

```
Prolog: Se non provato → FALSE
LNN:    Se non provato → UNKNOWN [0, 1]
```

Questo permette di ragionare con **conoscenza incompleta**.

## Architettura

### Neuroni Logici

In LNN, **ogni operatore logico è un neurone**:

```
           Input Neurons
                ↓
        ┌───────────────┐
        │  Logical AND  │  ← Neurone con funzione logica
        └───────┬───────┘
                ↓
           Output
```

**Esempio:**

```python
# Predicato = Neurone
Ha_Febbre = Predicate('Ha_Febbre')

# Operatore AND = Neurone compositivo
And(Ha_Febbre(x), Ha_Tosse(x))
```

### Formula come Grafo Neurale

Una regola logica diventa un **grafo di neuroni**:

```
Regola: (A ∧ B) → C

Grafo neurale:
    A ──┐
        ├─[AND]──[IMPLIES]── C
    B ──┘
```

Ogni nodo è un neurone con:
- **Input**: bounds da altri neuroni
- **Funzione**: operazione logica
- **Output**: bounds risultanti
- **Pesi**: (opzionale) per learning

### Grounding

Per ogni istanza concreta, LNN crea un **grounding**:

```python
# Predicato astratto
BornIn(x, y)

# Groundings (istanze concrete)
BornIn(Leonardo, Vinci)  → Neurone 1
BornIn(Mario, Roma)      → Neurone 2
...
```

Ogni grounding è un neurone separato nella rete.

## Operatori Logici

### AND (Congiunzione)

```
AND([L₁, U₁], [L₂, U₂]) = [max(0, L₁ + L₂ - 1), min(U₁, U₂)]
```

**Intuizione:** Entrambi devono essere veri

```python
from lnn import And

# Esempio
A = [0.7, 0.9]  # Probabilmente vero
B = [0.8, 1.0]  # Molto probabilmente vero

AND(A, B) ≈ [0.5, 0.9]  # Congiuntamente vero
```

### OR (Disgiunzione)

```
OR([L₁, U₁], [L₂, U₂]) = [max(L₁, L₂), min(1, U₁ + U₂)]
```

**Intuizione:** Almeno uno deve essere vero

```python
from lnn import Or

A = [0.3, 0.5]
B = [0.4, 0.6]

OR(A, B) ≈ [0.4, 1.0]  # Almeno uno è vero
```

### NOT (Negazione)

```
NOT([L, U]) = [1 - U, 1 - L]
```

**Intuizione:** Inverte i bounds

```python
from lnn import Not

A = [0.7, 0.9]  # Probabilmente vero
NOT(A) = [0.1, 0.3]  # Probabilmente falso
```

### IMPLIES (Implicazione)

```
A → B equivalente a ¬A ∨ B

IMPLIES([L_A, U_A], [L_B, U_B]) = OR(NOT(A), B)
```

**Intuizione:** "Se A allora B"

```python
from lnn import Implies

# Se ha_febbre → influenza
A = [0.9, 1.0]  # Ha febbre (quasi certo)
B = [0.0, 1.0]  # Influenza (sconosciuto)

IMPLIES(A, B) → inferisce bounds per B
```

### FORALL (Quantificatore Universale)

```
∀x: P(x)

Significa: P(x) è vero per OGNI x
```

**Implementazione:** Aggregazione su tutti i groundings

```python
from lnn import ForAll, Variable

x = Variable('x')

ForAll(x, Citizen(x))  # Tutti sono cittadini
```

## Inferenza Bidirezionale

LNN esegue inferenza in **entrambe le direzioni**:

### Forward (Upward)

```
Da premesse → conclusioni

A ∧ B → C
Se A=TRUE, B=TRUE → allora C=TRUE
```

**Modus Ponens classico**

### Backward (Downward)

```
Da conclusioni → premesse

A ∧ B → C
Se C=FALSE → allora (A=FALSE ∨ B=FALSE)
```

**Modus Tollens** (ragionamento per assurdo)

### Esempio Completo

```python
# Regola: Febbre ∧ Tosse → Influenza
Implies(And(Febbre, Tosse), Influenza)

# Forward:
Febbre=TRUE, Tosse=TRUE → Influenza=TRUE

# Backward:
Influenza=FALSE → (Febbre=FALSE ∨ Tosse=FALSE)
```

Questo permette di:
- **Dedurre** conseguenze (forward)
- **Spiegare** cause (backward)
- **Verificare** consistenza

## Learning

### Training End-to-End

LNN può **apprendere pesi** per le regole:

```python
# Regola con peso apprendibile
Implies(A, B, world=World.AXIOM)
         ↑
    Peso w ∈ [0, 1]
```

Durante training, LNN ottimizza quanto ogni regola è "affidabile".

### Loss Function

```python
Loss = LOGICAL_CONTRADICTION

Misura: Quanto le regole violano i dati
```

**Esempio:**

```
Regola: Amici → Simili
Dato:   Amici(A, B)=TRUE, Simili(A, B)=FALSE

→ CONTRADDIZIONE → Loss alta
→ Backprop riduce peso della regola
```

### Gradient Flow

```
Input Bounds
    ↓
[Forward Inference]  ← Differenziabile
    ↓
Output Bounds
    ↓
[Loss Computation]   ← Differenziabile
    ↓
[Backpropagation]    ← Gradient sui pesi
    ↓
[Update Weights]     ← Ottimizza regole
```

**Ogni operatore logico ha gradiente definito!**

### Esempio Training

```python
model = Model()

# Regola apprendibile
model.add_knowledge(
    Implies(Friend(x, y), Similar(x, y)),
    world=World.AXIOM  # Può avere eccezioni
)

# Dati con eccezioni
data = {
    Friend: {('A', 'B'): TRUE, ('C', 'D'): TRUE},
    Similar: {('A', 'B'): TRUE, ('C', 'D'): FALSE}  # Eccezione!
}

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    loss = model.loss(Loss.CONTRADICTION)
    loss.backward()
    optimizer.step()

# Risultato: peso della regola si adatta alle eccezioni
```

## Confronti

### LNN vs Prolog

| Caratteristica | Prolog | LNN |
|----------------|--------|-----|
| Logica | First-order | First-order + Probabilistica |
| Valori | Binari (T/F) | Continui [0, 1] |
| Incertezza | ❌ | ✅ Bounds |
| Learning | ❌ | ✅ End-to-end |
| Eccezioni | ❌ Fallisce | ✅ Gestisce |
| Open World | ❌ | ✅ |

### LNN vs Neural Networks

| Caratteristica | Neural Net | LNN |
|----------------|------------|-----|
| Struttura | Layers generici | Grafo logico |
| Interpretabilità | ❌ Black box | ✅ Regole chiare |
| Sample Efficiency | ❌ Richiede molti dati | ✅ Pochi sample |
| Ragionamento | ❌ Implicito | ✅ Esplicito |
| Garanzie | ❌ | ✅ Parziali |

### LNN vs Probabilistic Logic

| Caratteristica | ProbLog/MLN | LNN |
|----------------|-------------|-----|
| Inferenza | MCMC/Sampling | Differenziale |
| Scalabilità | ⭐⭐ | ⭐⭐⭐⭐ |
| Gradiente | ❌ | ✅ |
| Training | Lento | Veloce |
| Incertezza | Probabilità | Bounds |

## Vantaggi Chiave di LNN

### 1. Unione di Logica e Learning

```
Logica Simbolica ──┐
                   ├─→ LNN ──→ Ragionamento + Apprendimento
Neural Networks  ──┘
```

### 2. Interpretabilità Completa

Ogni inferenza è **tracciabile**:

```
C inferred perché:
  ← A ∧ B → C (Regola 1)
    ← A (Osservato)
    ← B (Inferito da Regola 2)
```

### 3. Sample Efficiency

Poche regole + pochi dati → buone performance

### 4. Gestione Incertezza Formale

Bounds matematicamente fondati (Łukasiewicz logic)

### 5. Differenziabilità

End-to-end trainable con PyTorch/TensorFlow

## Limitazioni

1. **Grounding explosion**: Con molte entità, numero di groundings cresce
2. **Espressività**: First-order logic (no higher-order)
3. **Performance**: Inferenza O(n) groundings (ottimizzabile)
4. **Complessità**: Curva di apprendimento per utenti

## Riferimenti

- **Paper originale**: [Logical Neural Networks (arXiv:2006.13155)](https://arxiv.org/abs/2006.13155)
- **GitHub**: [IBM/LNN](https://github.com/IBM/LNN)
- **Tutorial**: [IBM Research Blog](https://research.ibm.com/blog/logical-neural-networks)
- **Teoria logica**: Łukasiewicz Logic, Fuzzy Logic

## Conclusione

LNN rappresenta un **salto paradigmatico** nella Neuro-Symbolic AI:

✅ Combina ragionamento rigoroso e apprendimento automatico
✅ Interpreta completamente ogni decisione
✅ Gestisce incertezza in modo formale
✅ Differenziabile end-to-end

È ideale per applicazioni che richiedono sia **intelligenza** (learning) che **affidabilità** (logic).
