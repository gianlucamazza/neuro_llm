# LNN vs Altri Approcci

Confronto dettagliato tra IBM Logical Neural Networks e altre tecnologie per ragionamento e AI.

## Indice

1. [LNN vs Prolog](#lnn-vs-prolog)
2. [LNN vs Neural Networks Puri](#lnn-vs-neural-networks)
3. [LNN vs Probabilistic Logic](#lnn-vs-probabilistic-logic)
4. [LNN vs Knowledge Graphs](#lnn-vs-knowledge-graphs)
5. [LNN vs Expert Systems Tradizionali](#lnn-vs-expert-systems)
6. [LNN vs LLM Puri](#lnn-vs-llm)
7. [Quando Usare Cosa](#quando-usare-cosa)

## LNN vs Prolog

### Overview

| Caratteristica | Prolog | LNN |
|----------------|--------|-----|
| **Paradigma** | Logica simbolica pura | Neuro-Simbolico |
| **Valori verità** | Binari (true/false) | Continui [0, 1] con bounds |
| **World Assumption** | Closed World | Open World |
| **Incertezza** | ❌ Non gestita | ✅ Nativa (bounds) |
| **Learning** | ❌ Regole manuali | ✅ Pesi apprendibili |
| **Contraddizioni** | ❌ Fallisce | ✅ Gestisce gracefully |
| **Differenziabile** | ❌ | ✅ End-to-end |

### Esempio Comparativo

#### Prolog

```prolog
% Regola: tutti gli amici sono simili
similar(X, Y) :- friend(X, Y).

% Fatti
friend(alice, bob).
friend(charlie, diana).

% Query
?- similar(alice, bob).
true.

% Problema: eccezione causa fallimento
friend(frank, george).
% Se frank e george NON sono simili → contraddizione → fallimento
```

#### LNN

```python
from lnn import *

# Regola: amici → simili (ma apprendibile)
model.add_knowledge(
    Implies(Friend(x, y), Similar(x, y)),
    world=World.AXIOM  # Può avere eccezioni
)

# Fatti con incertezza
Friend: {
    ('alice', 'bob'): [0.9, 1.0],      # Amici certi
    ('frank', 'george'): [0.8, 0.9]     # Amici probabili
}

Similar: {
    ('alice', 'bob'): [0.7, 0.9],      # Simili
    ('frank', 'george'): [0.1, 0.3]     # NON simili (eccezione!)
}

# LNN apprende che la regola ha eccezioni e adatta i pesi
# Nessun fallimento, gestione robusta
```

### Quando Usare Prolog

✅ **Pro:**
- Più veloce per logica puramente simbolica
- Sintassi più concisa
- Mature tooling
- Ottimo per constraint solving

❌ **Contro:**
- Nessuna gestione incertezza
- Nessun learning
- Fragile con eccezioni

**Use Case Ideali:**
- Sistemi esperti con regole certe al 100%
- Parsing e pattern matching
- Constraint satisfaction problems
- Prototipazione rapida logica simbolica

### Quando Usare LNN

✅ **Pro:**
- Gestisce incertezza
- Apprende da dati
- Robusto a contraddizioni
- Differenziabile

❌ **Contro:**
- Setup più complesso
- Più lento di Prolog puro
- Richiede comprensione NN

**Use Case Ideali:**
- Dati rumorosi o incerti
- Serve learning da esempi
- Regole hanno eccezioni
- Integrazione con ML pipeline

## LNN vs Neural Networks

### Overview

| Caratteristica | Deep Neural Networks | LNN |
|----------------|---------------------|-----|
| **Interpretabilità** | ❌ Black box | ✅ Regole esplicite |
| **Spiegabilità** | ❌ Difficile | ✅ Tracciabile |
| **Sample Efficiency** | ❌ Richiede molti dati | ✅ Pochi esempi sufficienti |
| **Garanzie** | ❌ Nessuna | ✅ Parziali (logiche) |
| **Transfer Learning** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Scalabilità** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Domain Knowledge** | ❌ Difficile iniettare | ✅ Facile (regole) |

### Esempio: Classificazione Medica

#### Neural Network

```python
import torch.nn as nn

class DiagnosisNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)  # 10 sintomi
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)   # 4 diagnosi

    def forward(self, symptoms):
        x = F.relu(self.fc1(symptoms))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# PRO: Può catturare pattern complessi
# CONTRO: "Perché ha diagnosticato influenza?" → nessuna risposta chiara
```

#### LNN

```python
from lnn import *

# Regola esplicita
Implies(
    And(Fever(x), MusclePain(x)),
    Influenza(x)
)

# PRO: "Diagnosticato influenza perché: Fever=TRUE ∧ MusclePain=TRUE"
# CONTRO: Pattern non espliciti nelle regole sono persi
```

### Quando Usare Neural Networks

✅ **Usare NN quando:**
- Pattern complessi non esprimibili con regole
- Enorme quantità di dati disponibile
- Performance > Interpretabilità
- Computer vision, NLP, speech
- Transfer learning da modelli pre-trained

### Quando Usare LNN

✅ **Usare LNN quando:**
- Interpretabilità è critica (medicina, finanza, legal)
- Pochi dati disponibili
- Domain knowledge esistente (regole note)
- Serve spiegare decisioni
- Affidabilità e verificabilità richieste

### Soluzione: Hybrid (LNN + NN)

```python
# BEST OF BOTH WORLDS

# NN per feature extraction
features = neural_net.extract_features(image)

# LNN per ragionamento
lnn_input = convert_features_to_predicates(features)
diagnosis = lnn.infer(lnn_input)

# → Feature learning + Ragionamento interpretabile
```

## LNN vs Probabilistic Logic

### Overview

| Caratteristica | ProbLog / MLN | LNN |
|----------------|---------------|-----|
| **Semantica** | Probabilità | Bounds [L, U] |
| **Inferenza** | MCMC/Sampling | Differenziale |
| **Training** | EM / Sampling | Gradient Descent |
| **Scalabilità** | ⭐⭐ | ⭐⭐⭐⭐ |
| **Velocità** | Lento | Veloce |
| **Incertezza** | Probabilità precise | Range di valori |

### Probabilistic Logic (ProbLog)

```prolog
% Ogni regola ha probabilità
0.8 :: similar(X, Y) :- friend(X, Y).
0.6 :: friend(X, Y) :- similar(X, Y), interact(X, Y).

% Query: P(similar(alice, bob)) = ?
% Calcola con sampling o exact inference
```

**Pro:**
- Probabilità esatte
- Teoria ben fondata (probability theory)

**Contro:**
- Inferenza lenta (NP-hard)
- Non differenziabile
- Difficile scale

### LNN

```python
# Regole con bounds
Implies(Friend(x, y), Similar(x, y))  # Non probabilità, bounds

# Inferenza rapida (forward/backward pass)
# Differenziabile → training veloce
```

**Pro:**
- Molto più veloce
- Differenziabile
- Scala meglio

**Contro:**
- Bounds invece di probabilità esatte
- Meno mature teoricamente

### Quando Usare Cosa

| Scenario | ProbLog/MLN | LNN |
|----------|-------------|-----|
| Serve probabilità esatta | ✅ | ❌ |
| Dataset grande | ❌ | ✅ |
| Training con gradient | ❌ | ✅ |
| Integrazione con NN | ❌ | ✅ |
| Teoria consolidata | ✅ | ⭐⭐⭐ |

## LNN vs Knowledge Graphs

### Overview

| Caratteristica | Knowledge Graph (Neo4j) | LNN |
|----------------|------------------------|-----|
| **Rappresentazione** | Grafo (nodi + edges) | Grafo logico (predicati) |
| **Query** | Cypher, SPARQL | Inferenza logica |
| **Ragionamento** | ⭐⭐ (limitato) | ⭐⭐⭐⭐⭐ |
| **Storage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Learning** | ⭐⭐ (embedding) | ⭐⭐⭐⭐⭐ |

### Knowledge Graph

```cypher
// Neo4j
CREATE (leo:Person {name: "Leonardo"})
CREATE (vinci:City {name: "Vinci"})
CREATE (leo)-[:BORN_IN]->(vinci)

// Query: transitività
MATCH (p:Person)-[:BORN_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
RETURN p, country

// PRO: Efficiente per grandi grafi
// CONTRO: Ragionamento limitato, regole complesse difficili
```

### LNN

```python
# Regole logiche con inferenza automatica
BornIn(leonardo, vinci)
LocatedIn(vinci, italy)

# Regola transitività (automatica)
Implies(
    And(BornIn(x, y), LocatedIn(y, z)),
    BornIn(x, z)  # O CitizenOf(x, z)
)

# Inferisce automaticamente: leonardo è in italy
```

### Soluzione Ibrida

```python
# Usa KG per storage + LNN per reasoning

# 1. Carica dati da Neo4j
facts = load_from_neo4j(query)

# 2. Popola LNN
lnn.add_data(facts)

# 3. Ragiona con LNN
inferred = lnn.infer()

# 4. Salva inferenze in Neo4j
save_to_neo4j(inferred)

# → Storage scalabile + Ragionamento potente
```

## LNN vs Expert Systems Tradizionali

### Expert Systems (CLIPS, Drools)

```clips
; CLIPS rule
(defrule diagnose-flu
  (symptom fever yes)
  (symptom cough yes)
  =>
  (assert (diagnosis influenza)))

; PRO: Mature, production-ready
; CONTRO: No uncertainty, no learning
```

### LNN

```python
# Stesso concetto, ma con:
# - Incertezza (bounds)
# - Learning (training)
# - Differenziabilità

Implies(
    And(Fever(x), Cough(x)),
    Influenza(x)
)
# Pesi apprendibili da dati clinici
```

## LNN vs LLM Puri

### Large Language Models (GPT, Claude)

| Caratteristica | LLM | LNN | LLM + LNN |
|----------------|-----|-----|-----------|
| **Comprensione NL** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Ragionamento** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Hallucinations** | ⚠️ Alto | ✅ Nessuna | ✅ Minimizzate |
| **Spiegabilità** | ❌ | ✅ | ✅ |
| **Costo** | 💰💰💰 | 💰 | 💰💰 |

### Solo LLM

```python
# Prompt engineering
response = llm.query("""
Se Leonardo nacque a Vinci in Toscana,
di quale paese è cittadino?
""")

# PRO: Flessibile, comprende NL
# CONTRO: Può "allucinare", non tracciabile
```

### Solo LNN

```python
# Fatti strutturati
BornIn(leonardo, vinci)
LocatedIn(vinci, toscana)
LocatedIn(toscana, italia)

# Regola
BornIn(x, y) → CitizenOf(x, y)

# Inferisce: CitizenOf(leonardo, italia)

# PRO: Affidabile, tracciabile
# CONTRO: Richiede fatti strutturati (no NL)
```

### Hybrid: LLM + LNN ✨

```python
# 1. LLM estrae fatti da testo
text = "Leonardo nacque a Vinci in Toscana, Italia"
facts = llm.extract_facts(text)
# → {BornIn(Leonardo, Vinci), LocatedIn(Vinci, Toscana), ...}

# 2. LNN ragiona
lnn.add_facts(facts)
inferred = lnn.infer()
# → CitizenOf(Leonardo, Italia) [0.9, 1.0]

# 3. LLM genera risposta
answer = llm.generate_response(inferred)
# → "Leonardo è cittadino italiano perché nacque a Vinci..."

# ✅ Comprensione NL + Ragionamento rigoroso + Spiegabilità
```

Vedi [Esempio 03](../examples/03_hybrid_llm/) per implementazione completa.

## Quando Usare Cosa

### Decision Tree

```
Hai già regole logiche chiare?
│
├─ SÌ → Domain è incerto/probabilistico?
│       │
│       ├─ SÌ → Serve learning?
│       │       │
│       │       ├─ SÌ → LNN ✅
│       │       └─ NO → ProbLog o LNN
│       │
│       └─ NO → Regole sempre valide?
│               │
│               ├─ SÌ → Prolog ✅
│               └─ NO → LNN ✅
│
└─ NO → Hai molti dati?
        │
        ├─ SÌ → Serve interpretabilità?
        │       │
        │       ├─ SÌ → LNN + NN ibrido ✅
        │       └─ NO → Neural Network ✅
        │
        └─ NO → Hai expert domain knowledge?
                │
                ├─ SÌ → LNN ✅
                └─ NO → LLM + LNN ✅
```

### Use Cases per Tecnologia

#### Prolog ✅
- Compilatori, parser
- Constraint satisfaction
- Sistemi esperti deterministici
- Prototipazione logica

#### Neural Networks ✅
- Computer vision
- Speech recognition
- Pattern recognition complessi
- Quando interpretabilità non critica

#### LNN ✅
- Diagnosi medica
- Sistemi legali/compliance
- Fraud detection
- Scientific reasoning
- Quando serve spiegabilità + learning

#### LLM ✅
- Chatbots
- Content generation
- General question answering
- Quando comprensione NL è priorità

#### LNN + LLM ✅ (Hybrid)
- Question answering su documenti
- Compliance checking con NL
- Medical diagnosis da patient notes
- Legal reasoning

## Conclusione

**Non esiste soluzione universale.** La scelta dipende da:

1. **Dati disponibili**: Pochi → LNN, Molti → NN
2. **Interpretabilità**: Critica → LNN, No → NN
3. **Incertezza**: Alta → LNN o ProbLog, Nessuna → Prolog
4. **Learning**: Necessario → LNN o NN, No → Prolog
5. **Domain**: NL → LLM, Logica → Prolog/LNN, Pattern → NN

**LNN brilla quando serve:**
- ✅ Ragionamento rigoroso
- ✅ Interpretabilità completa
- ✅ Gestione incertezza
- ✅ Apprendimento da pochi dati
- ✅ Integrazione domain knowledge

**Best Practice: Architetture Ibride**

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│   LLM   │────▶│   LNN   │────▶│   KG    │
│(NL→Facts)│    │(Reasoning)│   │(Storage)│
└─────────┘     └─────────┘     └─────────┘

= Comprensione + Ragionamento + Scalabilità
```

---

Per approfondimenti, vedi:
- [Teoria LNN](teoria_lnn.md)
- [Esempi Pratici](../examples/)
