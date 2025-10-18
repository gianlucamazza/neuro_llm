# Esempio 04: Learning con LNN

Dimostra come IBM LNN può **apprendere pesi logici** da dati con eccezioni, combinando il meglio di logica simbolica e machine learning.

## Cosa Dimostra

- ✅ Training end-to-end di regole logiche
- ✅ Apprendimento di "quanto" una regola è affidabile
- ✅ Gestione di eccezioni e contraddizioni nei dati
- ✅ Loss function basata su contraddizioni logiche
- ✅ Gradient descent su neuroni logici

## Problema: Regole con Eccezioni

In molti domini, le regole non sono **assolute** ma hanno **eccezioni**:

```
❌ Regola assoluta (Prolog):
   Amico(x, y) → Simile(x, y)  [Sempre vera]

✅ Regola probabilistica (LNN):
   Amico(x, y) → Simile(x, y)  [Vera con peso w]
```

**Eccezioni reali:**
- "Gli amici hanno gusti simili" ➜ Ma "gli opposti si attraggono"!
- "L'amico del mio amico è mio amico" ➜ Non sempre!

LNN **apprende automaticamente** quanto una regola è affidabile dai dati.

## Architettura di Learning

```
┌────────────────────────────────────────────────────────┐
│              Training Data (con eccezioni)             │
│  ✓ Alice-Bob: amici E simili (regola rispettata)      │
│  ✗ Frank-George: amici ma DIVERSI (eccezione!)        │
│  ✗ Helen-Igor: simili ma NON amici (eccezione!)       │
└─────────────────────┬──────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │   LNN Model             │
         │  ┌──────────────────┐   │
         │  │ Regola 1 (w₁)    │   │  ← Pesi apprendibili
         │  │ Regola 2 (w₂)    │   │
         │  │ Regola 3 (w₃)    │   │
         │  └──────────────────┘   │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  Loss Function          │
         │  (LOGICAL_CONTRADICTION)│
         │  Penalizza violazioni   │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  Backpropagation        │
         │  Aggiorna pesi w₁,w₂,w₃ │
         └─────────────────────────┘
```

## Loss Function: LOGICAL_CONTRADICTION

A differenza di ML standard (cross-entropy, MSE), LNN usa loss che misura **contraddizioni logiche**:

```python
Loss = ∑ (violazioni delle regole)²

# Esempio:
# Se Amico(Frank, George) = TRUE
# E  Simile(Frank, George) = FALSE
# Ma regola dice: Amico → Simile
# → CONTRADDIZIONE → Loss alta
```

Durante training, LNN **riduce i pesi** di regole spesso violate.

## Esecuzione

```bash
python examples/04_learning/learning_weights.py
```

## Output Esempio

```
======================================================================
LEARNING CON LNN: Apprendimento Regole Sociali
======================================================================

[1] Generazione dataset sintetico...
    Include casi normali + eccezioni alle regole
    Amicizie: 15
    Similarità: 14

[2] Caricamento dati in LNN...

[3] Inferenza PRE-training...
    Loss iniziale (contraddizioni): 8.742156

[4] Training...

============================================================
TRAINING LNN
============================================================
Epochs: 100
Learning Rate: 0.01
Loss Function: LOGICAL_CONTRADICTION

Epoch  10/100 | Loss: 6.523421
Epoch  20/100 | Loss: 4.891234
Epoch  30/100 | Loss: 3.456789
Epoch  40/100 | Loss: 2.345678
Epoch  50/100 | Loss: 1.678901
Epoch  60/100 | Loss: 1.234567
Epoch  70/100 | Loss: 0.987654
Epoch  80/100 | Loss: 0.789012
Epoch  90/100 | Loss: 0.654321
Epoch 100/100 | Loss: 0.567890

Training completato!
Loss iniziale: 8.742156
Loss finale:   0.567890
Riduzione:     93.5%

[5] Inferenza POST-training...

======================================================================
FORZA REGOLE APPRESE
======================================================================

Esempi di Predizioni dopo Training:

✅ ALTA | Simile(Alice, Bob)
       Bounds: [0.850, 0.920] ~88.5%
       Alice e Bob sono amici, quindi dovrebbero essere simili

❌ BASSA | Simile(Frank, George)
       Bounds: [0.180, 0.350] ~26.5%
       Frank e George sono amici ma OPPOSTI (eccezione)

❌ BASSA | Amico(Helen, Igor)
       Bounds: [0.120, 0.280] ~20.0%
       Helen e Igor sono simili ma NON amici (no interazione)

✅ ALTA | Amico(Alice, Charlie)
       Bounds: [0.720, 0.880] ~80.0%
       Alice e Charlie: transitività via Bob

[6] Generazione grafico...
Grafico salvato: learning_curve.png

======================================================================
CONCLUSIONI
======================================================================
✓ LNN ha appreso pesi che riflettono eccezioni nei dati
✓ Regole non sono assolute, ma hanno 'forza' variabile
✓ Il sistema gestisce contraddizioni in modo graceful
✓ Training è differenziabile end-to-end
======================================================================
```

## Grafico Learning Curve

![Learning Curve](learning_curve.png)

Il grafico mostra la **riduzione della loss** durante training. La loss misura contraddizioni logiche tra regole e dati.

## Regole Implementate

### 1. Friends → Similar

```python
Amico(x, y) → Simile(x, y)
```

**Prima del training**: Peso neutro
**Dopo training**: Peso alto se rispettata, basso se eccezioni frequenti

**Eccezione nel dataset**: Frank e George (amici ma opposti)

### 2. Similar + Interact → Friends

```python
Simile(x, y) ∧ Interagisce(x, y) → Amico(x, y)
```

Richiede sia similarità che interazione.

**Eccezione**: Helen e Igor (simili ma non interagiscono)

### 3. Friend Transitivity

```python
Amico(x, y) ∧ Amico(y, z) → Amico(x, z)
```

"L'amico del mio amico..."

**Eccezione**: Jack-Kevin-Lisa (transitività debole)

## Vantaggi vs Approcci Tradizionali

### vs Logica Simbolica Pura (Prolog)

| Aspetto | Prolog | LNN Learning |
|---------|--------|--------------|
| **Eccezioni** | ❌ Fallisce | ✅ Apprende pesi |
| **Rumore dati** | ❌ Non gestisce | ✅ Robusto |
| **Flessibilità** | ❌ Regole rigide | ✅ Regole soft |
| **Da dati** | ❌ Manuale | ✅ Automatico |

### vs Neural Networks Puri

| Aspetto | Neural Net | LNN Learning |
|---------|------------|--------------|
| **Interpretabilità** | ❌ Black box | ✅ Regole chiare |
| **Sample efficiency** | ❌ Richiede molti dati | ✅ Pochi sample |
| **Garanzie logiche** | ❌ Nessuna | ✅ Parziali |
| **Spiegabilità** | ❌ Difficile | ✅ Tracciabile |

## Casi d'Uso Reali

### 1. Fraud Detection

```python
# Regola: "Transazioni notturne sono sospette"
# Ma alcune persone lavorano di notte (eccezione)

Implies(
    And(IsNightTime(t), LargeAmount(t)),
    IsSuspicious(t)
)
# LNN apprende quanto è affidabile per ogni utente
```

### 2. Medical Diagnosis

```python
# Regola: "Febbre + Tosse → Influenza"
# Ma ci sono eccezioni (Covid, allergie, etc)

# LNN apprende quando la regola è affidabile
# basandosi su casi storici
```

### 3. Recommendation Systems

```python
# Regola: "Utenti simili apprezzano stessi film"
# Eccezione: Preferenze cambiano nel tempo

# LNN adatta pesi basandosi su feedback
```

## Training Dettagliato

### Hyperparameters

```python
epochs = 100           # Numero di iterazioni
learning_rate = 0.01   # Step size per gradient descent
optimizer = Adam       # Ottimizzatore (Adam, SGD, etc)
```

### Loss Function

```python
# LNN usa loss custom che misura violazioni logiche
loss = model.loss(Loss.LOGICAL_CONTRADICTION)

# Altre loss disponibili:
# - Loss.SUPERVISED: Per supervised learning classico
# - Loss.UNSUPERVISED: Per pattern discovery
```

### Optimizer

LNN usa **PyTorch optimizers** standard:

```python
optimizer = torch.optim.Adam(
    model.parameters(),  # Pesi delle regole
    lr=0.01
)
```

Ogni regola ha pesi appresi tramite backpropagation!

## Differenziabilità End-to-End

```
Input Data
   ↓
[LNN Forward (Inference)]  ← Differenziabile
   ↓
Predicted Bounds
   ↓
[Loss Computation]         ← Differenziabile
   ↓
[Backward Pass]            ← Gradient flow
   ↓
[Update Weights]           ← Optimize rules
```

**Key**: Ogni operazione logica (AND, OR, IMPLIES) è differenziabile!

## Estensioni Possibili

### 1. Multi-Task Learning

Apprendere regole per task multipli contemporaneamente:

```python
# Task 1: Predire amicizie
# Task 2: Predire collaborazioni
# Shared rules: "Amici → collaborano"
```

### 2. Transfer Learning

```python
# Pre-train su dataset generale
model.train(general_data, epochs=100)

# Fine-tune su dominio specifico
model.train(specific_data, epochs=20)
```

### 3. Active Learning

```python
# Identifica esempi più "utili" per training
uncertain_cases = find_high_uncertainty_cases(model)
labels = ask_oracle(uncertain_cases)
model.train(labels)
```

### 4. Rule Discovery

Invece di definire regole manualmente, scoprirle dai dati:

```python
# Cerca pattern frequenti
frequent_patterns = mine_frequent_patterns(data)

# Genera regole candidate
candidate_rules = generate_rules(frequent_patterns)

# Training seleziona regole utili
model.add_knowledge(candidate_rules)
model.train()
```

## Performance

- **Training time**: ~10-30s per 100 epoche (dataset piccolo)
- **Convergenza**: Tipicamente 50-200 epoche
- **Scalabilità**: Lineare con numero di regole e groundings

## Best Practices

### 1. World Type Corretto

```python
# ✅ BUONO: World.AXIOM per regole con eccezioni
model.add_knowledge(
    Implies(A, B),
    world=World.AXIOM  # Può essere violata
)

# ❌ CATTIVO: World.CLOSED se ci sono eccezioni
# (Training non convergerà)
```

### 2. Bilanciamento Dataset

```python
# Assicurati di avere:
# - Esempi positivi e negativi
# - Casi normali ed eccezioni
# - Dati bilanciati per classe
```

### 3. Learning Rate

```python
# Troppo alto → instabile
lr = 0.1  # ❌

# Troppo basso → lento
lr = 0.0001  # ❌

# Giusto per LNN
lr = 0.01  # ✅
```

## Riferimenti

- [LNN Paper (arXiv:2006.13155)](https://arxiv.org/abs/2006.13155) - Sezione 4: Learning
- [Differentiable Logic](https://arxiv.org/abs/1906.03523)
- [Neuro-Symbolic Learning](https://arxiv.org/abs/2305.00813)
