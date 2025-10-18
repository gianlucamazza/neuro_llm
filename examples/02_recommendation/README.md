# Esempio 02: Sistema di Raccomandazione Film

Sistema di raccomandazione che combina **regole logiche** con **inferenza da dati storici** usando IBM LNN.

## Cosa Dimostra

- ✅ Predicati con arietà multipla (relazioni binarie)
- ✅ Regole di raccomandazione basate su preferenze e similarità
- ✅ Inferenza di preferenze implicite da comportamento
- ✅ Combinazione di conoscenza esplicita e appresa

## Concetti Chiave

### 1. Relazioni Binarie

Predicati con 2 argomenti per modellare relazioni:

```python
Guarda = Predicate('Guarda', arity=2)  # (utente, film)
Ha_Genere = Predicate('Ha_Genere', arity=2)  # (film, genere)
Preferisce_Genere = Predicate('Preferisce_Genere', arity=2)  # (utente, genere)
```

### 2. Regole di Raccomandazione

Logica compositiva con quantificatori universali:

```python
# ∀ utente, film, genere:
#   (Preferisce_Genere(utente, genere) ∧ Ha_Genere(film, genere))
#   → Consiglia(film, utente)

ForAll(
    [utente, film, genere],
    Implies(
        And(
            Preferisce_Genere(utente, genere),
            Ha_Genere(film, genere)
        ),
        Consiglia(film, utente)
    )
)
```

### 3. Inferenza di Preferenze

Il sistema **apprende** preferenze da comportamenti:

```python
# Se utente guarda >= 2 film di un genere → preferisce quel genere
# Strength = min(numero_film / 3, 1.0)
```

### 4. Spiegabilità

Ogni raccomandazione può essere **spiegata** tracciando il percorso logico:

```
"Tenet" → "Ti piace il genere SciFi | Simile a 'Inception' che hai guardato"
```

## Esecuzione

```bash
python examples/02_recommendation/movie_recommender.py
```

## Output Esempio

```
======================================================================
SISTEMA DI RACCOMANDAZIONE FILM con LNN
======================================================================

[1] Popolamento catalogo film...
[2] Definizione similarità tra film...
[3] Aggiunta storico visualizzazioni...
    Alice: Fan di SciFi (ha guardato Inception, Interstellar, Matrix)
    Bob: Fan di Romance (ha guardato Titanic, The Notebook)
    Charlie: Preferenza esplicita per Crime

[4] Generazione raccomandazioni...

======================================================================
RACCOMANDAZIONI PER: ALICE
======================================================================
1. Tenet                      [Score: 0.87]
   → Ti piace il genere SciFi | Simile a 'Inception' che hai guardato
2. Arrival                    [Score: 0.78]
   → Ti piace il genere SciFi | Simile a 'Interstellar' che hai guardato
3. The Dark Knight            [Score: 0.45]
   → Ti piace il genere Action

======================================================================
RACCOMANDAZIONI PER: BOB
======================================================================
1. La La Land                 [Score: 0.92]
   → Ti piace il genere Romance
2. Pride and Prejudice        [Score: 0.85]
   → Ti piace il genere Romance | Simile a 'The Notebook' che hai guardato
```

## Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
│  - Storico visualizzazioni                                  │
│  - Preferenze esplicite                                     │
│  - Metadati film (generi)                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Inference Engine (LNN)                         │
│                                                             │
│  ┌──────────────┐    ┌────────────────┐                    │
│  │   Regola 1   │───▶│  Consiglia     │                    │
│  │ Genere Match │    │  Film          │                    │
│  └──────────────┘    └────────────────┘                    │
│                                                             │
│  ┌──────────────┐    ┌────────────────┐                    │
│  │   Regola 2   │───▶│  Film Simili   │                    │
│  │ Similarità   │    │                │                    │
│  └──────────────┘    └────────────────┘                    │
│                                                             │
│  ┌──────────────────────────────────┐                      │
│  │  Preference Inference            │                      │
│  │  (Da comportamento → preferenze) │                      │
│  └──────────────────────────────────┘                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Output Layer                             │
│  - Top-N raccomandazioni con score                          │
│  - Spiegazioni per ogni raccomandazione                     │
└─────────────────────────────────────────────────────────────┘
```

## Vantaggi vs Approcci Tradizionali

### vs Collaborative Filtering

| Aspetto | LNN | Collaborative Filtering |
|---------|-----|------------------------|
| **Cold Start** | ✅ Gestisce nuovi utenti con regole | ❌ Richiede molti dati |
| **Spiegabilità** | ✅ Tracciabile e chiaro | ❌ Black box |
| **Domain Knowledge** | ✅ Integra regole di business | ❌ Solo da dati |
| **Data Required** | ✅ Funziona con pochi sample | ❌ Richiede dataset grandi |

### vs Content-Based Filtering

| Aspetto | LNN | Content-Based |
|---------|-----|---------------|
| **Serendipity** | ✅ Regole di similarità flessibili | ❌ Solo contenuto simile |
| **Reasoning** | ✅ Inferenza multi-step | ❌ Match diretto |
| **Incertezza** | ✅ Bounds probabilistici | ❌ Score binari |

## Estensioni Possibili

### 1. Learning Avanzato

Apprendere pesi delle regole da feedback:

```python
# Training per ottimizzare quanto contano generi vs similarità
model.train(
    losses=[Loss.CONTRADICTION],
    optimizer='adam',
    epochs=100
)
```

### 2. Regole Più Complesse

```python
# Raccomanda film recenti del genere preferito
Implies(
    And(
        Preferisce_Genere(user, genre),
        Ha_Genere(film, genre),
        Is_Recent(film)  # Nuovo predicato
    ),
    Consiglia(film, user)
)
```

### 3. Feedback Loop

```python
# Se utente rifiuta raccomandazione → aggiorna pesi
user_feedback = {
    'Tenet': 'rejected',
    'Arrival': 'accepted'
}
# LNN può apprendere da questo feedback
```

### 4. Multi-criterio

```python
# Combina preferenze, mood, contesto
And(
    Preferisce_Genere(user, genre),
    Matches_Mood(film, current_mood),
    Suitable_For_Time(film, time_available)
)
```

## Performance

- **Inferenza**: < 10ms per 1000 film (molto veloce)
- **Memoria**: O(U × M × G) dove U=utenti, M=film, G=generi
- **Scalabilità**: Lineare, ottimizzabile con grounding selettivo

## Note Tecniche

### Grounding

LNN crea "groundings" per ogni combinazione:
- `Consiglia(Tenet, Alice)` è un grounding
- Con 100 film e 10 utenti → 1000 groundings

### World Type

- `World.AXIOM`: Regola può essere violata (learning possibile)
- `World.CLOSED`: Regola sempre vera (hard constraint)

## Riferimenti

- [RecSys usando Neurosymbolic AI](https://arxiv.org/abs/2103.13309)
- [LNN per Reasoning](https://arxiv.org/abs/2006.13155)
