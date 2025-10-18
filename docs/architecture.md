# Architettura LNN - Diagrammi e Struttura del Sistema

Questa guida illustra l'architettura interna di IBM Logical Neural Networks (LNN) attraverso diagrammi ASCII e spiegazioni strutturali.

## Indice

1. [Architettura Generale](#architettura-generale)
2. [Componenti Core](#componenti-core)
3. [Flusso di Inferenza](#flusso-di-inferenza)
4. [Training e Ottimizzazione](#training-e-ottimizzazione)

## Architettura Generale

```
┌─────────────────────────────────────────────────────────────┐
│                    Logical Neural Network                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Predicati   │  │ Operatori   │  │ Modello     │         │
│  │ Logici      │  │ Logici      │  │ LNN         │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Fatti       │  │ Regole      │  │ Vincoli     │         │
│  │ (Dati)      │  │ (Logica)    │  │ (Bounds)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ Inferenza   │  │ Training    │                          │
│  │ Engine      │  │ Engine      │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Componenti Core

### Predicati

I predicati rappresentano relazioni o proprietà nel dominio:

```
Predicato Unario:    P(x)     → "x ha proprietà P"
Predicato Binario:   R(x,y)   → "x è in relazione R con y"
```

### Operatori Logici

```
AND:  A ∧ B  → Coniunzione
OR:   A ∨ B  → Disgiunzione
NOT:  ¬A     → Negazione
IMPL: A → B  → Implicazione
```

### Modello LNN

Il modello combina logica simbolica con reti neurali:

```
Modello LNN
├── Predicati (neuroni logici)
├── Operatori (connessioni logiche)
├── Bounds (valori di verità [0,1])
└── Parametri apprendibili
```

## Flusso di Inferenza

```
Input Fatti → Predicati → Operatori Logici → Inferenza → Output
     ↓             ↓             ↓              ↓          ↓
   Dati        Attivazione    Combinazione   Propagazione Risultati
   Grezzi      Logica        Logica         Bounds       Finali
```

### Esempio: Inferenza in Azione

```
Fatti:     paziente(fred) = 1.0
           febbre(fred) = 0.8

Regola:    paziente(x) ∧ febbre(x) → malattia(x)

Inferenza: malattia(fred) = min(1.0, 0.8) = 0.8
```

## Training e Ottimizzazione

```
Dataset → Loss Function → Gradient Descent → Aggiornamento Parametri
    ↓          ↓               ↓                   ↓
Training   Calcolo Errore   Ottimizzazione     Miglioramento
Data       Supervisionato   Automatica         Modello
```

### Loss Functions

- **Logical Loss**: Penalizza violazioni logiche
- **Bound Loss**: Ottimizza intervalli di verità
- **Custom Loss**: Combinazioni specifiche per dominio

## Relazioni tra Componenti

```
┌─────────────┐     ┌─────────────┐
│   Dataset   │────▶│ Predicati   │
└─────────────┘     └─────────────┘
                         │
                         ▼
┌─────────────┐     ┌─────────────┐
│ Operatori   │◀────│   Regole    │
│ Logici      │     └─────────────┘
└─────────────┘           │
                         ▼
┌─────────────┐     ┌─────────────┐
│  Inferenza  │────▶│   Output    │
└─────────────┘     └─────────────┘
                         │
                         ▼
┌─────────────┐     ┌─────────────┐
│   Loss      │────▶│ Aggiornamento│
│ Function    │     │ Parametri   │
└─────────────┘     └─────────────┘
```

Questa architettura permette a LNN di combinare il rigore della logica simbolica con la flessibilità dell'apprendimento neurale.