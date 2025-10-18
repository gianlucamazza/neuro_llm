# Esempio 01: Sistema Esperto Medico

Sistema di diagnosi medica basato su **regole logiche** usando IBM Logical Neural Networks.

## Cosa Dimostra

- ✅ Definizione di predicati logici (sintomi e diagnosi)
- ✅ Regole di inferenza con operatori logici (AND, OR, NOT, IMPLIES)
- ✅ Gestione dell'incertezza con bounds [lower, upper]
- ✅ Inferenza bidirezionale automatica

## Concetti Chiave

### 1. Predicati come Neuroni

Ogni sintomo e diagnosi è un **neurone logico**:

```python
Ha_Febbre = Predicate('Ha_Febbre')
Influenza = Predicate('Influenza')
```

### 2. Regole Logiche

Le regole mediche diventano connessioni neurali:

```python
# Se (Febbre ∧ Dolori Muscolari) → Influenza
Implies(
    And(Ha_Febbre(x), Ha_Dolori_Muscolari(x)),
    Influenza(x)
)
```

### 3. Bounds per Incertezza

LNN gestisce incertezza con **limiti inferiore e superiore**:

```python
# Tosse moderata con incertezza
'tosse': [0.6, 0.8]  # bounds: [lower, upper]

# Febbre certa
'febbre': Fact.TRUE  # equivalente a [1.0, 1.0]
```

### 4. Open World Assumption

A differenza di Prolog (closed world), LNN assume che:
- Conoscenza può essere **incompleta**
- Assenza di informazione ≠ False
- Mantiene bounds per rappresentare "non so"

## Esecuzione

```bash
python examples/01_medical_expert/medical_diagnosis.py
```

## Output Esempio

```
==============================================================
SISTEMA ESPERTO DI DIAGNOSI MEDICA con LNN
==============================================================

Esecuzione inferenza LNN...

============================================================
DIAGNOSI PER: Mario
============================================================
⚠️ Influenza     : PROBABILE     [0.90, 1.00] ~95.0%
❓ Covid         : POSSIBILE     [0.60, 0.80] ~70.0%
✓ Raffreddore   : IMPROBABILE   [0.00, 0.20] ~10.0%
✓ Bronchite     : IMPROBABILE   [0.00, 0.10] ~5.0%

============================================================
DIAGNOSI PER: Anna
============================================================
⚠️ Raffreddore  : PROBABILE     [0.85, 1.00] ~92.5%
✓ Influenza     : IMPROBABILE   [0.00, 0.00] ~0.0%
✓ Covid         : IMPROBABILE   [0.00, 0.15] ~7.5%
```

## Regole Implementate

| Regola | Condizione | Diagnosi |
|--------|-----------|----------|
| R1 | Febbre ∧ Dolori Muscolari | Influenza |
| R2 | Tosse ∧ Mal di Gola ∧ ¬Febbre | Raffreddore |
| R3 | Febbre ∧ Tosse ∧ (Dolori ∨ Congestione) | Covid |
| R4 | Tosse ∧ Respiro Corto ∧ Dolori Muscolari | Bronchite |

## Vantaggi vs Approcci Tradizionali

### vs Sistema a Regole Tradizionale (Prolog)

- ✅ **Gestisce incertezza**: Bounds invece di True/False binario
- ✅ **Apprendibile**: Può imparare pesi delle regole dai dati
- ✅ **Robusto**: Gestisce contraddizioni senza fallire

### vs Solo Machine Learning

- ✅ **Interpretabile**: Regole esplicite leggibili
- ✅ **Spiegabile**: Traccia il ragionamento
- ✅ **Sample-efficient**: Richiede meno dati di training

## Estensioni Possibili

1. **Learning**: Apprendere pesi regole da dataset medico reale
2. **Spiegazioni**: Estrarre percorso di inferenza per spiegare diagnosi
3. **Integrazione sensori**: Input da dispositivi medici con incertezza
4. **Multi-step reasoning**: Diagnosi differenziale con test progressivi

## Note Tecniche

- **Inferenza**: Bidirezionale (forward + backward chaining)
- **Complessità**: O(n) per n regole (molto efficiente)
- **Memoria**: Mantiene bounds per ogni grounding di predicato
- **Differenziabilità**: End-to-end, permette gradient-based learning

## Riferimenti

- [LNN Paper](https://arxiv.org/abs/2006.13155) - Teoria matematica
- [IBM LNN Docs](https://ibm.github.io/LNN/) - Documentazione ufficiale
