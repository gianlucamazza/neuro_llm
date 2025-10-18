# IBM Logical Neural Networks (LNN) - Tutorial Pratico

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/YOUR_USERNAME/neuro_llm/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/neuro_llm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/neuro_llm?style=social)](https://github.com/YOUR_USERNAME/neuro_llm)

Repository esplicativo che dimostra l'uso di **IBM Logical Neural Networks (LNN)**, un framework rivoluzionario che fonde completamente reti neurali e logica simbolica.

> **📝 Nota**: Prima di pubblicare su GitHub, sostituisci `YOUR_USERNAME` nei badge sopra con il tuo username GitHub.

## Cosa Rende LNN Rivoluzionario

LNN è un framework che crea una corrispondenza **1-a-1 tra neuroni e operazioni logiche**. A differenza delle reti neurali tradizionali che processano solo in avanti, LNN esegue **inferenza bidirezionale**, permettendo sia modus ponens che modus tollens.

### Caratteristiche Chiave

- **Neuroni Logici**: Ogni neurone rappresenta un'operazione logica con pesi interpretabili
- **Open World Assumption**: Gestisce conoscenza incompleta mantenendo limiti superiori e inferiori per ogni variabile
- **End-to-End Differenziabile**: Loss function che cattura contraddizioni logiche
- **Ragionamento + Apprendimento**: Combina regole logiche esplicite con apprendimento dai dati

## Quick Start

### Opzione 1: Esegui su Google Colab (Consigliato)

Prova gli esempi direttamente nel browser senza installazione:

| Esempio | Descrizione | Colab |
|---------|-------------|-------|
| 01. Sistema Esperto Medico | Diagnosi con regole logiche | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/01_medical_expert/medical_diagnosis.ipynb) |
| 02. Raccomandazione Film | Sistema di raccomandazione | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/02_recommendation/movie_recommender.ipynb) |
| 03. Sistema Ibrido LNN+LLM | NLU con LNN e Claude | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/03_hybrid_llm/hybrid_nlu.ipynb) |
| 04. Learning con LNN | Training di pesi logici | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/04_learning/learning_weights.ipynb) |

> **Nota**: Per l'esempio 03 (Hybrid LNN+LLM), configura la tua API key Anthropic nei [Colab Secrets](docs/colab_setup.md#configurazione-api-keys). Il notebook funziona anche in demo mode senza API key.

### Opzione 2: Installazione Locale

```bash
# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### Esegui Esempi

```bash
# Sistema esperto medico
python examples/01_medical_expert/medical_diagnosis.py

# Sistema di raccomandazione
python examples/02_recommendation/movie_recommender.py

# Learning con LNN
python examples/04_learning/learning_weights.py
```

### Run Tests

```bash
# Esegui tutti i test
pytest

# Con coverage report
pytest --cov=examples --cov-report=html
```

## Esempi Inclusi

### 1. Sistema Esperto Medico

Sistema di diagnosi medica che usa regole logiche per inferire malattie da sintomi.

**Tecniche dimostrate:**
- Definizione di predicati e regole logiche
- Gestione dell'incertezza con bounds
- Inferenza bidirezionale

[Vai all'esempio →](examples/01_medical_expert/)

### 2. Sistema di Raccomandazione Film

Raccomandazione personalizzata che apprende preferenze da comportamenti passati.

**Tecniche dimostrate:**
- Regole di raccomandazione
- Apprendimento da pattern nei dati
- Training con loss function logica

[Vai all'esempio →](examples/02_recommendation/)

### 3. Sistema Ibrido LNN + LLM

Integrazione di LNN con Large Language Models (Claude) per NLU avanzata.

**Tecniche dimostrate:**
- LLM per estrazione fatti da testo naturale
- LNN per ragionamento strutturato
- Pipeline ibrida end-to-end

[Vai all'esempio →](examples/03_hybrid_llm/)

### 4. Learning con LNN

Esempio di training per apprendere pesi logici da dati con eccezioni.

**Tecniche dimostrate:**
- Training con contraddizioni logiche
- Apprendimento di "quanto" una regola è affidabile
- Gestione di eccezioni nei dati

[Vai all'esempio →](examples/04_learning/)

## Struttura del Repository

```
neuro_llm/
├── README.md                       # Questo file
├── requirements.txt                # Dipendenze Python
├── docs/                          # Documentazione approfondita
│   ├── teoria_lnn.md              # Come funziona LNN
│   ├── installation.md            # Guida installazione
│   ├── comparisons.md             # LNN vs altri approcci
│   └── colab_setup.md             # Guida Google Colab
├── examples/                      # Esempi funzionanti
│   ├── 01_medical_expert/
│   │   ├── medical_diagnosis.py   # Codice riutilizzabile (fonte di verità)
│   │   └── medical_diagnosis.ipynb # Demo interattiva Colab
│   ├── 02_recommendation/
│   │   ├── movie_recommender.py
│   │   └── movie_recommender.ipynb
│   ├── 03_hybrid_llm/
│   │   ├── hybrid_nlu.py
│   │   └── hybrid_nlu.ipynb
│   └── 04_learning/
│       ├── learning_weights.py
│       └── learning_weights.ipynb
└── tests/                         # Test suite pytest
    ├── test_medical.py
    ├── test_recommendation.py
    ├── test_learning.py
    └── test_hybrid.py
```

### Organizzazione del Codice (DRY Principle)

Il progetto segue le best practice **DRY (Don't Repeat Yourself)**:

- **File `.py`**: Contengono le classi e funzioni riutilizzabili (fonte di verità)
- **File `.ipynb`**: Notebook interattivi che importano dai `.py` per demo educative
- **Su Colab**: I notebook scaricano automaticamente i `.py` da GitHub
- **Localmente**: I `.py` sono già presenti per import diretto

Questa architettura garantisce:
✅ Zero duplicazione del codice
✅ Facile manutenibilità (modifica solo i `.py`)
✅ Testing semplificato (classi isolate)
✅ Massima riusabilità

Vedi [docs/colab_setup.md](docs/colab_setup.md#architettura-del-codice-dry-principle) per dettagli.

## Vantaggi di LNN

| Caratteristica | LNN | Prolog + LLM | Solo LLM |
|----------------|-----|--------------|----------|
| **Interpretabilità** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Apprendimento** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Incertezza** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Ragionamento** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Facilità d'uso** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Real World Applications

LNN è ideale per domini che richiedono sia **ragionamento rigoroso** che **apprendimento da dati**. Ecco applicazioni concrete dove LNN eccelle:

### 🏥 Healthcare & Medical AI

**Diagnosi Assistita da AI**
- Sistema che combina linee guida mediche (regole) con dati clinici (learning)
- Gestisce incertezza nei sintomi e nelle diagnosi differenziali
- Spiegabilità completa per decisioni cliniche critiche
- **Use case**: Sistema di triage ER che prioritizza pazienti basandosi su sintomi + storia clinica

**Drug Discovery**
- Ragionamento su interazioni farmacologiche note (regole chimiche)
- Apprendimento da trial clinici per predire efficacia
- Identificazione di contraddizioni in letteratura medica

### ⚖️ Legal & Compliance

**Automated Contract Review**
- Regole legali codificate in logica formale
- Apprendimento da precedenti per interpretare clausole ambigue
- Identificazione automatica di clausole contrattuali problematiche
- **Use case**: Review automatica di contratti M&A per clausole non conformi

**Regulatory Compliance**
- Verifica automatica conformità a regolamenti (GDPR, SOX, etc)
- Ragionamento su catene di responsabilità
- Audit trail completo per spiegare decisioni di compliance

### 💰 Finance & Risk Management

**Fraud Detection**
- Regole di rilevamento frodi note (pattern espliciti)
- Apprendimento di nuovi pattern da transazioni storiche
- Spiegazione chiara del perché una transazione è flagged
- **Use case**: Sistema bancario che combina regole AML con ML per rilevare frodi emergenti

**Credit Scoring**
- Regole di credit assessment standard
- Apprendimento da default storici
- Trasparenza richiesta da Fair Lending laws
- Gestione di dati incompleti (nuovi clienti)

### 🤖 Autonomous Systems

**Robot Planning**
- Regole di sicurezza hard-coded (non violabili)
- Apprendimento di strategie ottimali da esperienza
- Ragionamento su conseguenze di azioni in ambienti incerti
- **Use case**: Robot industriale che deve rispettare safety constraints mentre ottimizza efficienza

**Autonomous Driving**
- Highway code come regole logiche
- Apprendimento da driving data per situazioni edge-case
- Reasoning su intenzioni di altri agenti (pedestrian, veicoli)

### 🔬 Scientific Research

**Hypothesis Generation**
- Ragionamento su letteratura scientifica esistente
- Identificazione di gap nella conoscenza
- Generazione di ipotesi testabili basate su pattern nei dati
- **Use case**: Sistema che analizza paper biologici e suggerisce esperimenti

**Experimental Design**
- Regole del metodo scientifico
- Ottimizzazione di design sperimentale basandosi su risultati precedenti
- Spiegazione del rationale dietro design choices

### 🏢 Enterprise Knowledge Management

**Question Answering su Documenti Aziendali**
- Regole di business logic codificate
- Estrazione e ragionamento su politiche aziendali
- Integrazione LLM (comprensione NL) + LNN (reasoning rigoroso)
- **Use case**: Chatbot HR che risponde a domande su benefits policy con riferimenti esatti

**Business Process Automation**
- Workflow aziendali come regole logiche
- Apprendimento di ottimizzazioni da esecuzioni storiche
- Verifica formale di correttezza dei processi

### 🎓 Education

**Intelligent Tutoring Systems**
- Prerequisiti di apprendimento come regole
- Personalizzazione basata su performance studente
- Spiegazioni passo-passo di concetti
- **Use case**: Tutor di matematica che adatta difficoltà e spiega ogni step logico

**Assessment & Grading**
- Rubrics di valutazione come regole esplicite
- Apprendimento di criteri sfumati da esempi di grading
- Feedback consistente e spiegabile

## Perché LNN per Questi Domini?

| Requisito | Perché LNN |
|-----------|-----------|
| **Regolamentazione** | Regole esplicite verificabili da auditor |
| **Safety-Critical** | Constraints hard che non possono essere violati |
| **Accountability** | Spiegazione completa di ogni decisione |
| **Data Scarcity** | Sample-efficient, funziona con pochi dati |
| **Evolving Rules** | Facile aggiornare regole senza retraining |
| **Hybrid** | Combina domain expertise + data-driven learning |

## Case Study: Fraud Detection Bancaria

```python
# Esempio realistico semplificato
from lnn import *

# Regole esplicite (non apprendibili)
rules = [
    # R1: Transazioni > €10k richiedono verifica
    Implies(Amount(t) > 10000, RequiresVerification(t)),

    # R2: Transazioni da paese blacklisted → suspicious
    Implies(FromBlacklistedCountry(t), Suspicious(t)),

    # R3: Pattern insolito → potential fraud
    Implies(
        And(UnusualTime(t), UnusualLocation(t), UnusualAmount(t)),
        PotentialFraud(t)
    )
]

# Learning: Apprende pattern da dati storici
# Quali combinazioni di features predicono fraud reale?
model.train(historical_transactions, fraud_labels)

# Risultato: Sistema che combina:
# - Compliance rules (esplicite)
# - Learned patterns (da dati)
# - Spiegabilità (traccia reasoning)
```

**Benefici:**
- ✅ Regole compliance sempre rispettate
- ✅ Adattamento a nuovi pattern di frode
- ✅ Spiegazione chiara per ogni alert (per investigatori)
- ✅ Audit trail completo (requisito regolatorio)

---

Per implementazioni complete di questi use case, vedi la [roadmap future examples](https://github.com/YOUR_USERNAME/neuro_llm/issues).

## Risorse Utili

- **GitHub Ufficiale**: [IBM/LNN](https://github.com/IBM/LNN)
- **Paper**: [Logical Neural Networks (arXiv:2006.13155)](https://arxiv.org/abs/2006.13155)
- **Documentazione**: [IBM LNN Docs](https://ibm.github.io/LNN/)
- **Tutorial IBM**: [LNN Examples](https://github.com/IBM/LNN/tree/master/examples)

## Contribuire

Questo è un repository educativo. Sentiti libero di:
- Segnalare bug o miglioramenti via Issues
- Proporre nuovi esempi via Pull Request
- Condividere il repository con chi è interessato a Neuro-Symbolic AI

## Licenza

Esempi rilasciati sotto licenza MIT. IBM LNN è sotto Apache 2.0.

---

**Creato con** ❤️ **per dimostrare le potenzialità della Neuro-Symbolic AI**
