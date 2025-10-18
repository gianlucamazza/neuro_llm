# IBM Logical Neural Networks (LNN) - Tutorial Pratico

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/YOUR_USERNAME/neuro_llm/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/neuro_llm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/neuro_llm?style=social)](https://github.com/YOUR_USERNAME/neuro_llm)

Repository educativo che dimostra **IBM Logical Neural Networks (LNN)**, un framework neuro-simbolico che fonde reti neurali e logica simbolica, ispirandosi a processi cognitivi umani.

## 📚 Documentazione

| Sezione | Descrizione |
|---------|-------------|
| [🧠 Teoria LNN](docs/teoria_lnn.md) | Fondamenti matematici, architettura e analogie neuroscientifiche |
| [⚙️ Installazione](docs/installation.md) | Setup ambiente e dipendenze |
| [🔍 Confronti](docs/comparisons.md) | LNN vs altri approcci (Prolog, NN, ProbLog) |
| [☁️ Google Colab](docs/colab_setup.md) | Guida completa per esecuzione su Colab |
| [📚 API Reference](docs/api_reference.md) | Guida completa alle API LNN |
| [🏗️ Architettura](docs/architecture.md) | Diagrammi e struttura del sistema |
| [📖 Glossario](docs/glossary.md) | Termini e concetti chiave |
| [🔧 Troubleshooting](docs/troubleshooting.md) | Risoluzione problemi comuni |
| [💻 Piattaforme](docs/platform_specific.md) | Configurazioni per OS specifici |


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

Vedi [docs/colab_setup.md](docs/colab_setup.md#architettura-del-codice-dry-principle) per dettagli.

## 💡 Perché LNN?

LNN eccelle in domini che richiedono **ragionamento rigoroso + apprendimento da dati**:
- **Healthcare**: Diagnosi con regole mediche + apprendimento da casi clinici
- **Finance**: Fraud detection con compliance rules + pattern learning
- **Autonomous Systems**: Safety constraints + ottimizzazione da esperienza

Vedi [confronti dettagliati](docs/comparisons.md) con altri approcci.

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

