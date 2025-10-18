# Contributing to IBM LNN Examples

Prima di tutto, grazie per il tuo interesse nel contribuire! 🎉

Questo repository è pensato come risorsa educativa per la community di Neuro-Symbolic AI. Ogni contributo, grande o piccolo, è ben accetto.

## Come Contribuire

### 🐛 Report Bugs

Se trovi un bug, apri un [Issue](../../issues/new?template=bug_report.md) con:
- Descrizione chiara del problema
- Steps per riprodurlo
- Output/error message
- Ambiente (Python version, OS, LNN version)

### ✨ Proponi Nuove Feature

Per proporre nuovi esempi o miglioramenti:
1. Apri un [Issue](../../issues/new?template=feature_request.md) per discutere l'idea
2. Aspetta feedback prima di iniziare l'implementazione
3. Una volta approvato, procedi con la Pull Request

### 📝 Migliora Documentazione

La documentazione è fondamentale! Puoi contribuire:
- Correggendo typos o errori
- Migliorando spiegazioni esistenti
- Aggiungendo esempi o screenshot
- Traducendo in altre lingue

### 💻 Contribuisci Codice

#### Prerequisites

```bash
# Fork e clone il repository
git clone https://github.com/TUO_USERNAME/neuro_llm.git
cd neuro_llm

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Installa dev dependencies
pip install black flake8 pytest-cov
```

#### Workflow

1. **Crea un branch** per la tua feature:
   ```bash
   git checkout -b feature/nome-feature
   ```

2. **Implementa** la tua modifica:
   - Segui lo stile del codice esistente
   - Aggiungi docstrings chiare
   - Commenta logica complessa

3. **Aggiungi test**:
   ```bash
   # Crea test in tests/test_your_feature.py
   pytest tests/test_your_feature.py -v
   ```

4. **Format code**:
   ```bash
   black examples/ tests/
   flake8 examples/ tests/ --max-line-length=100
   ```

5. **Run all tests**:
   ```bash
   pytest tests/ -v --cov=examples
   ```

6. **Commit** con messaggio descrittivo:
   ```bash
   git add .
   git commit -m "Add: descrizione feature

   - Dettaglio 1
   - Dettaglio 2

   Closes #ISSUE_NUMBER"
   ```

7. **Push e crea Pull Request**:
   ```bash
   git push origin feature/nome-feature
   ```
   Poi apri PR su GitHub con descrizione dettagliata.

## 📋 Guidelines

### Code Style

- **Python**: Segui [PEP 8](https://peps.python.org/pep-0008/)
- **Line length**: Max 100 caratteri
- **Docstrings**: Google style
- **Type hints**: Usa dove possibile

Esempio:

```python
def diagnose_patient(symptoms: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Esegue diagnosi basata su sintomi.

    Args:
        symptoms: Dizionario {sintomo: intensità}

    Returns:
        Dizionario {malattia: (lower_bound, upper_bound)}

    Raises:
        ValueError: Se symptoms è vuoto
    """
    if not symptoms:
        raise ValueError("symptoms cannot be empty")

    # Implementation...
```

### Commit Messages

Usa [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: Nuova feature
- `fix`: Bug fix
- `docs`: Solo documentazione
- `style`: Formatting, no code change
- `refactor`: Refactoring senza cambiare behavior
- `test`: Aggiunta/modifica test
- `chore`: Maintenance (dependencies, etc)

Esempi:
```
feat(medical): add COVID-19 diagnosis rules
fix(hybrid): correct LLM API call format
docs(readme): update installation instructions
```

### Testing

- **Coverage**: Target 80%+
- **Test naming**: `test_<what>_<condition>_<expected>`
- **Fixtures**: Usa pytest fixtures per setup comune
- **Assertions**: Usa assertions chiare e specifiche

```python
def test_diagnosis_with_fever_returns_influenza(system, tolerance):
    """Test that fever symptom triggers influenza diagnosis"""
    system.add_patient('Test', {'febbre': Fact.TRUE})
    results = system.diagnose()

    assert 'Influenza' in results['Test']
    assert results['Test']['Influenza'][0] >= 0.5, "Expected high probability"
```

### Documentazione

Ogni nuovo esempio deve includere:

1. **Docstring module** completo
2. **README.md** nella cartella esempio con:
   - Descrizione
   - Cosa dimostra
   - Come eseguire
   - Output atteso
   - Estensioni possibili
3. **Comments inline** per logica non ovvia
4. **Type hints** per tutte le funzioni pubbliche

## 🆕 Aggiungere Nuovo Esempio

Template per nuovo esempio:

```
examples/
└── 05_nuovo_esempio/
    ├── README.md              # Documentazione
    ├── esempio.py             # Implementazione
    ├── esempio.ipynb          # Notebook Jupyter (opzionale)
    └── __init__.py            # Vuoto

tests/
└── test_nuovo_esempio.py      # Test suite
```

Checklist:
- [ ] Codice implementato e funzionante
- [ ] Docstrings complete
- [ ] README con spiegazione dettagliata
- [ ] Test con coverage > 80%
- [ ] Esempio di output nel README
- [ ] Jupyter notebook (opzionale)
- [ ] Aggiornato README principale

## 🔍 Review Process

Tutte le PR vengono revisionate per:
1. **Correttezza**: Codice funziona come previsto
2. **Test**: Coverage adeguata e test passano
3. **Style**: Segue guidelines
4. **Documentazione**: Chiara e completa
5. **Value**: Aggiunge valore al repository

Tempi di risposta tipici: 2-5 giorni.

## 📚 Risorse

- [IBM LNN Documentation](https://ibm.github.io/LNN/)
- [IBM LNN GitHub](https://github.com/IBM/LNN)
- [Neuro-Symbolic AI Papers](https://github.com/neurosymbolic-ai/awesome-neurosymbolic-ai)

## 💬 Domande?

- Apri un [Discussion](../../discussions)
- Contatta [@gianlucamazza](https://github.com/gianlucamazza) (se applicabile)

## ⚖️ License

Contribuendo, accetti che il tuo codice sarà rilasciato sotto [MIT License](LICENSE).

---

**Grazie per contribuire alla community Neuro-Symbolic AI!** 🚀
