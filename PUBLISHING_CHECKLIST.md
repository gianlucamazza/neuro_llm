# GitHub Publishing Checklist

Usa questa checklist prima di pubblicare il repository su GitHub.

## Pre-Publishing

### 1. Configura Informazioni Repository

- [ ] Sostituisci `YOUR_USERNAME` con il tuo GitHub username in:
  - [ ] `README.md` (badge links, righe 3-8, 324)
  - [ ] `CHANGELOG.md` (link in fondo)
  - [ ] `.github/workflows/tests.yml` (se necessario)

- [ ] Verifica username in Colab badges (README.md righe 33-36):
  - Attualmente: `gianlucamazza`
  - [ ] Conferma o aggiorna se hai fatto fork

- [ ] Aggiorna `CHANGELOG.md`:
  - [ ] Cambia data release se diversa da 2025-01-18
  - [ ] Aggiungi note specifiche se hai fatto modifiche

### 2. Verifica File Essenziali

- [x] `LICENSE` presente (MIT)
- [x] `README.md` completo con badges
- [x] `CONTRIBUTING.md` presente
- [x] `CODE_OF_CONDUCT.md` presente
- [x] `CHANGELOG.md` presente
- [x] `.gitignore` configurato
- [x] `requirements.txt` aggiornato
- [x] `.github/` folder con templates e workflows

### 3. Testa Esempi Localmente

```bash
# Attiva environment
source venv/bin/activate

# Test esempi funzionano
python examples/01_medical_expert/medical_diagnosis.py
python examples/02_recommendation/movie_recommender.py
python examples/04_learning/learning_weights.py

# Esempio 03 richiede API key (opzionale)
export ANTHROPIC_API_KEY='your-key'
python examples/03_hybrid_llm/hybrid_nlu.py
```

- [ ] Esempio 01 funziona senza errori
- [ ] Esempio 02 funziona senza errori
- [ ] Esempio 03 funziona (o conferma demo mode ok)
- [ ] Esempio 04 funziona e genera grafico

### 4. Verifica Tests

```bash
# Run test suite
pytest tests/ -v

# Opzionale: check coverage
pytest tests/ --cov=examples --cov-report=html
```

- [ ] Test passano o fallimenti sono documentati
- [ ] Hai verificato warning non critici
- [ ] Coverage > 60% (ottimale > 80%)

### 5. Verifica Notebooks

**Se hai modificato i file .py**, verifica che i notebooks siano sincronizzati:

- [ ] Notebook 01 importa correttamente da `medical_diagnosis.py`
- [ ] Notebook 02 importa correttamente da `movie_recommender.py`
- [ ] Notebook 03 importa correttamente da `hybrid_nlu.py`
- [ ] Notebook 04 importa correttamente da `learning_weights.py`

**Celle di setup Colab** (già presenti nei tuoi notebooks):
- [ ] Celle `!pip install` per LNN, torch, etc
- [ ] Celle per scaricare file .py da GitHub
- [ ] Celle per setup API keys (esempio 03)

### 6. Verifica Documentazione

- [ ] Tutti i link nel README funzionano
- [ ] Link ai file examples/ sono corretti
- [ ] Link a docs/ sono corretti
- [ ] Link esterni (IBM LNN, paper, etc) funzionano
- [ ] Immagini/screenshot se aggiunti sono visibili

### 7. Code Quality

```bash
# Formatting
black examples/ tests/ --check --line-length=100

# Linting
flake8 examples/ tests/ --max-line-length=100
```

- [ ] Codice formattato con black
- [ ] Nessun errore critico da flake8
- [ ] Docstrings presenti per classi/funzioni pubbliche

## Publishing su GitHub

### 8. Crea Repository

1. Vai su [github.com/new](https://github.com/new)
2. Nome repository: `neuro_llm` (o scegli altro)
3. Descrizione: "Tutorial pratico su IBM Logical Neural Networks (LNN) - Neuro-Symbolic AI"
4. Public/Private: Scegli
5. **NON** inizializzare con README (hai già il tuo)

### 9. Push Iniziale

```bash
# Nella directory neuro_llm/

# Inizializza git (se non fatto)
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/neuro_llm.git

# Add files
git add .

# Commit iniziale
git commit -m "Initial commit: IBM LNN Tutorial Examples

- 4 complete examples (medical, recommendation, hybrid, learning)
- Full test suite with pytest
- Comprehensive documentation
- GitHub Actions CI/CD
- Jupyter notebooks with Colab support
- Real-world applications guide

Ready for v1.0.0 release"

# Push
git branch -M main
git push -u origin main
```

### 10. Configurazione Repository GitHub

Una volta pubblicato, vai su Settings del repository:

#### General
- [ ] Aggiungi Topics: `python`, `machine-learning`, `neuro-symbolic-ai`, `logical-neural-networks`, `ibm-lnn`
- [ ] Aggiungi Description breve
- [ ] Abilita Issues
- [ ] Abilita Discussions (opzionale)

#### Actions
- [ ] Abilita GitHub Actions (dovrebbe attivarsi automaticamente)
- [ ] Verifica che workflow `Tests` sia visibile

#### Pages (opzionale)
- [ ] Abilita GitHub Pages per docs/ se vuoi hosting

### 11. Release v1.0.0

Crea prima release ufficiale:

1. Vai su Releases → "Draft a new release"
2. Tag: `v1.0.0`
3. Title: `v1.0.0 - Initial Release 🎉`
4. Description: Copia da CHANGELOG.md sezione [1.0.0]
5. Publish release

- [ ] Release v1.0.0 creata

### 12. Post-Publishing

#### README Badges
I badge ora dovrebbero funzionare automaticamente:
- [ ] Badge Tests mostra status (potrebbe richiedere primo workflow run)
- [ ] Badge License funziona
- [ ] Badge Python version visibile
- [ ] Badge GitHub Stars funziona

#### Test Colab Notebooks
- [ ] Clicca ogni badge Colab nel README
- [ ] Verifica che notebooks si aprono in Colab
- [ ] Test quick run di almeno uno

#### Community
- [ ] Condividi su social (Twitter, LinkedIn, etc)
- [ ] Post su r/MachineLearning, r/artificial
- [ ] Considera submit a Awesome-Neuro-Symbolic-AI list

## Maintenance

### Dopo Pubblicazione

- [ ] Monitor Issues per bug reports
- [ ] Rispondi a PRs entro 1 settimana
- [ ] Aggiorna CHANGELOG.md per ogni modifica
- [ ] Considera creare milestone per future features
- [ ] Setup Dependabot per aggiornamenti dipendenze (opzionale)

### Quarterly Review

Ogni 3-6 mesi:
- [ ] Aggiorna requirements.txt con versioni LNN più recenti
- [ ] Verifica link esterni ancora funzionanti
- [ ] Review issues chiusi per pattern comuni
- [ ] Considera aggiungere esempi richiesti dalla community

---

## Troubleshooting

### GitHub Actions Falliscono

Se tests falliscono su GitHub Actions:
1. Verifica che passino localmente prima
2. Check logs dettagliati in Actions tab
3. Probabilmente API compatibility issues LNN - è normale, workflow ha `continue-on-error: true`

### Badge Non Funzionano

- **Tests badge**: Attendi primo workflow run, poi dovrebbe funzionare
- **Stars badge**: Funziona solo dopo pubblicazione
- **Altri**: Verifica che URL contenga username corretto

### Colab Links Non Funzionano

- Verifica path nel link: `github.com/USERNAME/neuro_llm/blob/main/examples/...`
- Branch deve essere `main` (o `master` se usi quello)
- Files .ipynb devono esistere nel path specificato

---

**Good luck con la pubblicazione! 🚀**

Hai domande? Apri un Issue o contatta su GitHub Discussions.
