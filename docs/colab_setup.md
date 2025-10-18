# Guida Google Colab per neuro_llm

Questa guida completa ti aiuta a utilizzare i notebook del progetto **neuro_llm** su Google Colab.

## Indice

- [Introduzione](#introduzione)
- [Come Aprire i Notebook](#come-aprire-i-notebook)
- [Configurazione API Keys](#configurazione-api-keys)
- [Esecuzione dei Notebook](#esecuzione-dei-notebook)
- [Troubleshooting](#troubleshooting)
- [Best Practice 2025](#best-practice-2025)
- [FAQ](#faq)

---

## Introduzione

### Cos'è Google Colab?

Google Colab è un ambiente Jupyter notebook gratuito che viene eseguito nel cloud. Offre:

- **Nessuna installazione richiesta** - tutto funziona nel browser
- **Accesso a GPU/TPU gratuite** (con limiti)
- **Integrazione con Google Drive**
- **Collaborazione in tempo reale**
- **Ambiente Python preconfigurato**

### Perché usare Colab per neuro_llm?

- Esecuzione immediata senza setup locale
- Ambiente consistente per tutti gli utenti
- Condivisione facile di esperimenti
- Accesso a risorse computazionali gratuite

---

## Architettura del Codice (DRY Principle)

I notebook di questo progetto seguono le **best practice DRY (Don't Repeat Yourself)** per evitare duplicazione del codice.

### Come Funziona

**File `.py` = Fonte di verità**
- Contengono le classi e funzioni riutilizzabili
- Esempio: `medical_diagnosis.py` contiene `MedicalDiagnosisSystem`
- Possono essere eseguiti direttamente per un quick-test
- Possono essere importati da altri script Python

**File `.ipynb` = Demo interattiva**
- Import automatico delle classi dai file `.py`
- Su Colab: download automatico dei `.py` da GitHub
- Celle educative con spiegazioni e analisi
- Esperienza di apprendimento guidata

### Perché Questo Approccio?

1. **Manutenibilità**: Modifiche in un solo posto (il file `.py`)
2. **Testing**: Classi isolate sono più facili da testare
3. **Riusabilità**: Codice può essere importato in altri progetti
4. **Git-friendly**: File più piccoli, diff più chiari
5. **Best practice**: Raccomandato da Google Cloud e Jupyter community

### Su Google Colab

Non devi fare nulla! Il notebook scarica automaticamente i file `.py` necessari:

```python
# Questo accade automaticamente nella prima cella
if IN_COLAB:
    import urllib.request
    url = "https://raw.githubusercontent.com/user/repo/file.py"
    urllib.request.urlretrieve(url, "file.py")
```

Poi puoi importare normalmente:

```python
from medical_diagnosis import MedicalDiagnosisSystem
system = MedicalDiagnosisSystem()
```

### In Locale

Se esegui i notebook in locale, i file `.py` sono già presenti nella stessa cartella, quindi l'import funziona immediatamente.

### Per Sviluppatori

Se vuoi modificare il codice:

1. **Edita il file `.py`** (esempio: `medical_diagnosis.py`)
2. Il notebook importerà automaticamente la versione aggiornata
3. Per notebook in esecuzione: riavvia il kernel per ricaricare il modulo

```python
# Oppure forza il reload senza riavviare
import importlib
import medical_diagnosis
importlib.reload(medical_diagnosis)
```

---

## Come Aprire i Notebook

### Metodo 1: Badge "Open in Colab" (Più Facile)

1. Vai alla [pagina principale del repository](https://github.com/gianlucamazza/neuro_llm)
2. Nella sezione "Quick Start" trova la tabella degli esempi
3. Clicca sul badge "Open in Colab" per l'esempio che ti interessa
4. Il notebook si aprirà direttamente in Google Colab

### Metodo 2: URL Diretto

Puoi aprire qualsiasi notebook usando l'URL pattern:

```
https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/[PATH]/[NOTEBOOK].ipynb
```

Esempi:
- Medical: `https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/01_medical_expert/medical_diagnosis.ipynb`
- Recommender: `https://colab.research.google.com/github/gianlucamazza/neuro_llm/blob/main/examples/02_recommendation/movie_recommender.ipynb`

### Metodo 3: Da Google Colab

1. Vai su [colab.research.google.com](https://colab.research.google.com)
2. Clicca su "File" → "Open notebook"
3. Seleziona la tab "GitHub"
4. Inserisci: `gianlucamazza/neuro_llm`
5. Seleziona il notebook desiderato

---

## Configurazione API Keys

### Solo per: Esempio 03 (Sistema Ibrido LNN+LLM)

L'esempio 03 utilizza l'API di Anthropic (Claude) per dimostrare l'integrazione LNN+LLM. Gli altri esempi NON richiedono API keys.

### Opzione A: Con API Key (Funzionalità Completa)

#### Passo 1: Ottieni una API Key Anthropic

1. Vai su [console.anthropic.com](https://console.anthropic.com)
2. Crea un account o effettua il login
3. Vai su "API Keys"
4. Crea una nuova API key
5. **Copia la chiave** (inizia con `sk-ant-...`)

#### Passo 2: Configura Colab Secrets (Best Practice 2025)

1. **Apri il notebook** 03_hybrid_nlu.ipynb su Colab

2. **Trova l'icona della chiave** nella barra laterale sinistra:
   ```
   📁 Files
   🔍 Search
   🔑 Secrets  ← Clicca qui
   💾 Variables
   ```

3. **Aggiungi un nuovo secret**:
   - Clicca su "+ Add new secret"
   - **Name**: `ANTHROPIC_API_KEY`
   - **Value**: Incolla la tua API key (`sk-ant-...`)
   - Clicca "Save"

4. **Abilita l'accesso** per il notebook:
   - Trova il secret appena creato
   - Attiva il toggle switch "Notebook access"

5. **Esegui il notebook**:
   - Il codice carica automaticamente la chiave con:
     ```python
     from google.colab import userdata
     api_key = userdata.get('ANTHROPIC_API_KEY')
     ```

### Opzione B: Senza API Key (Demo Mode)

Se non hai una API key, il notebook 03 funziona comunque in **DEMO MODE**:

- Usa fatti predefiniti invece di estrarre da testo reale
- Mostra comunque l'inferenza LNN
- Perfetto per capire l'architettura del sistema

Semplicemente:
1. Apri il notebook
2. Esegui tutte le celle
3. Vedrai un warning che conferma il demo mode

---

## Esecuzione dei Notebook

### Setup Automatico

Ogni notebook include una cella di setup che installa automaticamente le dipendenze:

```python
!pip install -q lnn>=1.2.0 torch>=2.0.0 numpy>=1.24.0
```

Il flag `-q` (quiet) mantiene l'output pulito.

### Esecuzione Celle

#### Opzione 1: Esegui tutto
- Click su "Runtime" → "Run all"
- Oppure: `Ctrl+F9` (Windows/Linux) / `Cmd+F9` (Mac)

#### Opzione 2: Cella per cella
- Click sul bottone ▶️ a sinistra di ogni cella
- Oppure: `Shift+Enter` per eseguire e passare alla successiva

### Tempi di Esecuzione Tipici

| Notebook | Tempo CPU | Note |
|----------|-----------|------|
| 01. Medical Diagnosis | ~30 sec | Setup + inferenza |
| 02. Movie Recommender | ~45 sec | Più dati, più inferenze |
| 03. Hybrid LNN+LLM | ~60 sec | Dipende da API latency |
| 04. Learning Weights | ~2-3 min | Include training loop |

### Salvataggio Risultati

Colab salva automaticamente i notebook su Google Drive. Per scaricare:

1. "File" → "Download" → ".ipynb"
2. Oppure salva output/grafici:
   ```python
   from google.colab import files
   files.download('learning_curve.png')
   ```

---

## Troubleshooting

### Problema: "Package installation failed"

**Causa**: Timeout di rete o versione incompatibile

**Soluzione**:
```python
# Riprova con timeout più lungo
!pip install --timeout=100 lnn>=1.2.0

# Oppure installa versione specifica
!pip install lnn==1.2.0 torch==2.0.0
```

### Problema: "Runtime crashed" o "Out of Memory"

**Causa**: Consumo eccessivo di RAM

**Soluzione**:
1. "Runtime" → "Restart runtime"
2. Riduci dimensione dataset:
   ```python
   # Invece di 100 epoche
   learner.train(epochs=50, learning_rate=0.01)
   ```

### Problema: "API Key not found" (Notebook 03)

**Causa**: Secret non configurato correttamente

**Soluzione**:
1. Verifica che il secret si chiami esattamente `ANTHROPIC_API_KEY` (case-sensitive)
2. Controlla che "Notebook access" sia abilitato (toggle verde)
3. Riavvia runtime: "Runtime" → "Restart runtime"
4. Riesegui la cella che carica l'API key

### Problema: "Module not found" dopo installazione

**Causa**: Runtime non aggiornato dopo install

**Soluzione**:
```python
# Forza reload del modulo
import importlib
import sys
if 'lnn' in sys.modules:
    importlib.reload(sys.modules['lnn'])
```

### Problema: Notebook lento o si blocca

**Causa**: Risorse gratuite Colab limitate

**Soluzioni**:
1. **Riduci carico computazionale**:
   ```python
   # Meno epoche di training
   losses = learner.train(epochs=50)

   # Dataset più piccolo
   # Rimuovi alcuni test cases
   ```

2. **Usa GPU** (se disponibile):
   - "Runtime" → "Change runtime type"
   - "Hardware accelerator" → "GPU"
   - **Nota**: Non sempre necessario per LNN

3. **Restart & Clear**:
   - "Runtime" → "Restart and run all"

### Problema: "403 Forbidden" su GitHub

**Causa**: Rate limiting di GitHub

**Soluzione**:
1. Aspetta 5 minuti
2. Oppure clona manualmente:
   ```python
   !git clone https://github.com/gianlucamazza/neuro_llm.git
   %cd neuro_llm
   ```

---

## Best Practice 2025

Questi notebook seguono le **moderne best practice 2025** per Google Colab:

### 1. Secrets Manager per API Keys

**Cosa facciamo**:
```python
from google.colab import userdata
api_key = userdata.get('ANTHROPIC_API_KEY')
```

**Perché è meglio**:
- Nessun hardcoding di secrets nel codice
- Keys crittografate da Google
- Non incluse quando condividi il notebook
- Conforme agli standard di sicurezza 2025

**Da evitare** (vecchio metodo):
```python
# ❌ MAI fare questo
api_key = "sk-ant-..."
os.environ['API_KEY'] = "..."
```

### 2. Installazione Silenziosa

**Cosa facciamo**:
```python
!pip install -q lnn>=1.2.0  # -q flag
```

**Perché è meglio**:
- Output pulito e leggibile
- Mostra solo errori importanti
- Esperienza utente professionale

### 3. Markdown Cells Educative

**Cosa facciamo**:
- Spiegazioni chiare prima di ogni sezione
- Link a risorse esterne
- Tabelle comparative
- Note e avvisi

**Perché è meglio**:
- Self-contained learning
- Riduce confusione
- Professional documentation

### 4. Gestione Errori Graceful

**Cosa facciamo**:
```python
try:
    from google.colab import userdata
    api_key = userdata.get('ANTHROPIC_API_KEY')
except:
    print("Demo mode activated")
    api_key = None
```

**Perché è meglio**:
- Notebook funziona anche senza API
- Messaggi chiari all'utente
- Nessun crash inaspettato

### 5. Pinning Versioni

**Cosa facciamo**:
```python
!pip install -q lnn>=1.2.0 torch>=2.0.0
```

**Perché è meglio**:
- Riproducibilità garantita
- Evita breaking changes
- Compatibilità verificata

---

## FAQ

### D: Posso usare Colab gratuitamente?

**R**: Sì! Colab offre un tier gratuito con:
- ~12 GB RAM
- Accesso limitato a GPU/TPU
- Timeout sessione dopo inattività
- Perfetto per questi notebook educativi

### D: I miei dati sono sicuri su Colab?

**R**: Sì, se usi correttamente i Secrets:
- API keys sono crittografate
- Non visibili ad altri utenti
- Non incluse quando condividi il notebook

**Ma attenzione**:
- Non hardcodare secrets nel codice
- Non stampare secrets nell'output

### D: Posso eseguire i notebook offline?

**R**: No, Colab richiede connessione internet.

**Alternativa**:
- Installa Jupyter localmente
- Usa i file `.ipynb` scaricati
- Oppure usa gli script `.py` originali

### D: Cosa succede se supero i limiti gratuiti?

**R**:
- **Caso 1 - RAM**: Runtime si riavvia, perdi variabili
- **Caso 2 - Tempo**: Sessione disconnette dopo ~12h
- **Caso 3 - GPU**: Accesso temporaneamente limitato

**Soluzioni**:
- Riduci dimensione dataset
- Usa meno epoche di training
- Fai pause tra esecuzioni

### D: Posso collaborare in tempo reale?

**R**: Sì! Come Google Docs:
1. Salva il notebook su Google Drive
2. "Share" → Aggiungi collaboratori
3. Modifiche visibili in tempo reale

### D: I notebook funzionano anche su Jupyter locale?

**R**: Sì, ma con modifiche minori:

**Differenze**:
- Rimuovi installazione pacchetti (già installati)
- Per API keys: usa `.env` invece di Colab Secrets:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  api_key = os.getenv('ANTHROPIC_API_KEY')
  ```

### D: Come cito questo progetto?

**R**:
```
Mazza, G. (2025). neuro_llm: Tutorial Pratico su IBM Logical Neural Networks.
GitHub repository: https://github.com/gianlucamazza/neuro_llm
```

---

## Risorse Utili

### Documentazione Ufficiale

- [Google Colab Guide](https://colab.research.google.com/notebooks/welcome.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [IBM LNN Documentation](https://ibm.github.io/LNN/)
- [Anthropic API Docs](https://docs.anthropic.com)

### Repository GitHub

- [Progetto neuro_llm](https://github.com/gianlucamazza/neuro_llm)
- [IBM LNN Official](https://github.com/IBM/LNN)

### Community

- [Discussions](https://github.com/gianlucamazza/neuro_llm/discussions)
- [Issues](https://github.com/gianlucamazza/neuro_llm/issues)

---

## Contribuisci

Hai trovato un bug o hai suggerimenti per migliorare l'esperienza Colab?

1. Apri una [Issue](https://github.com/gianlucamazza/neuro_llm/issues)
2. Proponi miglioramenti via Pull Request
3. Condividi il tuo feedback nelle [Discussions](https://github.com/gianlucamazza/neuro_llm/discussions)

---

**Ultima revisione**: Ottobre 2025
**Versione**: 1.0
**Compatibilità**: Google Colab (ambiente standard)
