# Guida Installazione

Guida completa per installare IBM LNN e le dipendenze necessarie per gli esempi.

## Requisiti di Sistema

### Minimi

- **Python**: 3.8 o superiore (consigliato 3.9+)
- **RAM**: 4GB minimo (8GB consigliato)
- **Spazio disco**: 2GB
- **OS**: macOS, Linux, Windows (con WSL consigliato)

### Consigliati

- **Python**: 3.10 o 3.11
- **RAM**: 16GB (per training con dataset grandi)
- **GPU**: Opzionale, ma accelera training

## Metodo 1: Installazione Standard (Consigliata)

### Step 1: Crea Ambiente Virtuale

```bash
# Naviga alla directory del progetto
cd /path/to/neuro_llm

# Crea virtual environment
python -m venv venv

# Attiva (macOS/Linux)
source venv/bin/activate

# Attiva (Windows)
venv\Scripts\activate
```

### Step 2: Installa Dipendenze

```bash
# Installa tutte le dipendenze
pip install -r requirements.txt
```

Questo installerà:
- `lnn` (IBM Logical Neural Networks)
- `torch` (PyTorch per training)
- `anthropic` (per esempio hybrid LLM)
- `pytest` (per testing)
- `matplotlib` (per grafici)

### Step 3: Verifica Installazione

```bash
# Testa import LNN
python -c "import lnn; print(f'LNN version: {lnn.__version__}')"

# Testa PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Esegui test
pytest tests/ -v
```

## Metodo 2: Installazione con Conda

Se preferisci Conda:

```bash
# Crea environment
conda create -n lnn_env python=3.10 -y

# Attiva
conda activate lnn_env

# Installa PyTorch (con CUDA se hai GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Installa altre dipendenze
pip install lnn anthropic pytest matplotlib
```

## Metodo 3: Installazione LNN da Sorgente

Per versione latest o sviluppo:

```bash
# Clone repository IBM LNN
git clone https://github.com/IBM/LNN.git
cd LNN

# Installa in modalità editable
pip install -e .

# Torna al nostro progetto
cd /path/to/neuro_llm

# Installa altre dipendenze
pip install anthropic pytest matplotlib torch
```

## Configurazione API Keys

### Per Esempio Hybrid (LNN + LLM)

L'esempio 03 richiede Anthropic API key (opzionale).

#### Metodo 1: Variabile d'Ambiente (Consigliata)

```bash
# Aggiungi al tuo .bashrc o .zshrc
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc

# Oppure set temporaneo
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Metodo 2: File di Configurazione

```bash
# Copia template
cp examples/03_hybrid_llm/config.example.py examples/03_hybrid_llm/config.py

# Modifica con la tua key
nano examples/03_hybrid_llm/config.py
```

**IMPORTANTE**: `config.py` è in `.gitignore`, non sarà committato.

#### Ottieni API Key

1. Vai su [console.anthropic.com](https://console.anthropic.com/)
2. Crea account o login
3. Vai su "API Keys"
4. Genera nuova key
5. Copia e salva (mostrata solo una volta!)

**Nota**: L'esempio 03 funziona anche **senza API key** in demo mode.

## Configurazioni Avanzate

### Per Sviluppo

```bash
# Installa dipendenze extra per sviluppo
pip install -r requirements.txt
pip install black flake8 mypy ipython jupyter

# Setup pre-commit hooks (opzionale)
pip install pre-commit
pre-commit install
```

### Per Production

```bash
# Installa solo dipendenze necessarie (no test/dev)
pip install lnn torch anthropic

# Opzionale: freeze versioni
pip freeze > requirements-prod.txt
```

### Per Jupyter Notebooks

```bash
# Installa jupyter
pip install jupyter ipykernel

# Aggiungi kernel
python -m ipykernel install --user --name=lnn_env --display-name="LNN Environment"

# Avvia notebook
jupyter notebook
```

## Aggiornamento e Manutenzione

### Aggiorna Tutte le Dipendenze

```bash
pip install --upgrade -r requirements.txt
```

### Aggiorna Solo LNN

```bash
pip install --upgrade lnn
```

### Aggiorna da GitHub (latest)

```bash
pip install --upgrade git+https://github.com/IBM/LNN.git
```

### Disinstallazione

```bash
# Disattiva environment
deactivate

# Rimuovi environment
rm -rf venv/

# Opzionale: rimuovi cache pip
pip cache purge
```

## Risorse Aggiuntive

- [Documentazione LNN Ufficiale](https://ibm.github.io/LNN/)

## Verifica Completa

Esegui questo script per verificare tutto:

```python
# save as verify_installation.py

import sys

def check_import(module_name):
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {module_name:15s} - version {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:15s} - NOT INSTALLED")
        return False

def main():
    print("="*50)
    print("Verifying Installation")
    print("="*50)

    modules = [
        'lnn',
        'torch',
        'anthropic',
        'pytest',
        'matplotlib',
        'numpy'
    ]

    results = [check_import(m) for m in modules]

    print("\n" + "="*50)
    if all(results):
        print("✅ All dependencies installed correctly!")
    else:
        print("⚠️  Some dependencies missing. See above.")
    print("="*50)

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("\n💻 Running on CPU (CUDA not available)")
    except:
        pass

if __name__ == "__main__":
    main()
```

Esegui:

```bash
python verify_installation.py
```

