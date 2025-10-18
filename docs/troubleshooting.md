# Troubleshooting

Guida alla risoluzione dei problemi comuni durante installazione e utilizzo di LNN.

## Problemi di Installazione

### LNN non si installa

**Errore**: `"error: metadata-generation-failed"`

**Soluzione**:
```bash
# Aggiorna pip e setuptools
pip install --upgrade pip setuptools wheel

# Riprova installazione
pip install lnn
```

### PyTorch troppo grande

**Errore**: `"Download too large"`

**Soluzione**: Installa versione CPU-only
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Import Error su macOS

**Errore**: `"ImportError: cannot import name 'BinaryPredicate'"`

**Soluzione**:
```bash
# Reinstalla LNN
pip uninstall lnn
pip install --no-cache-dir lnn
```

### Test falliscono

**Errore**: `"ModuleNotFoundError"`

**Soluzione**:
```bash
# Verifica path e reinstalla
pip install -e .
pytest tests/ -v
```

### CUDA non disponibile

**Warning**: `"CUDA not available, using CPU"`

**Nota**: Questo è OK! LNN funziona su CPU.

**Per abilitare GPU** (se hai NVIDIA):
```bash
# 1. Verifica driver CUDA
nvidia-smi

# 2. Reinstalla PyTorch con CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Verifica Installazione Completa

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

## Problemi con API Keys

### API Key Anthropic non trovata

**Per Colab**: Verifica che il secret `ANTHROPIC_API_KEY` sia configurato correttamente.

**Per locale**: Verifica variabile d'ambiente:
```bash
echo $ANTHROPIC_API_KEY
```

### Rate limiting API

**Errore**: `"rate_limit_exceeded"`

**Soluzione**: Aspetta qualche minuto, Anthropic ha limiti di richieste.

## Problemi di Performance

### Notebook lento

**Su Colab**:
- Runtime → Restart runtime
- Riduci dimensione dataset negli esempi
- Usa meno epoche di training

**Su locale**:
- Verifica RAM disponibile
- Chiudi altri programmi
- Usa versione CPU di PyTorch se GPU è lenta

### Out of Memory

**Errore**: Runtime crash

**Soluzioni**:
1. Riduci batch size negli esempi
2. Usa dataset più piccoli
3. Riavvia runtime/kernel

## Problemi con Notebooks

### Modulo non trovato

**Errore**: `"ModuleNotFoundError"` nei notebook

**Soluzione**:
```python
# Riavvia kernel
# Runtime → Restart runtime (Colab)
# Oppure ricarica moduli
import importlib
import medical_diagnosis
importlib.reload(medical_diagnosis)
```

### Celle non eseguono

**Su Colab**: Verifica connessione internet, a volte Colab si disconnette.

## Problemi di Compatibilità

### Versioni Python incompatibili

LNN richiede Python 3.8+. Verifica versione:
```bash
python --version
```

### Dipendenze in conflitto

Se hai problemi con versioni, usa virtual environment dedicato.

## Debug Avanzato

### Log dettagliati

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ora esegui il codice per vedere log dettagliati
```

### Test isolati

```bash
# Testa solo un modulo
python -c "import lnn; print('LNN OK')"

# Testa PyTorch
python -c "import torch; print(torch.__version__)"
```

## Supporto

### Documentazione Ufficiale
- [IBM LNN Docs](https://ibm.github.io/LNN/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Anthropic API Docs](https://docs.anthropic.com/)

### Community
- [GitHub Issues LNN](https://github.com/IBM/LNN/issues)
- [Questo repository](https://github.com/gianlucamazza/neuro_llm/issues)

---

**Ricorda**: La maggior parte dei problemi si risolve con un ambiente virtuale pulito e seguendo la [guida installazione](installation.md).</content>
</xai:function_call<docs/troubleshooting.md