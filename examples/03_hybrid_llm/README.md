# Esempio 03: Sistema Ibrido LNN + LLM

Integrazione di **Large Language Models** (Claude) con **Logical Neural Networks** per Natural Language Understanding avanzata.

## Cosa Dimostra

- ✅ Integrazione LLM per estrazione fatti da testo naturale
- ✅ LNN per ragionamento logico strutturato
- ✅ Pipeline end-to-end: Testo → Fatti → Inferenza → Risposta
- ✅ Spiegabilità completa del ragionamento

## Architettura

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT: Testo Naturale                     │
│  "Leonardo da Vinci nacque a Vinci, in Toscana, Italia..."  │
└────────────────────────┬─────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │   LLM (Claude)          │
            │  Fact Extraction        │
            └────────────┬────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Structured Facts (JSON)      │
         │  - BornIn(Leonardo, Vinci)    │
         │  - LocatedIn(Vinci, Toscana)  │
         │  - LocatedIn(Toscana, Italia) │
         └───────────────┬───────────────┘
                         │
            ┌────────────▼────────────┐
            │   LNN Reasoning         │
            │  Logical Inference      │
            │  - Rule: BornIn → Citizen│
            │  - Rule: Transitivity   │
            └────────────┬────────────┘
                         │
         ┌───────────────▼────────────────┐
         │  Inferred Facts                │
         │  - CitizenOf(Leonardo, Vinci)  │
         │  - CitizenOf(Leonardo, Toscana)│
         │  - CitizenOf(Leonardo, Italia) │
         └───────────────┬────────────────┘
                         │
            ┌────────────▼────────────┐
            │   LLM (Claude)          │
            │  Answer Generation      │
            └────────────┬────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              OUTPUT: Risposta Naturale                       │
│  "Leonardo da Vinci è cittadino italiano, poiché nacque      │
│   a Vinci che si trova in Toscana, Italia."                 │
└──────────────────────────────────────────────────────────────┘
```

## Vantaggi del Approccio Ibrido

### LLM Solo vs LNN Solo vs Ibrido

| Capacità | LLM Solo | LNN Solo | **LNN + LLM** |
|----------|----------|----------|---------------|
| Comprensione testo naturale | ✅ | ❌ | ✅ |
| Ragionamento logico rigoroso | ❌ | ✅ | ✅ |
| Spiegabilità | ❌ | ✅ | ✅ |
| Gestione incertezza | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tracciabilità ragionamento | ❌ | ✅ | ✅ |
| Flessibilità linguistica | ✅ | ❌ | ✅ |
| **Hallucinations** | ⚠️ Alto rischio | ✅ Nessuna | ✅ **Minimizzate** |

### Perché Ibrido è Meglio

1. **Anti-Hallucination**: LNN verifica logicamente i fatti estratti dall'LLM
2. **Spiegabilità**: Ogni inferenza è tracciabile tramite regole logiche
3. **Estensibilità**: Facile aggiungere nuove regole di dominio
4. **Robustezza**: Gestisce contraddizioni e incertezza in modo formale

## Setup

### Requisiti

```bash
pip install lnn anthropic
```

### Configurazione API Key

**Opzione 1**: Variabile d'ambiente (consigliata)

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

**Opzione 2**: File di configurazione

```bash
cp config.example.py config.py
# Modifica config.py con la tua API key
```

Ottieni la tua chiave da: [console.anthropic.com](https://console.anthropic.com/)

### Demo Mode (senza API key)

Il sistema funziona anche **senza API key** usando pattern matching semplice per demo:

```bash
# Funziona anche senza ANTHROPIC_API_KEY
python hybrid_nlu.py
```

## Esecuzione

```bash
python examples/03_hybrid_llm/hybrid_nlu.py
```

## Output Esempio

```
======================================================================
SISTEMA IBRIDO LNN + LLM per Natural Language Understanding
======================================================================

Testo di input:
----------------------------------------------------------------------
Leonardo da Vinci nacque a Vinci, un piccolo paese in Toscana, Italia.
Michelangelo Buonarroti nacque a Caprese, sempre in Toscana.
----------------------------------------------------------------------


DOMANDA 1: Di quale paese è cittadino Leonardo da Vinci?
======================================================================
[1/4] Estrazione fatti da testo con LLM...
      Estratti: 4 fatti, 2 persone, 3 luoghi
[2/4] Caricamento fatti in LNN...
[3/4] Inferenza logica...
[4/4] Estrazione fatti inferiti...
      Inferiti: 6 nuovi fatti

[5/4] Generazione risposta...

RISPOSTA:
Leonardo da Vinci è cittadino italiano. Questo si deduce dal fatto che
nacque a Vinci, che si trova in Toscana, che a sua volta si trova in
Italia. Secondo le regole di cittadinanza, chi nasce in un luogo è
cittadino di quel paese e dei suoi territori superiori per transitività.

======================================================================
FATTI INFERITI DA LNN:
======================================================================
✓ Leonardo da Vinci è cittadino/a di Vinci (confidenza: [1.00, 1.00])
✓ Leonardo da Vinci è cittadino/a di Toscana (confidenza: [1.00, 1.00])
✓ Leonardo da Vinci è cittadino/a di Italia (confidenza: [1.00, 1.00])
✓ Vinci si trova in Toscana (confidenza: [1.00, 1.00])
✓ Vinci si trova in Italia (confidenza: [1.00, 1.00])
✓ Toscana si trova in Italia (confidenza: [1.00, 1.00])
```

## Regole Logiche Implementate

### 1. Birth Citizenship

```python
BornIn(x, y) → CitizenOf(x, y)
```

Se una persona nasce in un luogo, è cittadina di quel luogo.

### 2. Residence Citizenship (più debole)

```python
LivesIn(x, y) → CitizenOf(x, y)
```

Se vive in un luogo, potrebbe essere cittadina (confidenza < 1.0).

### 3. Location Transitivity

```python
LocatedIn(x, y) ∧ LocatedIn(y, z) → LocatedIn(x, z)
```

Se X è in Y e Y è in Z, allora X è in Z (es: Vinci in Toscana, Toscana in Italia).

## Casi d'Uso Reali

### 1. Question Answering su Documenti

```python
document = load_pdf("contratto.pdf")
question = "Chi è responsabile per i danni?"

system = HybridNLUSystem()
answer = system.process_text_and_query(document, question)
# Risposta con riferimenti logici tracciabili
```

### 2. Compliance Checking

```python
policy = "Se dipendente ha accesso a dati sensibili, deve avere training."
facts = extract_from_database()

# LNN verifica violazioni
violations = system.check_compliance(policy, facts)
```

### 3. Knowledge Graph Construction

```python
articles = load_news_articles()

for article in articles:
    facts = system.extract_facts_with_llm(article)
    system.add_facts_to_lnn(facts)

# LNN inferisce relazioni implicite
inferred_graph = system.build_knowledge_graph()
```

## Estensioni Possibili

### 1. Multi-Hop Reasoning

Domande che richiedono inferenze su più step:

```
Q: "Leonardo e Michelangelo erano connazionali?"
→ LNN inferisce cittadinanza di entrambi
→ Confronta CitizenOf(Leonardo, X) con CitizenOf(Michelangelo, X)
→ Risponde: "Sì, entrambi italiani"
```

### 2. Temporal Reasoning

Aggiungere predicati temporali:

```python
BornIn = Predicate('BornIn', arity=3)  # (persona, luogo, anno)
# Regola: Se nato prima del 1900 → non è vivo ora
```

### 3. Confidence Propagation

```python
# LLM estrae con confidence
facts_with_conf = extract_facts(text)
# {"fact": BornIn(x, y), "confidence": [0.7, 0.9]}

# LNN propaga incertezza attraverso inferenze
inferred = lnn.infer()
# CitizenOf eredita bounds [0.7, 0.9]
```

### 4. Contradiction Detection

```python
# Fatto 1: BornIn(Leonardo, Vinci)
# Fatto 2: BornIn(Leonardo, Milano)

# LNN rileva contraddizione
contradictions = system.detect_contradictions()
# → Richiede disambiguazione all'utente
```

## Performance

- **Estrazione LLM**: ~2-5s per 500 parole
- **Inferenza LNN**: <100ms per 100 fatti
- **Total latency**: ~3-6s end-to-end

## Limitazioni

1. **API Cost**: Chiamate LLM hanno costo (usa cache per ottimizzare)
2. **Qualità estrazione**: Dipende da prompt engineering
3. **Dominio specifico**: Regole LNN vanno customizzate per dominio

## Best Practices

### 1. Prompt Engineering per Estrazione

```python
# ✅ BUONO: Prompt strutturato con esempi
prompt = f"""
Estrai fatti da: {text}

Formato JSON:
{{"facts": [{{"predicate": "...", "args": [...]}}]}}

Esempi:
Input: "Mario lavora per IBM"
Output: {{"facts": [{{"predicate": "WorksFor", "args": ["Mario", "IBM"]}}]}}
"""

# ❌ CATTIVO: Prompt vago
prompt = f"Trova fatti in: {text}"
```

### 2. Validazione Fatti

```python
# Valida fatti estratti prima di caricare in LNN
for fact in extracted_facts:
    if not validate_fact_schema(fact):
        log_error(fact)
        continue
    system.add_facts_to_lnn([fact])
```

### 3. Caching

```python
# Cache risultati LLM per testi ripetuti
@lru_cache(maxsize=1000)
def extract_facts_cached(text_hash):
    return system.extract_facts_with_llm(text)
```

## Riferimenti

- [LNN Paper](https://arxiv.org/abs/2006.13155)
- [Neuro-Symbolic AI Survey](https://arxiv.org/abs/2305.00813)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
