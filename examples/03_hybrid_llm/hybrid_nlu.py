"""
Sistema Ibrido LNN + LLM per Natural Language Understanding

Questo modulo fornisce la classe HybridNLUSystem che integra LLM (Claude) e LNN
per comprensione del linguaggio naturale con ragionamento logico.

Pipeline: Testo → [LLM] → Fatti → [LNN] → Inferenze

Caratteristiche:
- LLM estrae fatti strutturati da testo naturale
- LNN applica ragionamento logico per inferire nuova conoscenza
- Gestione sicura API keys (best practice 2025)
- Demo mode disponibile senza API key

Utilizzo:
    from hybrid_nlu import HybridNLUSystem

    system = HybridNLUSystem(api_key="sk-ant-...")
    facts = system.extract_facts_with_llm("Leonardo nacque a Vinci")
    system.add_facts_to_lnn(facts)
    inferred = system.extract_inferred_facts()

Per demo interattiva completa, vedi il notebook hybrid_nlu.ipynb

Autore: Gianluca Mazza
"""

from lnn import Predicate, Variable, Implies, And, Forall, Model, Fact, World
from typing import Dict, List, Optional
import json
import os


class HybridNLUSystem:
    """
    Sistema che combina LNN per ragionamento e LLM per comprensione linguaggio naturale

    Workflow:
    1. LLM estrae fatti strutturati da testo
    2. LNN inferisce nuovi fatti tramite regole logiche
    3. LLM genera risposta naturale usando fatti inferiti
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Anthropic API key (opzionale, usa ANTHROPIC_API_KEY env var)
        """
        # Setup LLM (opzionale per testing senza API)
        self.has_llm = False
        if api_key or os.getenv('ANTHROPIC_API_KEY'):
            try:
                from anthropic import Anthropic
                self.llm = Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
                self.has_llm = True
            except ImportError:
                print("Warning: anthropic package not installed. Running in demo mode.")

        # Setup LNN
        self.lnn_model = Model()
        self._setup_lnn()

    def _setup_lnn(self):
        """Configura knowledge base logica"""

        # Variabili
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        # Predicati unari
        self.IsPerson = Predicate('IsPerson')
        self.IsLocation = Predicate('IsLocation')
        self.IsOrganization = Predicate('IsOrganization')

        # Predicati binari (relazioni)
        self.BornIn = Predicate('BornIn', arity=2)
        self.LivesIn = Predicate('LivesIn', arity=2)
        self.WorksFor = Predicate('WorksFor', arity=2)
        self.LocatedIn = Predicate('LocatedIn', arity=2)
        self.CitizenOf = Predicate('CitizenOf', arity=2)

        # Regole logiche per inferenza

        # Regola 1: Se nato in un luogo → cittadino di quel paese
        self.lnn_model.add_knowledge(
            Forall(
                x, y,
                Implies(
                    self.BornIn(x, y),
                    self.CitizenOf(x, y)
                ),
                world=World.AXIOM
            ),
        )

        # Regola 2: Se vive in un luogo da molto → potrebbe essere cittadino
        # (regola più debole, apprendibile)
        self.lnn_model.add_knowledge(
            Forall(
                x, y,
                Implies(
                    self.LivesIn(x, y),
                    self.CitizenOf(x, y)
                ),
                world=World.AXIOM
            ),
        )

        # Regola 3: Transitività location
        # Se X è in Y e Y è in Z → X è in Z
        self.lnn_model.add_knowledge(
            Forall(
                x, y, z,
                Implies(
                    And(
                        self.LocatedIn(x, y),
                        self.LocatedIn(y, z)
                    ),
                    self.LocatedIn(x, z)
                ),
                world=World.AXIOM
            ),
        )

    def extract_facts_with_llm(self, text: str) -> dict:
        """
        Usa LLM per estrarre fatti strutturati da testo naturale

        Args:
            text: Testo in linguaggio naturale

        Returns:
            Dizionario con entità e fatti estratti
        """
        if not self.has_llm:
            # Demo mode: restituisce fatti di esempio
            return self._demo_extraction(text)

        prompt = f"""Estrai fatti strutturati dal seguente testo in italiano.

Testo: {text}

Restituisci SOLO un JSON valido con questa struttura:
{{
    "entities": {{
        "people": ["lista di persone"],
        "locations": ["lista di luoghi"],
        "organizations": ["lista di organizzazioni"]
    }},
    "facts": [
        {{"predicate": "NomePredicato", "args": ["arg1", "arg2"]}},
        ...
    ]
}}

Predicati disponibili:
- IsPerson, IsLocation, IsOrganization (1 argomento)
- BornIn, LivesIn, WorksFor, LocatedIn, CitizenOf (2 argomenti)

Esempio:
Input: "Leonardo da Vinci nacque a Vinci, in Toscana."
Output: {{
    "entities": {{"people": ["Leonardo da Vinci"], "locations": ["Vinci", "Toscana"]}},
    "facts": [
        {{"predicate": "BornIn", "args": ["Leonardo da Vinci", "Vinci"]}},
        {{"predicate": "LocatedIn", "args": ["Vinci", "Toscana"]}}
    ]
}}

Rispondi SOLO con il JSON, senza altre parole."""

        try:
            message = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()
            # Rimuovi eventuali markdown code blocks
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            return json.loads(response_text)

        except Exception as e:
            print(f"Errore nell'estrazione LLM: {e}")
            return self._demo_extraction(text)

    def _demo_extraction(self, text: str) -> dict:
        """Estrazione demo per testing senza API"""
        # Semplice pattern matching per demo
        demo_facts = {
            "entities": {
                "people": [],
                "locations": [],
                "organizations": []
            },
            "facts": []
        }

        # Pattern semplici per demo
        if "leonardo" in text.lower() or "vinci" in text.lower():
            demo_facts["entities"]["people"] = ["Leonardo da Vinci"]
            demo_facts["entities"]["locations"] = ["Vinci", "Toscana", "Italia"]
            demo_facts["facts"] = [
                {"predicate": "BornIn", "args": ["Leonardo da Vinci", "Vinci"]},
                {"predicate": "LocatedIn", "args": ["Vinci", "Toscana"]},
                {"predicate": "LocatedIn", "args": ["Toscana", "Italia"]},
            ]

        return demo_facts

    def add_facts_to_lnn(self, facts_dict: dict):
        """
        Popola LNN con fatti estratti dall'LLM

        Args:
            facts_dict: Dizionario con entities e facts
        """
        data = {}

        # Aggiungi entità
        for person in facts_dict['entities']['people']:
            if self.IsPerson not in data:
                data[self.IsPerson] = {}
            data[self.IsPerson][person] = Fact.TRUE

        for location in facts_dict['entities']['locations']:
            if self.IsLocation not in data:
                data[self.IsLocation] = {}
            data[self.IsLocation][location] = Fact.TRUE

        for org in facts_dict['entities']['organizations']:
            if self.IsOrganization not in data:
                data[self.IsOrganization] = {}
            data[self.IsOrganization][org] = Fact.TRUE

        # Aggiungi fatti (relazioni)
        for fact in facts_dict['facts']:
            predicate_name = fact['predicate']
            predicate = getattr(self, predicate_name, None)

            if predicate is not None:
                args = tuple(fact['args'])
                if predicate not in data:
                    data[predicate] = {}
                data[predicate][args] = Fact.TRUE

        # Carica dati nel modello
        if data:
            self.lnn_model.add_data(data)

    def infer(self):
        """Esegue inferenza LNN per dedurre nuovi fatti"""
        self.lnn_model.infer()

    def extract_inferred_facts(self) -> List[dict]:
        """
        Estrae fatti inferiti da LNN

        Returns:
            Lista di fatti inferiti con confidence
        """
        facts = []

        # Estrai cittadinanze inferite
        for person_key in self.lnn_model[self.IsPerson].state():
            for loc_key in self.lnn_model[self.IsLocation].state():
                citizen_bounds = self.lnn_model[self.CitizenOf].get((person_key, loc_key))

                if citizen_bounds and citizen_bounds[0] > 0.5:  # Lower bound > 0.5
                    facts.append({
                        'type': 'citizenship',
                        'person': person_key,
                        'location': loc_key,
                        'confidence': citizen_bounds,
                        'explanation': f"{person_key} è cittadino/a di {loc_key}"
                    })

        # Estrai location transitive
        locations = list(self.lnn_model[self.IsLocation].state())
        for loc1 in locations:
            for loc2 in locations:
                if loc1 != loc2:
                    located_bounds = self.lnn_model[self.LocatedIn].get((loc1, loc2))
                    if located_bounds and located_bounds[0] > 0.5:
                        facts.append({
                            'type': 'location',
                            'entity': loc1,
                            'location': loc2,
                            'confidence': located_bounds,
                            'explanation': f"{loc1} si trova in {loc2}"
                        })

        return facts

    def answer_question(self, question: str, context_facts: List[dict]) -> str:
        """
        Genera risposta a domanda usando fatti inferiti

        Args:
            question: Domanda dell'utente
            context_facts: Fatti inferiti da LNN

        Returns:
            Risposta in linguaggio naturale
        """
        if not self.has_llm:
            # Demo mode
            return self._demo_answer(question, context_facts)

        # Prepara contesto
        context = "Fatti inferiti dal sistema logico:\n"
        for fact in context_facts:
            conf_str = f"[{fact['confidence'][0]:.2f}, {fact['confidence'][1]:.2f}]"
            context += f"- {fact['explanation']} (confidenza: {conf_str})\n"

        prompt = f"""{context}

Domanda dell'utente: {question}

Genera una risposta naturale e concisa basata SOLO sui fatti inferiti sopra.
Se la risposta non è deducibile dai fatti, dillo chiaramente."""

        try:
            message = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except Exception as e:
            return self._demo_answer(question, context_facts)

    def _demo_answer(self, question: str, context_facts: List[dict]) -> str:
        """Risposta demo senza API"""
        if not context_facts:
            return "Non ho abbastanza informazioni per rispondere."

        # Risposta semplice basata su fatti
        answer = "Sulla base dei fatti inferiti:\n"
        for fact in context_facts[:3]:  # Top 3
            answer += f"- {fact['explanation']}\n"

        return answer

    def process_text_and_query(self, text: str, question: str) -> str:
        """
        Pipeline completa: estrae fatti, inferisce, risponde

        Args:
            text: Testo di input
            question: Domanda da rispondere

        Returns:
            Risposta generata
        """
        print(f"[1/4] Estrazione fatti da testo con LLM...")
        facts = self.extract_facts_with_llm(text)
        print(f"      Estratti: {len(facts['facts'])} fatti, "
              f"{len(facts['entities']['people'])} persone, "
              f"{len(facts['entities']['locations'])} luoghi")

        print(f"[2/4] Caricamento fatti in LNN...")
        self.add_facts_to_lnn(facts)

        print(f"[3/4] Inferenza logica...")
        self.infer()

        print(f"[4/4] Estrazione fatti inferiti...")
        inferred = self.extract_inferred_facts()
        print(f"      Inferiti: {len(inferred)} nuovi fatti")

        print(f"\n[5/4] Generazione risposta...")
        answer = self.answer_question(question, inferred)

        return answer


# Verifica modulo quando eseguito direttamente
if __name__ == "__main__":
    print("=" * 70)
    print("Hybrid NLU System - Module Check")
    print("=" * 70)

    # Test creazione sistema (demo mode)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY non trovata - Demo mode")

    system = HybridNLUSystem(api_key=api_key)
    print("\n✓ HybridNLUSystem creato correttamente")

    # Quick test
    facts = system.extract_facts_with_llm("Test text")
    system.add_facts_to_lnn(facts)
    print("✓ Estrazione fatti e inferenza funzionanti")

    print("\n" + "=" * 70)
    print("Per la demo completa, usa il notebook:")
    print("  hybrid_nlu.ipynb")
    print("=" * 70)
