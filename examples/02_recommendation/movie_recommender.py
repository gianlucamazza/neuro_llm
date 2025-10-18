"""
Sistema di Raccomandazione Film usando IBM Logical Neural Networks (LNN)

Questo modulo implementa raccomandazioni basate su regole logiche LNN.
Dimostra inferenza preferenze e similarità tra film.

Per demo interattiva, vedi movie_recommender.ipynb
"""

from lnn import Predicate, Variable, Implies, And, Forall, Model, Fact, World
from typing import Dict, List, Set


class MovieRecommendationSystem:
    """Sistema raccomandazione basato su regole logiche LNN + learning."""

    def __init__(self):
        self.model = Model()
        self._setup_predicates()
        self._setup_rules()

        self.users: Set[str] = set()
        self.movies: Set[str] = set()
        self.genres: Set[str] = set()

    def _setup_predicates(self):
        """Definisce i predicati per il sistema di raccomandazione"""
        # Predicati binari (arity=2)
        self.Guarda = Predicate('Guarda', arity=2)  # (utente, film)
        self.Ha_Genere = Predicate('Ha_Genere', arity=2)  # (film, genere)
        self.Preferisce_Genere = Predicate('Preferisce_Genere', arity=2)  # (utente, genere)
        self.Consiglia = Predicate('Consiglia', arity=2)  # (film, utente)
        self.Simile_A = Predicate('Simile_A', arity=2)  # (film1, film2)

    def _setup_rules(self):
        """Definisce le regole logiche per la raccomandazione"""
        utente = Variable('utente')
        film = Variable('film')
        genere = Variable('genere')
        film2 = Variable('film2')

        # Regola 1: Se utente preferisce genere E film ha quel genere → consiglia film
        self.model.add_knowledge(
            Forall(
                utente, film, genere,
                Implies(
                    And(
                        self.Preferisce_Genere(utente, genere),
                        self.Ha_Genere(film, genere)
                    ),
                    self.Consiglia(film, utente)
                ),
                world=World.AXIOM
            ),
        )

        # Regola 2: Se utente guarda film E film è simile a film2 → consiglia film2
        self.model.add_knowledge(
            Forall(
                utente, film, film2,
                Implies(
                    And(
                        self.Guarda(utente, film),
                        self.Simile_A(film, film2)
                    ),
                    self.Consiglia(film2, utente)
                ),
                world=World.AXIOM
            ),
        )

    def add_movie(self, title: str, genres: List[str]):
        """
        Aggiunge un film al catalogo

        Args:
            title: Titolo del film
            genres: Lista di generi del film
        """
        self.movies.add(title)

        data = {self.Ha_Genere: {}}
        for genre in genres:
            self.genres.add(genre)
            data[self.Ha_Genere][(title, genre)] = Fact.TRUE

        self.model.add_data(data)

    def add_user_preference(self, user: str, genre: str, strength: float = 1.0):
        """
        Aggiunge preferenza esplicita di un utente per un genere

        Args:
            user: Nome utente
            genre: Genere preferito
            strength: Forza della preferenza [0.0, 1.0]
        """
        self.users.add(user)
        self.genres.add(genre)

        if strength == 1.0:
            value = Fact.TRUE
        else:
            value = (strength, strength)

        self.model.add_data({
            self.Preferisce_Genere: {
                (user, genre): value
            }
        })

    def add_viewing_history(self, user: str, movies: List[str]):
        """
        Aggiunge lo storico di visualizzazione di un utente

        Args:
            user: Nome utente
            movies: Lista di film guardati
        """
        self.users.add(user)

        data = {self.Guarda: {}}
        for movie in movies:
            if movie in self.movies:
                data[self.Guarda][(user, movie)] = Fact.TRUE

        self.model.add_data(data)

    def add_movie_similarity(self, movie1: str, movie2: str, score: float = 1.0):
        """
        Aggiunge similarità tra due film

        Args:
            movie1, movie2: Titoli dei film
            score: Score di similarità [0.0, 1.0]
        """
        if score == 1.0:
            value = Fact.TRUE
        else:
            value = (score, score)

        self.model.add_data({
            self.Simile_A: {
                (movie1, movie2): value,
                (movie2, movie1): value,  # Simmetrica
            }
        })

    def infer_preferences_from_history(self):
        """
        Inferisce preferenze di genere dallo storico di visualizzazione

        Se un utente guarda molti film di un genere → preferisce quel genere
        """
        for user in self.users:
            genre_counts = {}

            # Conta quanti film per genere ha guardato
            guarda_state = self.Guarda.state()
            ha_genere_state = self.Ha_Genere.state()
            preferisce_state = self.Preferisce_Genere.state()

            for movie in self.movies:
                watched = guarda_state.get((user, movie))
                # Gestisce sia Fact che tuple bounds
                if watched:
                    if isinstance(watched, tuple):
                        watched_val = watched[0]  # Lower bound
                    else:
                        # Fact.TRUE o World.TRUE
                        watched_val = 1.0

                    if watched_val > 0.5:  # Ha guardato il film
                        # Controlla i generi del film
                        for genre in self.genres:
                            has_genre = ha_genere_state.get((movie, genre))
                            if has_genre:
                                if isinstance(has_genre, tuple):
                                    genre_val = has_genre[0]
                                else:
                                    genre_val = 1.0

                                if genre_val > 0.5:
                                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

            # Inferisci preferenza se ha guardato >= 2 film del genere
            for genre, count in genre_counts.items():
                if count >= 2:
                    # Forza preferenza proporzionale al numero di film
                    strength = min(count / 3.0, 1.0)  # Max a 3 film

                    # Aggiungi solo se non già esplicita
                    existing = preferisce_state.get((user, genre))
                    if existing is None:
                        self.add_user_preference(user, genre, strength)

    def get_recommendations(self, user: str, top_n: int = 5) -> List[tuple]:
        """
        Genera raccomandazioni per un utente

        Args:
            user: Nome utente
            top_n: Numero massimo di raccomandazioni

        Returns:
            Lista di (film, score) ordinata per score decrescente
        """
        # Prima inferisci preferenze da storico
        self.infer_preferences_from_history()

        # Esegui inferenza LNN
        self.model.infer()

        # Estrai raccomandazioni
        recommendations = []
        guarda_state = self.Guarda.state()
        consiglia_state = self.Consiglia.state()

        for movie in self.movies:
            # Verifica se già guardato
            watched = guarda_state.get((user, movie))
            if watched:
                if isinstance(watched, tuple):
                    watched_val = watched[0]
                else:
                    watched_val = 1.0

                if watched_val > 0.5:
                    continue  # Skip film già visti

            # Ottieni score raccomandazione
            rec_bounds = consiglia_state.get((movie, user))
            if rec_bounds:
                # Usa media dei bounds come score
                if isinstance(rec_bounds, tuple):
                    score = (rec_bounds[0] + rec_bounds[1]) / 2
                else:
                    score = 1.0

                if score > 0.3:  # Soglia minima
                    recommendations.append((movie, score))

        # Ordina per score decrescente
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_n]

    def explain_recommendation(self, user: str, movie: str) -> str:
        """
        Genera spiegazione per una raccomandazione

        Args:
            user: Nome utente
            movie: Film raccomandato

        Returns:
            Stringa di spiegazione
        """
        explanations = []
        ha_genere_state = self.Ha_Genere.state()
        preferisce_state = self.Preferisce_Genere.state()
        guarda_state = self.Guarda.state()
        simile_state = self.Simile_A.state()

        # Check generi del film
        movie_genres = []
        for genre in self.genres:
            has_genre = ha_genere_state.get((movie, genre))
            if has_genre:
                genre_val = has_genre[0] if isinstance(has_genre, tuple) else 1.0
                if genre_val > 0.5:
                    movie_genres.append(genre)

        # Check preferenze utente
        for genre in movie_genres:
            pref = preferisce_state.get((user, genre))
            if pref:
                pref_val = pref[0] if isinstance(pref, tuple) else 1.0
                if pref_val > 0.5:
                    explanations.append(f"Ti piace il genere {genre}")

        # Check film simili guardati
        for watched_movie in self.movies:
            watched = guarda_state.get((user, watched_movie))
            if watched:
                watched_val = watched[0] if isinstance(watched, tuple) else 1.0
                if watched_val > 0.5:
                    similarity = simile_state.get((watched_movie, movie))
                    if similarity:
                        sim_val = similarity[0] if isinstance(similarity, tuple) else 1.0
                        if sim_val > 0.5:
                            explanations.append(f"Simile a '{watched_movie}' che hai guardato")

        if not explanations:
            return "Raccomandato in base al tuo profilo"

        return " | ".join(explanations)


# Verifica modulo quando eseguito direttamente
if __name__ == "__main__":
    print("=" * 70)
    print("Movie Recommendation System - Module Check")
    print("=" * 70)

    # Test creazione sistema
    system = MovieRecommendationSystem()
    print("\n✓ MovieRecommendationSystem creato correttamente")

    # Quick test - only creation
    system.add_movie('Inception', ['SciFi', 'Thriller'])
    print("✓ Metodi funzionanti")

    print("\n" + "=" * 70)
    print("Per la demo completa con raccomandazioni, usa il notebook:")
    print("  movie_recommender.ipynb")
    print("=" * 70)
