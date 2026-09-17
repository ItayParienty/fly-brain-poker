"""Basic playing-card representation for Texas Hold'em."""

import random

RANKS = "23456789TJQKA"  # T = 10
SUITS = "shdc"           # spades, hearts, diamonds, clubs


class Card:
    def __init__(self, rank, suit):
        assert rank in RANKS, f"invalid rank {rank!r}"
        assert suit in SUITS, f"invalid suit {suit!r}"
        self.rank = rank
        self.suit = suit

    @property
    def rank_value(self):
        return RANKS.index(self.rank) + 2  # "2" -> 2, ..., "A" -> 14

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))


class Deck:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        self._rng.shuffle(self.cards)

    def deal(self, n=1):
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401

    deck = Deck(seed=0)
    print("שני קלפים אישיים לדוגמה:", deck.deal(2))
    print("פלופ לדוגמה:", deck.deal(3))
