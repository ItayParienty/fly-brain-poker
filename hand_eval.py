"""Evaluates the best 5-card poker hand out of up to 7 cards.

The core trick: every 5-card hand gets turned into a tuple like
(category, tiebreak1, tiebreak2, ...) where a bigger category always wins
(straight flush > quads > full house > ...), and within the same category
Python's normal tuple comparison breaks ties (e.g. a pair of Kings beats a
pair of Queens). That means comparing two hands is just `score_a > score_b`.
"""

from collections import Counter
from itertools import combinations

CATEGORY_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
]


def _score_five(cards):
    values = sorted((c.rank_value for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(values)
    # groups: e.g. a full house -> [(3, king_value), (2, queen_value)]
    groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    group_sizes = [count for _, count in groups]
    ordered_values = [value for value, _ in groups]

    is_flush = len(set(suits)) == 1

    unique_values = sorted(set(values), reverse=True)
    is_straight = False
    straight_high = None
    if len(unique_values) == 5:
        if unique_values[0] - unique_values[4] == 4:
            is_straight = True
            straight_high = unique_values[0]
        elif unique_values == [14, 5, 4, 3, 2]:  # wheel: A-2-3-4-5
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return (8, straight_high)
    if group_sizes == [4, 1]:
        return (7, *ordered_values)
    if group_sizes == [3, 2]:
        return (6, *ordered_values)
    if is_flush:
        return (5, *values)
    if is_straight:
        return (4, straight_high)
    if group_sizes == [3, 1, 1]:
        return (3, *ordered_values)
    if group_sizes == [2, 2, 1]:
        return (2, *ordered_values)
    if group_sizes == [2, 1, 1, 1]:
        return (1, *ordered_values)
    return (0, *values)


def best_hand_score(cards):
    """cards: 5, 6 or 7 Card objects (hole cards + community cards so far).
    Returns the best achievable (category, tiebreakers...) tuple."""

    assert 5 <= len(cards) <= 7
    return max(_score_five(list(combo)) for combo in combinations(cards, 5))


def describe(score):
    return CATEGORY_NAMES[score[0]]


if __name__ == "__main__":
    from cards import Card

    hand = [Card("A", "s"), Card("K", "s"), Card("Q", "s"), Card("J", "s"), Card("T", "s"),
            Card("2", "h"), Card("3", "h")]
    score = best_hand_score(hand)
    print(hand, "->", describe(score), score)

    hand2 = [Card("9", "h"), Card("9", "d"), Card("9", "s"), Card("4", "c"), Card("4", "h"),
             Card("2", "s"), Card("7", "d")]
    score2 = best_hand_score(hand2)
    print(hand2, "->", describe(score2), score2)
