"""Reference opponents to train and measure against.

The choice of opponent decides what there is to learn. Against a player who
never folds, simply raising more wins chips no matter what the cards are, so
a fly can look like it improved without ever having learned to read its hand.
A folding opponent removes that shortcut: bluffing stops paying, and the only
way left to make money is to put chips in when the cards deserve it.
"""

from cards.fly import hand_strength
from cards.game import CALL, FOLD, RAISE


class AlwaysCall:
    """Calling station. Never folds, so bluffing never works and aggression
    is rewarded only when the cards back it up - eventually."""

    def act(self, obs):
        return CALL


class AlwaysRaise:
    def act(self, obs):
        return RAISE


class TightOpponent:
    """Folds weak hands, raises strong ones, calls in between. Punishes
    indiscriminate aggression by folding to it when weak and calling it when
    strong."""

    def __init__(self, fold_below=0.35, raise_above=0.70):
        self.fold_below = fold_below
        self.raise_above = raise_above

    def act(self, obs):
        strength = hand_strength(obs.hole, obs.community)
        if strength >= self.raise_above:
            return RAISE
        if strength < self.fold_below and obs.to_call > 0:
            return FOLD
        return CALL


class RandomOpponent:
    """Uniform noise, as a floor to compare against."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs):
        return self.rng.choice([FOLD, CALL, RAISE])
