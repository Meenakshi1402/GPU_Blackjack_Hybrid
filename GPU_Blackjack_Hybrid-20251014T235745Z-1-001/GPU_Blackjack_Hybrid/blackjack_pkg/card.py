from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .utils import randint_inclusive


class Suit(Enum):
    # Suits of a standard deck
    NO_SUIT = 0
    HEARTS = ord('H')
    SPADES = ord('S')
    DIAMONDS = ord('D')
    CLUBS = ord('C')


class Rank(Enum):
    # Ranks Ace (1) through King (13)
    NO_RANK = 0
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


ACE_AS_11 = 10


@dataclass
class Card:
    # Simple container for a playing card
    rank: Rank
    suit: Suit


def deal() -> Card:
    # Construct a random card using uniform suit and rank
    return Card(random_rank(), random_suit())


def is_face(card: Card) -> bool:
    # True if card is J, Q, or K
    return card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING)


def is_ace(card: Card) -> bool:
    # True if card is an Ace
    return card.rank == Rank.ACE


def is_10(card: Card) -> bool:
    # True if card counts as 10 (TEN, J, Q, K)
    return is_face(card) or card.rank == Rank.TEN


def random_suit() -> Suit:
    # Uniformly select a suit
    suits = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]
    index = randint_inclusive(0, 3)
    return suits[index]


def random_rank() -> Rank:
    # Uniformly select a rank from Ace through King
    value = randint_inclusive(Rank.ACE.value, Rank.KING.value)
    return Rank(value)


