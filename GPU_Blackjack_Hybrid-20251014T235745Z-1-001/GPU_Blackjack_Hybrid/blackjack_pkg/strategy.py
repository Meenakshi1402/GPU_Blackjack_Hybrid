from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .hand import Hand
from .card import Card, Rank


class Play:
    # Possible actions a player can take
    NO_PLAY = 0
    STAY = ord('S')
    HIT = ord('H')
    DOUBLE_DOWN = ord('D')
    SPLIT = ord('P')


NUMBER_RULES = 10 * 43

SECTION1 = 0
SECTION2 = 1
SECTION3 = 2
SECTION4 = 3

S, H, D, P, X = Play.STAY, Play.HIT, Play.DOUBLE_DOWN, Play.SPLIT, Play.NO_PLAY


@dataclass
class Strategy:
    # Lookup table of plays; pl reserved for future use
    pl: float
    rules: List[int]


def Strategy_() -> Strategy:
    # Default strategy with 'no play' everywhere
    rules = [X] * NUMBER_RULES
    return Strategy(0.0, rules)


def BasicStrategy_() -> Strategy:
    # Populate the strategy table with standard basic-strategy rules
    rules = [X] * NUMBER_RULES
    # Fill rules following Strategy.cpp arrays
    def set_row(start_row: int, values: List[int]):
        for row_offset, row in enumerate(values):
            for col, val in enumerate(row):
                rules[(start_row + row_offset) * 10 + col] = val

    # Section I: rows 0..9 (21..12)
    set_row(0, [
        [S,S,S,S,S,S,S,S,S,S],  # 21
        [S,S,S,S,S,S,S,S,S,S],  # 20
        [S,S,S,S,S,S,S,S,S,S],  # 19
        [S,S,S,S,S,S,S,S,S,S],  # 18
        [S,S,S,S,S,S,S,S,S,S],  # 17
        [S,S,S,S,S,H,H,H,H,H],  # 16
        [S,S,S,S,S,H,H,H,H,H],  # 15
        [S,S,S,S,S,H,H,H,H,H],  # 14
        [S,S,S,S,S,H,H,H,H,H],  # 13
        [H,H,S,S,S,H,H,H,H,H],  # 12
    ])

    # Section II: rows 10..16 (11..5), row 16 is 4s for splitback
    set_row(10, [
        [D,D,D,D,D,D,D,D,D,H],  # 11
        [D,D,D,D,D,D,D,D,H,H],  # 10
        [H,D,D,D,D,H,H,H,H,H],  # 9
        [H,H,H,H,H,H,H,H,H,H],  # 8
        [H,H,H,H,H,H,H,H,H,H],  # 7
        [H,H,H,H,H,H,H,H,H,H],  # 6
        [H,H,H,H,H,H,H,H,H,H],  # 5
    ])
    set_row(17, [
        [H,H,H,H,H,H,H,H,H,H],  # 4 (accommodate splitback)
    ])

    # Section III: rows 18..29 (A,K .. A,2)
    set_row(18, [
        [S,S,S,S,S,S,S,S,S,S],  # A,K
        [S,S,S,S,S,S,S,S,S,S],  # A,Q
        [S,S,S,S,S,S,S,S,S,S],  # A,J
        [S,S,S,S,S,S,S,S,S,S],  # A,T
        [S,S,S,S,S,S,S,S,S,S],  # A,9
        [S,S,S,S,S,S,S,S,S,S],  # A,8
        [S,D,D,D,D,S,S,H,H,H],  # A,7
        [H,D,D,D,D,H,H,H,H,H],  # A,6
        [H,H,D,D,D,H,H,H,H,H],  # A,5
        [H,H,D,D,D,H,H,H,H,H],  # A,4
        [H,H,H,D,D,H,H,H,H,H],  # A,3
        [H,H,H,D,D,H,H,H,H,H],  # A,2
    ])

    # Section IV: rows 30..42 (pairs)
    set_row(30, [
        [P,P,P,P,P,P,P,P,P,P],  # A,A
        [S,S,S,S,S,S,S,S,S,S],  # T,K
        [S,S,S,S,S,S,S,S,S,S],  # Q,Q
        [S,S,S,S,S,S,S,S,S,S],  # J,J
        [S,S,S,S,S,S,S,S,S,S],  # T,T
        [P,P,P,P,P,S,P,P,S,S],  # 9,9
        [P,P,P,P,P,P,P,P,P,P],  # 8,8
        [P,P,P,P,P,P,H,H,H,H],  # 7,7
        [P,P,P,P,P,H,H,H,H,H],  # 6,6
        [D,D,D,D,D,D,D,D,H,H],  # 5,5
        [H,H,H,P,P,H,H,H,H,H],  # 4,4
        [P,P,P,P,P,P,H,H,H,H],  # 3,3
        [P,P,P,P,P,P,H,H,H,H],  # 2,2
    ])

    return Strategy(0.0, rules)


def _rank_value_for_col(rank: Rank) -> int:
    # Convert rank to strategy column (T/J/Q/K treated as 10)
    return 10 if rank.value >= 10 else rank.value


_jump_tab: List[Tuple[int, int]] = [
    # Row ranges for each strategy section
    (0, 9),   # Section I
    (10, 17), # Section II
    (18, 29), # Section III
    (30, 42), # Section IV
]


def doSection4(hand: Hand, upcard: Card, strategy: Strategy) -> int:
    # Pairs: pick play from section IV table
    assert len(hand.cards) == 2
    c1, c2 = hand.cards
    assert c1.rank == c2.rank
    offset = 0 if c1.rank == Rank.ACE else (Rank.KING.value - c1.rank.value + 1)
    row = _jump_tab[3][0] + offset
    col = 9 if upcard.rank == Rank.ACE else _rank_value_for_col(upcard.rank) - 2
    return strategy.rules[row * 10 + col]


def doSection3(hand: Hand, upcard: Card, strategy: Strategy) -> int:
    # Soft totals (Ace + non-Ace)
    assert len(hand.cards) == 2
    c1, c2 = hand.cards
    card = c2 if c1.rank == Rank.ACE else c1
    offset = Rank.KING.value - card.rank.value
    row = _jump_tab[2][0] + offset
    col = 9 if upcard.rank == Rank.ACE else _rank_value_for_col(upcard.rank) - 2
    return strategy.rules[row * 10 + col]


def doSection2(hand: Hand, upcard: Card, strategy: Strategy) -> int:
    # Totals 4..11
    assert len(hand.cards) >= 2 and 4 <= hand.value <= 11
    offset = 11 - hand.value
    row = _jump_tab[1][0] + offset
    col = 9 if upcard.rank == Rank.ACE else _rank_value_for_col(upcard.rank) - 2
    index = row * 10 + col
    play = strategy.rules[index]
    if play == Play.DOUBLE_DOWN and len(hand.cards) > 2:
        return Play.HIT
    return play


def doSection1(hand: Hand, upcard: Card, strategy: Strategy) -> int:
    # Totals 12..21
    assert len(hand.cards) >= 2 and 12 <= hand.value <= 21
    offset = 21 - hand.value
    row = _jump_tab[0][0] + offset
    col = 9 if upcard.rank == Rank.ACE else _rank_value_for_col(upcard.rank) - 2
    index = row * 10 + col
    play = strategy.rules[index]
    if play == Play.DOUBLE_DOWN and len(hand.cards) > 2:
        return Play.HIT
    return play


