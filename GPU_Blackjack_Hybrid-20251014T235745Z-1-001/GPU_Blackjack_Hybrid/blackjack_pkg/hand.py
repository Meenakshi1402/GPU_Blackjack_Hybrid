from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any

from .card import Card, Rank, ACE_AS_11, deal, is_face


MAX_HAND_CARDS = 10


@dataclass
class Hand:
    # Represents a player's or dealer's hand
    cards: List[Card] = field(default_factory=list)
    value: int = 0
    bet: float = 1.0
    player: Optional[Any] = None

    def score(self) -> int:
        # Compute hand value, treating Aces as 1 or 11 when possible
        total = 0
        n_aces = 0
        for card in self.cards:
            total += 10 if is_face(card) else card.rank.value
            if card.rank == Rank.ACE:
                n_aces += 1
        for _ in range(n_aces):
            if total + ACE_AS_11 > 21:
                break
            total += ACE_AS_11
        return total

    def hit(self, card: Optional[Card] = None) -> Card:
        # Add a card (or deal one) and update cached value
        if card is None:
            card = deal()
        assert len(self.cards) < MAX_HAND_CARDS
        assert not self.is_broke()
        self.cards.append(card)
        self.value = self.score()
        return card

    def is_broke(self) -> bool:
        # True if value > 21
        return self.value > 21

    def is_charlie(self) -> bool:
        # Optional rule: 5 cards without busting
        return len(self.cards) == 5 and self.value <= 21

    def is_blackjack(self) -> bool:
        # Natural blackjack: 2 cards totaling 21
        return len(self.cards) == 2 and self.value == 21

    def is_pair(self) -> bool:
        # Two cards with same rank
        if len(self.cards) != 2:
            return False
        return self.cards[0].rank == self.cards[1].rank

    def is_ace_plus_x(self) -> bool:
        # Soft hand: Ace + non-Ace
        if len(self.cards) != 2:
            return False
        a, b = self.cards
        return (a.rank == Rank.ACE and b.rank != Rank.ACE) or (a.rank != Rank.ACE and b.rank == Rank.ACE)


