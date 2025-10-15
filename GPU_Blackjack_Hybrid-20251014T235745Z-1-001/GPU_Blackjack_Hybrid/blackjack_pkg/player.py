from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .hand import Hand, MAX_HAND_CARDS


MAX_YOUR_HANDS = 4


@dataclass
class Player:
    # A blackjack player who may have multiple hands due to splitting
    hands: List[Hand] = field(default_factory=lambda: [Hand()])
    strategy: Optional["Strategy"] = None
    pl: float = 0.0

    def __post_init__(self) -> None:
        # Link the first hand back to this player
        if self.hands:
            self.hands[0].player = self

    @property
    def size(self) -> int:
        # Number of hands currently held
        return len(self.hands)

    def add(self, hand: Hand) -> int:
        # Add a new split hand and return its index
        assert self.size < MAX_YOUR_HANDS
        hand.player = self
        self.hands.append(hand)
        return self.size - 1

    def hit(self, handno: Optional[int] = None):
        # Hit either the last hand or a specific hand index
        # Handno means “hand number” (index) in the player’s list of hands.
        assert self.size > 0
        if handno is None:
            handno = self.size - 1
        assert 0 <= handno < self.size
        return self.hands[handno].hit()


