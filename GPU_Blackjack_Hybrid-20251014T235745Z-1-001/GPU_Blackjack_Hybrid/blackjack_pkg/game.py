from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .player import Player, MAX_YOUR_HANDS
from .hand import Hand, MAX_HAND_CARDS
from .card import Card, Rank
from .strategy import Strategy, Play, doSection1, doSection2, doSection3, doSection4
from .utils import seed as rng_seed


PAYOFF_BLACKJACK = 1.5
PAYOFF_CHARLIE = 2.0
PAYOFF_PUSH = 0.0

WINS = 0
BLACKJACKS = 1
CHARLIES = 2
LOSSES = 3
BUSTS = 4
DEALER_BLACKJACKS = 5
PUSHES = 6


@dataclass
class Game:
    count: List[int]
    nohands: int
    pl: float


def Game_() -> Game:
    return Game([0, 0, 0, 0, 0, 0, 0], 0, 0.0)


def start(strategy: Strategy, ngames: int, seed: int) -> Game:
    rng_seed(seed)
    statistics = Game_()
    for _ in range(ngames):
        play(strategy, statistics)
    return statistics


def _get_play(hand: Hand, upcard: Card, strategy: Strategy) -> int:
    if hand.is_pair():
        return doSection4(hand, upcard, strategy)
    elif hand.is_ace_plus_x():
        return doSection3(hand, upcard, strategy)
    elif 5 <= hand.value <= 11:
        return doSection2(hand, upcard, strategy)
    else:
        return doSection1(hand, upcard, strategy)


def play(strategy: Strategy, statistics: Game) -> None:
    player = Player(strategy=strategy)
    dealer = Hand()

    player.hit()
    dealer.hit()
    player.hit()
    dealer.hit()

    upcard = dealer.cards[0]
    _play_player(player, upcard, strategy)
    _play_dealer_and_settle(dealer, player, statistics)

    statistics.pl += player.pl
    statistics.nohands += player.size


def _play_player(player: Player, upcard: Card, strategy: Strategy) -> None:
    hand = player.hands[0]
    _playout(hand, upcard, player, strategy)


def _playout(hand: Hand, upcard: Card, player: Player, strategy: Strategy) -> None:
    assert not hand.is_broke()
    assert hand.bet != 0
    play_move = _get_play(hand, upcard, strategy)
    if play_move == Play.STAY:
        return
    if play_move == Play.HIT:
        hand.hit()
        if hand.is_broke() or hand.is_blackjack() or len(hand.cards) >= MAX_HAND_CARDS:
            return
        _playout(hand, upcard, player, strategy)
        return
    if play_move == Play.DOUBLE_DOWN:
        assert len(hand.cards) == 2
        hand.bet *= 2.0
        hand.hit()
        return
    if play_move == Play.SPLIT:
        if len(hand.cards) != 2:
            return
        _split(hand, upcard, player, strategy)
        return


def _split(hand1: Hand, upcard: Card, player: Player, strategy: Strategy) -> None:
    assert len(hand1.cards) == 2
    if player.size >= MAX_YOUR_HANDS:
        _splitbackup(hand1, upcard, player, strategy)
        return
    play_through = not (hand1.is_pair() and hand1.cards[0].rank == Rank.ACE)

    new_hand = Hand(player=player)
    moved = hand1.cards.pop()
    hand1.value = hand1.score()

    hand1.hit()
    new_hand.hit(moved)
    new_hand.hit()

    index = player.add(new_hand)
    hand2 = player.hands[index]

    assert len(hand1.cards) == 2 and len(hand2.cards) == 2
    assert player.size > 1

    if not play_through:
        return
    _playout(hand1, upcard, player, strategy)
    _playout(hand2, upcard, player, strategy)


def _splitbackup(hand: Hand, upcard: Card, player: Player, strategy: Strategy) -> None:
    if hand.value >= 12:
        move = doSection1(hand, upcard, strategy)
    else:
        move = doSection2(hand, upcard, strategy)
    if move == Play.STAY:
        return
    if move == Play.HIT:
        hand.hit()
        if not hand.is_broke():
            _playout(hand, upcard, player, strategy)
        return
    if move == Play.DOUBLE_DOWN:
        hand.hit()
        hand.bet *= 2.0
        return


def _play_dealer_and_settle(dealer: Hand, player: Player, statistics: Game) -> None:
    remaining = player.size
    for hand in player.hands:
        assert hand.bet > 0
        assert hand.player is player
        if hand.is_broke():
            player.pl -= hand.bet
            statistics.count[BUSTS] += 1
            remaining -= 1
        elif hand.is_blackjack() and player.size == 1:
            player.pl += hand.bet * PAYOFF_BLACKJACK
            statistics.count[BLACKJACKS] += 1
            remaining -= 1

    if remaining == 0:
        return

    while dealer.value < 17:
        dealer.hit()

    for hand in player.hands:
        assert len(hand.cards) >= 2
        if hand.bet == 2:
            assert len(hand.cards) == 3
        if hand.is_broke() or (hand.is_blackjack() and player.size == 1):
            continue
        if dealer.is_blackjack():
            player.pl -= hand.bet
            statistics.count[DEALER_BLACKJACKS] += 1
        elif dealer.is_broke():
            player.pl += hand.bet
            statistics.count[WINS] += 1
        elif dealer.value < hand.value:
            player.pl += hand.bet
            statistics.count[WINS] += 1
        elif dealer.value > hand.value:
            player.pl -= hand.bet
            statistics.count[LOSSES] += 1
        elif dealer.value == hand.value:
            player.pl += 0
            statistics.count[PUSHES] += 1
        else:
            raise AssertionError("Unhandled comparison state")


