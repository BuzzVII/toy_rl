# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///


from dataclasses import dataclass
from enum import Enum
import argparse
import random
from typing import Literal

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2

PlayerKind = Literal["human", "random"]
ObservationMode = Literal["absolute", "current_player"]


class GameStatus(Enum):
    ONGOING = "ongoing"
    WIN = "win"
    DRAW = "draw"


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class ConnectFourEnv:
    """Minimal Connect Four environment suitable for toy RL experiments.

    Board encoding:
        0 = empty
        1 = player one
        2 = player two

    Actions:
        Integer column index from 0 to 6.

    Observation modes:
        absolute:
            6 x 7 integer board using 0/1/2.

        current_player:
            2 x 6 x 7 float tensor.
            channel 0 = current player's pieces
            channel 1 = opponent's pieces
    """

    def __init__(
        self,
        observation_mode: ObservationMode = "current_player",
        invalid_move_ends_game: bool = True,
    ) -> None:
        self.observation_mode = observation_mode
        self.invalid_move_ends_game = invalid_move_ends_game
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = PLAYER_ONE
        self.last_winner: int | None = None
        self.done = False

    def reset(self, starting_player: int = PLAYER_ONE) -> np.ndarray:
        if starting_player not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("starting_player must be 1 or 2")

        self.board.fill(EMPTY)
        self.current_player = starting_player
        self.last_winner = None
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        if self.observation_mode == "absolute":
            return self.board.copy()

        if self.observation_mode == "current_player":
            opponent = other_player(self.current_player)
            return np.stack(
                [
                    (self.board == self.current_player).astype(np.float32),
                    (self.board == opponent).astype(np.float32),
                ],
                axis=0,
            )

        raise ValueError(f"unknown observation mode: {self.observation_mode}")

    def legal_actions(self) -> list[int]:
        return [col for col in range(COLS) if self.board[0, col] == EMPTY]

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(COLS, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("cannot call step() after the game is done; call reset() first")

        if action not in range(COLS):
            return self._handle_invalid_move(action, reason="column out of range")

        if self.board[0, action] != EMPTY:
            return self._handle_invalid_move(action, reason="column is full")

        row = self._drop_piece(action, self.current_player)

        status = self._status_after_move(row, action)
        reward = 0.0
        info: dict = {
            "status": status.value,
            "player_moved": self.current_player,
            "row": row,
            "col": action,
            "winner": None,
            "legal_actions": self.legal_actions(),
            "legal_action_mask": self.legal_action_mask(),
        }

        if status == GameStatus.WIN:
            reward = 1.0
            self.done = True
            self.last_winner = self.current_player
            info["winner"] = self.current_player
        elif status == GameStatus.DRAW:
            reward = 0.0
            self.done = True
        else:
            self.current_player = other_player(self.current_player)

        return StepResult(
            observation=self.observation(),
            reward=reward,
            done=self.done,
            info=info,
        )

    def _handle_invalid_move(self, action: int, reason: str) -> StepResult:
        info = {
            "status": "invalid_move",
            "reason": reason,
            "action": action,
            "player_moved": self.current_player,
            "winner": other_player(self.current_player) if self.invalid_move_ends_game else None,
            "legal_actions": self.legal_actions(),
            "legal_action_mask": self.legal_action_mask(),
        }

        if self.invalid_move_ends_game:
            self.done = True
            self.last_winner = other_player(self.current_player)
            reward = -1.0
        else:
            reward = -0.1

        return StepResult(
            observation=self.observation(),
            reward=reward,
            done=self.done,
            info=info,
        )

    def _drop_piece(self, col: int, player: int) -> int:
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = player
                return row
        raise RuntimeError("attempted to drop piece into full column")

    def _status_after_move(self, row: int, col: int) -> GameStatus:
        if self._has_four_from(row, col):
            return GameStatus.WIN
        if not self.legal_actions():
            return GameStatus.DRAW
        return GameStatus.ONGOING

    def _has_four_from(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]

        for dr, dc in directions:
            total = 1
            total += self._count_direction(row, col, dr, dc, player)
            total += self._count_direction(row, col, -dr, -dc, player)
            if total >= 4:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        r = row + dr
        c = col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def render_text(self) -> str:
        symbols = {
            EMPTY: ".",
            PLAYER_ONE: "X",
            PLAYER_TWO: "O",
        }
        lines = []
        for row in range(ROWS):
            lines.append(" ".join(symbols[int(cell)] for cell in self.board[row]))
        lines.append("0 1 2 3 4 5 6")
        return "\n".join(lines)


class RandomPlayer:
    def choose_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no legal actions available")
        return random.choice(legal)


class HumanPlayer:
    def choose_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        while True:
            raw = input(f"Player {env.current_player}, choose column {legal}: ").strip()
            try:
                action = int(raw)
            except ValueError:
                print("Enter an integer column number.")
                continue

            if action in legal:
                return action

            print(f"Illegal move. Legal columns are: {legal}")


def other_player(player: int) -> int:
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    raise ValueError("player must be 1 or 2")


def make_player(kind: PlayerKind):
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer()
    raise ValueError(f"unknown player kind: {kind}")


def play_game(
    player_one: PlayerKind,
    player_two: PlayerKind,
    observation_mode: ObservationMode = "current_player",
    seed: int | None = None,
    verbose: bool = True,
) -> int | None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = ConnectFourEnv(observation_mode=observation_mode)
    players = {
        PLAYER_ONE: make_player(player_one),
        PLAYER_TWO: make_player(player_two),
    }

    env.reset()

    if verbose:
        print(env.render_text())
        print()

    while not env.done:
        player = players[env.current_player]
        action = player.choose_action(env)
        result = env.step(action)

        if verbose:
            print(f"Player {result.info['player_moved']} plays column {action}")
            print(env.render_text())
            print()

    winner = env.last_winner
    if verbose:
        if winner is None:
            print("Draw")
        else:
            print(f"Player {winner} wins")

    return winner


def run_random_smoke_tests(num_games: int, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wins = {PLAYER_ONE: 0, PLAYER_TWO: 0, None: 0}

    for _ in range(num_games):
        winner = play_game("random", "random", verbose=False)
        wins[winner] += 1
        LOGGER.debug(f"Game finished. Winner: {winner}. Current tally: {wins}")

    print(f"Random smoke test over {num_games} games")
    print(f"Player 1 wins: {wins[PLAYER_ONE]}")
    print(f"Player 2 wins: {wins[PLAYER_TWO]}")
    print(f"Draws:         {wins[None]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Connect Four environment")
    parser.add_argument(
        "--p1",
        choices=["human", "random"],
        default="human",
        help="Player 1 controller",
    )
    parser.add_argument(
        "--p2",
        choices=["human", "random"],
        default="random",
        help="Player 2 controller",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["absolute", "current_player"],
        default="current_player",
        help="Observation encoding used by the environment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--random-smoke-test",
        type=int,
        default=0,
        help="Run N random-vs-random games without printing boards",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print game boards and results",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if args.random_smoke_test > 0:
        run_random_smoke_tests(args.random_smoke_test, seed=args.seed)
        return

    play_game(
        player_one=args.p1,
        player_two=args.p2,
        observation_mode=args.observation_mode,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
