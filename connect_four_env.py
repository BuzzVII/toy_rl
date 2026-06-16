# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///

from dataclasses import dataclass
from enum import Enum
import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2

PlayerKind = Literal["human", "random", "uniform_random", "q"]
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


@dataclass(frozen=True)
class TacticalPosition:
    name: str
    board_text: str
    current_player: int
    expected_actions: tuple[int, ...]
    description: str


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


class UniformRandomPlayer:
    """Pure random player with no tactics. Useful as a weak baseline."""

    def choose_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no legal actions available")
        return random.choice(legal)


class RandomPlayer:
    """Heuristic-random player.

    This is still stochastic, but it avoids obviously bad tactical moves:
        1. Win immediately if possible.
        2. Block the opponent's immediate win if possible.
        3. Otherwise choose a legal move uniformly at random.

    Keeping this as the default random player gives the learner more useful
    tactical examples than pure random play.
    """

    def choose_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no legal actions available")

        winning_action = find_immediate_winning_action(env.board, legal, env.current_player)
        if winning_action is not None:
            return winning_action

        opponent = other_player(env.current_player)
        blocking_action = find_immediate_winning_action(env.board, legal, opponent)
        if blocking_action is not None:
            return blocking_action

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


class QLearningPlayer:
    """Greedy player backed by a tabular Q-table."""

    def __init__(self, q_table: dict[tuple[int, ...], np.ndarray], epsilon: float = 0.0) -> None:
        self.q_table = q_table
        self.epsilon = epsilon

    def choose_action(self, env: ConnectFourEnv) -> int:
        return choose_epsilon_greedy_action(
            q_table=self.q_table,
            state_key=state_key(env),
            legal_actions=env.legal_actions(),
            epsilon=self.epsilon,
        )



def find_immediate_winning_action(
    board: np.ndarray,
    legal_actions: list[int],
    player: int,
) -> int | None:
    """Return a legal column that wins immediately for player, if one exists."""

    winning_actions: list[int] = []
    for col in legal_actions:
        row = landing_row(board, col)
        if row is None:
            continue
        candidate = board.copy()
        candidate[row, col] = player
        if has_four_from_board(candidate, row, col):
            winning_actions.append(col)

    if not winning_actions:
        return None
    return random.choice(winning_actions)


def landing_row(board: np.ndarray, col: int) -> int | None:
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == EMPTY:
            return row
    return None


def has_four_from_board(board: np.ndarray, row: int, col: int) -> bool:
    player = int(board[row, col])
    if player == EMPTY:
        return False

    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    for dr, dc in directions:
        total = 1
        total += count_direction_on_board(board, row, col, dr, dc, player)
        total += count_direction_on_board(board, row, col, -dr, -dc, player)
        if total >= 4:
            return True
    return False


def count_direction_on_board(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> int:
    count = 0
    r = row + dr
    c = col + dc
    while 0 <= r < ROWS and 0 <= c < COLS and int(board[r, c]) == player:
        count += 1
        r += dr
        c += dc
    return count

def other_player(player: int) -> int:
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    raise ValueError("player must be 1 or 2")


def state_key(env: ConnectFourEnv) -> tuple[int, ...]:
    """Return a hashable symbolic state from the current player's perspective.

    The key is based on the current_player observation, not the absolute 1/2 board.
    This lets a single Q-table learn a policy for "me" versus "opponent".
    """

    original_mode = env.observation_mode
    env.observation_mode = "current_player"
    try:
        obs = env.observation().astype(np.int8)
    finally:
        env.observation_mode = original_mode
    return tuple(int(x) for x in obs.reshape(-1))


def q_values_for(q_table: dict[tuple[int, ...], np.ndarray], key: tuple[int, ...]) -> np.ndarray:
    if key not in q_table:
        q_table[key] = np.zeros(COLS, dtype=np.float32)
    return q_table[key]


def best_legal_action(q_values: np.ndarray, legal_actions: list[int]) -> int:
    if not legal_actions:
        raise RuntimeError("no legal actions available")
    legal_values = [(float(q_values[action]), action) for action in legal_actions]
    max_value = max(value for value, _ in legal_values)
    best_actions = [action for value, action in legal_values if value == max_value]
    return random.choice(best_actions)


def choose_epsilon_greedy_action(
    q_table: dict[tuple[int, ...], np.ndarray],
    state_key: tuple[int, ...],
    legal_actions: list[int],
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.choice(legal_actions)
    return best_legal_action(q_values_for(q_table, state_key), legal_actions)


def update_q_value(
    q_table: dict[tuple[int, ...], np.ndarray],
    key: tuple[int, ...],
    action: int,
    reward: float,
    next_key: tuple[int, ...] | None,
    next_legal_actions: list[int],
    alpha: float,
    gamma: float,
) -> None:
    q_values = q_values_for(q_table, key)
    old_value = float(q_values[action])

    if next_key is None or not next_legal_actions:
        target = reward
    else:
        next_q_values = q_values_for(q_table, next_key)
        next_best = max(float(next_q_values[next_action]) for next_action in next_legal_actions)
        target = reward + gamma * next_best

    q_values[action] = old_value + alpha * (target - old_value)


def train_q_learning_vs_random(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    tactical_training_ratio: float = 0.0,
    seed: int | None = None,
) -> dict[tuple[int, ...], np.ndarray]:
    """Train a tabular Q-learning policy against the default random opponent.

    The default RandomPlayer is heuristic-random: it wins immediately when it can,
    blocks immediate opponent wins when it can, and otherwise plays randomly.

    This is not self-play. The learned agent alternates playing first and second
    during normal episodes, but its opponent is still the RandomPlayer policy.

    If tactical_training_ratio is greater than zero, some episodes start from one
    of the hand-built tactical positions instead of an empty board. In those
    episodes, the Q-learner controls the side whose turn it is in the tactical
    position. This exposes the learner to rare immediate-win and must-block
    states, including diagonals.
    """

    if not 0.0 <= tactical_training_ratio <= 1.0:
        raise ValueError("tactical_training_ratio must be between 0.0 and 1.0")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    q_table: dict[tuple[int, ...], np.ndarray] = {}
    random_player = RandomPlayer()
    wins = 0
    losses = 0
    draws = 0
    tactical_episodes = 0
    positions = tactical_positions()

    for episode in range(1, episodes + 1):
        use_tactical_start = random.random() < tactical_training_ratio

        if use_tactical_start:
            tactical_episodes += 1
            position = random.choice(positions)
            env = make_env_from_position(position)
            agent_player = env.current_player
            LOGGER.debug("episode=%s tactical_start=%s", episode, position.name)
        else:
            env = ConnectFourEnv(observation_mode="current_player")
            agent_player = PLAYER_ONE if episode % 2 else PLAYER_TWO
            env.reset(starting_player=PLAYER_ONE)

            if agent_player == PLAYER_TWO:
                env.step(random_player.choose_action(env))

        while not env.done:
            key = state_key(env)
            legal = env.legal_actions()
            action = choose_epsilon_greedy_action(q_table, key, legal, epsilon)
            agent_result = env.step(action)

            if agent_result.done:
                reward = terminal_reward(env.last_winner, agent_player)
                update_q_value(q_table, key, action, reward, None, [], alpha, gamma)
                break

            opponent_action = random_player.choose_action(env)
            opponent_result = env.step(opponent_action)

            if opponent_result.done:
                reward = terminal_reward(env.last_winner, agent_player)
                update_q_value(q_table, key, action, reward, None, [], alpha, gamma)
                break

            next_key = state_key(env)
            update_q_value(
                q_table=q_table,
                key=key,
                action=action,
                reward=0.0,
                next_key=next_key,
                next_legal_actions=env.legal_actions(),
                alpha=alpha,
                gamma=gamma,
            )

        if env.last_winner == agent_player:
            wins += 1
        elif env.last_winner is None:
            draws += 1
        else:
            losses += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode == 1 or episode % max(1, episodes // 10) == 0:
            LOGGER.info(
                "episode=%s/%s q_states=%s epsilon=%.4f wins=%s losses=%s draws=%s tactical_starts=%s",
                episode,
                episodes,
                len(q_table),
                epsilon,
                wins,
                losses,
                draws,
                tactical_episodes,
            )

    return q_table


def terminal_reward(winner: int | None, agent_player: int) -> float:
    if winner == agent_player:
        return 1.0
    if winner is None:
        return 0.0
    return -1.0


def evaluate_q_vs_random(
    q_table: dict[tuple[int, ...], np.ndarray],
    games: int,
    seed: int | None = None,
) -> dict[str, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wins = 0
    losses = 0
    draws = 0
    random_player = RandomPlayer()
    q_player = QLearningPlayer(q_table, epsilon=0.0)

    for game in range(1, games + 1):
        env = ConnectFourEnv(observation_mode="current_player")
        agent_player = PLAYER_ONE if game % 2 else PLAYER_TWO
        env.reset(starting_player=PLAYER_ONE)

        while not env.done:
            if env.current_player == agent_player:
                action = q_player.choose_action(env)
            else:
                action = random_player.choose_action(env)
            env.step(action)

        if env.last_winner == agent_player:
            wins += 1
        elif env.last_winner is None:
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "losses": losses, "draws": draws}


def save_q_table(q_table: dict[tuple[int, ...], np.ndarray], path: str | Path) -> None:
    payload = {
        "rows": ROWS,
        "cols": COLS,
        "q_table": q_table,
    }
    with Path(path).open("wb") as f:
        pickle.dump(payload, f)


def load_q_table(path: str | Path) -> dict[tuple[int, ...], np.ndarray]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)

    if payload.get("rows") != ROWS or payload.get("cols") != COLS:
        raise ValueError("Q-table board dimensions do not match this environment")

    return payload["q_table"]


def make_player(kind: PlayerKind, q_table: dict[tuple[int, ...], np.ndarray] | None = None):
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer()
    if kind == "uniform_random":
        return UniformRandomPlayer()
    if kind == "q":
        if q_table is None:
            raise ValueError("q player requires a loaded Q-table")
        return QLearningPlayer(q_table)
    raise ValueError(f"unknown player kind: {kind}")


def play_game(
    player_one: PlayerKind,
    player_two: PlayerKind,
    observation_mode: ObservationMode = "current_player",
    seed: int | None = None,
    verbose: bool = True,
    q_table: dict[tuple[int, ...], np.ndarray] | None = None,
) -> int | None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = ConnectFourEnv(observation_mode=observation_mode)
    players = {
        PLAYER_ONE: make_player(player_one, q_table=q_table),
        PLAYER_TWO: make_player(player_two, q_table=q_table),
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



def board_from_text(board_text: str) -> np.ndarray:
    """Parse a 6-line board string into the internal board array.

    Accepted symbols:
        . = empty
        X = player one
        O = player two

    Whitespace inside rows is ignored, so both "X O ." and "XO." work.
    """

    symbol_to_value = {
        ".": EMPTY,
        "X": PLAYER_ONE,
        "O": PLAYER_TWO,
    }
    rows: list[list[int]] = []

    for raw_line in board_text.strip().splitlines():
        line = raw_line.strip().replace(" ", "")
        if not line:
            continue
        if len(line) != COLS:
            raise ValueError(f"expected {COLS} columns, got {len(line)} in row {line!r}")
        try:
            rows.append([symbol_to_value[char] for char in line])
        except KeyError as exc:
            raise ValueError(f"unsupported board symbol: {exc.args[0]!r}") from exc

    if len(rows) != ROWS:
        raise ValueError(f"expected {ROWS} rows, got {len(rows)}")

    return np.array(rows, dtype=np.int8)


def make_env_from_position(position: TacticalPosition) -> ConnectFourEnv:
    env = ConnectFourEnv(observation_mode="current_player")
    env.board = board_from_text(position.board_text)
    env.current_player = position.current_player
    env.last_winner = None
    env.done = False
    return env


def tactical_positions() -> list[TacticalPosition]:
    """Return small tactical positions used to evaluate local game sense.

    These are not full-game benchmarks. They test whether a player handles
    one-move tactics from already constructed legal-looking positions.
    """

    return [
        TacticalPosition(
            name="win_horizontal",
            board_text="""
.......
.......
.......
.......
OO.....
XXX....
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Current player can win immediately with a horizontal four.",
        ),
        TacticalPosition(
            name="block_horizontal",
            board_text="""
.......
.......
.......
.......
XX.....
OOO....
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Opponent has three horizontally; current player must block.",
        ),
        TacticalPosition(
            name="block_vertical",
            board_text="""
.......
.......
.......
...O...
...O...
XX.O...
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Opponent has three vertically; current player must block the column.",
        ),
        TacticalPosition(
            name="block_diagonal_positive_slope",
            board_text="""
.......
.......
.......
..OX...
.O.X...
O..X...
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Opponent threatens a bottom-left to top-right diagonal.",
        ),
        TacticalPosition(
            name="block_diagonal_negative_slope",
            board_text="""
.......
.......
O......
XO.....
XXO....
XXX....
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Opponent threatens a top-left to bottom-right diagonal.",
        ),
        TacticalPosition(
            name="block_horizontal_gap",
            board_text="""
.......
.......
.......
.......
XX.....
OO.O...
""",
            current_player=PLAYER_ONE,
            expected_actions=(2,),
            description="Opponent has a horizontal gap threat; current player must fill the gap.",
        ),
        TacticalPosition(
            name="win_beats_block",
            board_text="""
.......
.......
.......
.......
.......
XXX.OOO
""",
            current_player=PLAYER_ONE,
            expected_actions=(3,),
            description="Current player should win immediately rather than block opponent's threat.",
        ),
    ]


def evaluate_tactical_positions(
    player_kind: PlayerKind,
    q_table: dict[tuple[int, ...], np.ndarray] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    player = make_player(player_kind, q_table=q_table)
    passed = 0
    failed = 0

    print(f"Tactical evaluation for player={player_kind}")
    print()

    for position in tactical_positions():
        env = make_env_from_position(position)
        action = player.choose_action(env)
        ok = action in position.expected_actions

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        expected = ", ".join(str(a) for a in position.expected_actions)
        print(f"{status} {position.name}: chose {action}, expected {expected}")
        print(f"  {position.description}")
        if verbose or not ok:
            print(env.render_text())
        print()

    total = passed + failed
    print(f"Tactical score: {passed}/{total} passed, {failed} failed")
    return {"passed": passed, "failed": failed, "total": total}


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
        choices=["human", "random", "uniform_random", "q"],
        default="human",
        help="Player 1 controller. random means heuristic-random; uniform_random is pure random.",
    )
    parser.add_argument(
        "--p2",
        choices=["human", "random", "uniform_random", "q"],
        default="random",
        help="Player 2 controller. random means heuristic-random; uniform_random is pure random.",
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
        "--train-q",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Train a tabular Q-learning agent against the heuristic-random opponent",
    )
    parser.add_argument(
        "--eval-q",
        type=int,
        default=0,
        metavar="GAMES",
        help="Evaluate a loaded or newly trained Q-table against the heuristic-random opponent",
    )
    parser.add_argument(
        "--eval-tactics",
        choices=["human", "random", "uniform_random", "q"],
        default=None,
        metavar="PLAYER",
        help="Evaluate one player against fixed tactical positions",
    )
    parser.add_argument(
        "--q-table",
        default="connect_four_q_table.pkl",
        help="Path used to save/load a Q-table",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Q-learning update rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Q-learning discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.02, help="Minimum epsilon during training")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Per-episode epsilon decay")
    parser.add_argument(
        "--tactical-training-ratio",
        type=float,
        default=0.0,
        help="Fraction of Q-learning episodes that start from fixed tactical positions",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet-board",
        action="store_true",
        help="Do not print boards when playing a single game",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    q_table: dict[tuple[int, ...], np.ndarray] | None = None

    if args.random_smoke_test > 0:
        run_random_smoke_tests(args.random_smoke_test, seed=args.seed)
        return

    if args.train_q > 0:
        q_table = train_q_learning_vs_random(
            episodes=args.train_q,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            tactical_training_ratio=args.tactical_training_ratio,
            seed=args.seed,
        )
        save_q_table(q_table, args.q_table)
        print(f"Saved Q-table with {len(q_table)} states to {args.q_table}")

    if args.eval_q > 0:
        if q_table is None:
            q_table = load_q_table(args.q_table)
        results = evaluate_q_vs_random(q_table, games=args.eval_q, seed=args.seed)
        print(f"Q vs heuristic-random over {args.eval_q} games")
        print(f"Wins:   {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Draws:  {results['draws']}")
        if args.train_q > 0:
            return

    if args.eval_tactics is not None:
        if args.eval_tactics == "q" and q_table is None:
            q_table = load_q_table(args.q_table)
        evaluate_tactical_positions(
            player_kind=args.eval_tactics,
            q_table=q_table,
            seed=args.seed,
            verbose=args.verbose,
        )
        if args.train_q > 0 or args.eval_q > 0 or args.eval_tactics is not None:
            return

    if args.p1 == "q" or args.p2 == "q":
        if q_table is None:
            q_table = load_q_table(args.q_table)

    play_game(
        player_one=args.p1,
        player_two=args.p2,
        observation_mode=args.observation_mode,
        seed=args.seed,
        verbose=not args.quiet_board,
        q_table=q_table,
    )


if __name__ == "__main__":
    main()
