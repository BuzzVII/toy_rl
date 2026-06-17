# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "torch",
#   "matplotlib",
# ]
# ///

from dataclasses import dataclass
from enum import Enum
import argparse
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch import optim

torch.set_num_threads(1)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2

PlayerKind = Literal["human", "random", "uniform_random", "ac"]
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


@dataclass
class TrainingHistory:
    steps: list[int]
    episodes: list[int]
    total_losses: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    entropies: list[float]
    rewards: list[float]




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



class ActorCriticNet(nn.Module):
    """Small MLP with an actor head and a critic head.

    Input:
        2 x 6 x 7 symbolic current-player observation.

    Outputs:
        logits: 7 unnormalised action preferences, one per column.
        value: scalar V(state), estimated return for the current player.
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, COLS)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value


class ActorCriticPlayer:
    """Greedy player backed by the actor head of an actor-critic network."""

    def __init__(self, model: ActorCriticNet, device: torch.device | str = "cpu", sample: bool = False) -> None:
        self.model = model
        self.device = torch.device(device)
        self.sample = sample
        self.model.to(self.device)
        self.model.eval()

    def choose_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no legal actions available")

        obs = observation_from_env(env)
        legal_mask = torch.as_tensor(env.legal_action_mask(), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            masked_logits = logits[0].masked_fill(~legal_mask, -1.0e9)
            if self.sample:
                dist = torch.distributions.Categorical(logits=masked_logits)
                return int(dist.sample().item())
            return best_legal_action_from_tensor(masked_logits, legal)
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


def observation_from_env(env: ConnectFourEnv) -> np.ndarray:
    """Return a current-player symbolic observation without mutating the env permanently."""

    original_mode = env.observation_mode
    env.observation_mode = "current_player"
    try:
        return env.observation().astype(np.float32)
    finally:
        env.observation_mode = original_mode


def best_legal_action_from_tensor(q_values: torch.Tensor, legal_actions: list[int]) -> int:
    if not legal_actions:
        raise RuntimeError("no legal actions available")
    legal_tensor = torch.as_tensor(legal_actions, dtype=torch.long, device=q_values.device)
    legal_values = q_values.index_select(0, legal_tensor)
    max_value = torch.max(legal_values)
    best_indices = torch.nonzero(legal_values == max_value, as_tuple=False).flatten().tolist()
    return legal_actions[random.choice(best_indices)]


def terminal_reward(winner: int | None, agent_player: int) -> float:
    if winner == agent_player:
        return 1.0
    if winner is None:
        return 0.0
    return -1.0



def select_actor_action(
    model: ActorCriticNet,
    observation: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one legal action and return action, log_prob, entropy, value."""

    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
    logits, value = model(obs_tensor)
    masked_logits = logits[0].masked_fill(~mask_tensor, -1.0e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action), dist.entropy(), value.squeeze(0)


def optimise_actor_critic_step(
    model: ActorCriticNet,
    optimiser: optim.Optimizer,
    log_prob: torch.Tensor,
    entropy: torch.Tensor,
    value: torch.Tensor,
    reward: float,
    next_observation: np.ndarray | None,
    done: bool,
    gamma: float,
    value_loss_weight: float,
    entropy_weight: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Run one one-step actor-critic update.

    The target is reward + gamma * V(next_state), unless the game ended.
    The actor is updated with -log_prob(action) * advantage.
    The critic is updated with squared error against the bootstrapped target.
    """

    with torch.no_grad():
        if done or next_observation is None:
            target = torch.tensor(float(reward), dtype=torch.float32, device=device)
        else:
            next_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = model(next_tensor)
            target = torch.tensor(float(reward), dtype=torch.float32, device=device) + gamma * next_value.squeeze(0)

    advantage = target - value
    actor_loss = -log_prob * advantage.detach()
    critic_loss = advantage.pow(2)
    entropy_loss = -entropy
    loss = actor_loss + value_loss_weight * critic_loss + entropy_weight * entropy_loss

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimiser.step()

    return (
        float(loss.detach().cpu()),
        float(actor_loss.detach().cpu()),
        float(critic_loss.detach().cpu()),
        float(entropy.detach().cpu()),
    )


def train_actor_critic_vs_random(
    episodes: int,
    gamma: float,
    learning_rate: float,
    value_loss_weight: float,
    entropy_weight: float,
    tactical_training_ratio: float = 0.0,
    seed: int | None = None,
    device_name: str = "auto",
    training_plot_path: str | Path | None = None,
    training_metrics_csv: str | Path | None = None,
) -> ActorCriticNet:
    """Train actor-critic on symbolic observations against heuristic-random.

    The opponent is treated as part of the environment. Each actor-critic update
    follows one learner move plus the opponent response, so the next state is
    again from the learner's perspective.
    """

    if not 0.0 <= tactical_training_ratio <= 1.0:
        raise ValueError("tactical_training_ratio must be between 0.0 and 1.0")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = resolve_device(device_name)
    model = ActorCriticNet().to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    random_player = RandomPlayer()
    positions = tactical_positions()

    wins = 0
    losses = 0
    draws = 0
    tactical_episodes = 0
    learner_steps = 0
    recent_total_losses: list[float] = []
    recent_actor_losses: list[float] = []
    recent_critic_losses: list[float] = []
    recent_entropies: list[float] = []
    history = TrainingHistory(
        steps=[],
        episodes=[],
        total_losses=[],
        actor_losses=[],
        critic_losses=[],
        entropies=[],
        rewards=[],
    )

    for episode in range(1, episodes + 1):
        use_tactical_start = random.random() < tactical_training_ratio

        if use_tactical_start:
            tactical_episodes += 1
            position = random.choice(positions)
            env = make_env_from_position(position)
            agent_player = env.current_player
            LOGGER.debug("episode=%s actor_critic tactical_start=%s", episode, position.name)
        else:
            env = ConnectFourEnv(observation_mode="current_player")
            agent_player = PLAYER_ONE if episode % 2 else PLAYER_TWO
            env.reset(starting_player=PLAYER_ONE)
            if agent_player == PLAYER_TWO:
                env.step(random_player.choose_action(env))

        while not env.done:
            observation = observation_from_env(env)
            legal_mask = env.legal_action_mask()
            model.train()
            action, log_prob, entropy, value = select_actor_action(model, observation, legal_mask, device)
            agent_result = env.step(action)

            if agent_result.done:
                reward = terminal_reward(env.last_winner, agent_player)
                next_observation = None
                done = True
            else:
                opponent_action = random_player.choose_action(env)
                opponent_result = env.step(opponent_action)
                if opponent_result.done:
                    reward = terminal_reward(env.last_winner, agent_player)
                    next_observation = None
                    done = True
                else:
                    reward = 0.0
                    next_observation = observation_from_env(env)
                    done = False

            total_loss, actor_loss, critic_loss, ent = optimise_actor_critic_step(
                model=model,
                optimiser=optimiser,
                log_prob=log_prob,
                entropy=entropy,
                value=value,
                reward=reward,
                next_observation=next_observation,
                done=done,
                gamma=gamma,
                value_loss_weight=value_loss_weight,
                entropy_weight=entropy_weight,
                device=device,
            )
            learner_steps += 1
            history.steps.append(learner_steps)
            history.episodes.append(episode)
            history.total_losses.append(total_loss)
            history.actor_losses.append(actor_loss)
            history.critic_losses.append(critic_loss)
            history.entropies.append(ent)
            history.rewards.append(float(reward))
            recent_total_losses.append(total_loss)
            recent_actor_losses.append(actor_loss)
            recent_critic_losses.append(critic_loss)
            recent_entropies.append(ent)
            if len(recent_total_losses) > 500:
                recent_total_losses.pop(0)
                recent_actor_losses.pop(0)
                recent_critic_losses.pop(0)
                recent_entropies.pop(0)

        if env.last_winner == agent_player:
            wins += 1
        elif env.last_winner is None:
            draws += 1
        else:
            losses += 1

        if episode == 1 or episode % max(1, episodes // 10) == 0:
            mean_total = sum(recent_total_losses) / len(recent_total_losses) if recent_total_losses else float("nan")
            mean_actor = sum(recent_actor_losses) / len(recent_actor_losses) if recent_actor_losses else float("nan")
            mean_critic = sum(recent_critic_losses) / len(recent_critic_losses) if recent_critic_losses else float("nan")
            mean_entropy = sum(recent_entropies) / len(recent_entropies) if recent_entropies else float("nan")
            LOGGER.info(
                "episode=%s/%s steps=%s wins=%s losses=%s draws=%s tactical_starts=%s loss=%.5f actor=%.5f critic=%.5f entropy=%.5f",
                episode,
                episodes,
                learner_steps,
                wins,
                losses,
                draws,
                tactical_episodes,
                mean_total,
                mean_actor,
                mean_critic,
                mean_entropy,
            )

    model.eval()
    if training_metrics_csv is not None:
        write_training_metrics_csv(history, training_metrics_csv)
    if training_plot_path is not None:
        plot_training_history(history, training_plot_path)
    return model



def moving_average(values: list[float], window: int) -> tuple[list[int], list[float]]:
    if not values:
        return [], []
    window = max(1, min(window, len(values)))
    averaged: list[float] = []
    indices: list[int] = []
    running_sum = 0.0
    queue: list[float] = []
    for index, value in enumerate(values, start=1):
        queue.append(float(value))
        running_sum += float(value)
        if len(queue) > window:
            running_sum -= queue.pop(0)
        averaged.append(running_sum / len(queue))
        indices.append(index)
    return indices, averaged


def plot_training_history(history: TrainingHistory, output_path: str | Path, smoothing_window: int = 200) -> None:
    """Save a matplotlib training diagnostics plot.

    The plot shows smoothed total, actor, and critic losses. Entropy is saved as
    a separate plot beside it because it has a different scale from the losses.
    """

    if not history.steps:
        LOGGER.warning("no training history available; not writing plot")
        return

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs, total = moving_average(history.total_losses, smoothing_window)
    _, actor = moving_average(history.actor_losses, smoothing_window)
    _, critic = moving_average(history.critic_losses, smoothing_window)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, total, label="total loss")
    plt.plot(xs, actor, label="actor loss")
    plt.plot(xs, critic, label="critic loss")
    plt.xlabel("Learner update step")
    plt.ylabel("Smoothed loss")
    plt.title("Actor-critic training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    entropy_path = output_path.with_name(f"{output_path.stem}_entropy{output_path.suffix}")
    xs, entropy = moving_average(history.entropies, smoothing_window)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, entropy, label="entropy")
    plt.xlabel("Learner update step")
    plt.ylabel("Smoothed entropy")
    plt.title("Actor policy entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(entropy_path, dpi=150)
    plt.close()

    LOGGER.info("saved training loss plot to %s", output_path)
    LOGGER.info("saved entropy plot to %s", entropy_path)


def write_training_metrics_csv(history: TrainingHistory, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("step,episode,total_loss,actor_loss,critic_loss,entropy,reward\n")
        for row in zip(
            history.steps,
            history.episodes,
            history.total_losses,
            history.actor_losses,
            history.critic_losses,
            history.entropies,
            history.rewards,
        ):
            file.write(",".join(str(value) for value in row) + "\n")
    LOGGER.info("saved training metrics CSV to %s", output_path)


def actor_critic_snapshot(
    model: ActorCriticNet,
    env: ConnectFourEnv,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    obs = observation_from_env(env)
    legal_mask = env.legal_action_mask()
    mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, value = model(obs_tensor)
        masked_logits = logits[0].masked_fill(~mask_tensor, -1.0e9)
        probs = torch.softmax(masked_logits, dim=0)
    return logits[0].detach().cpu().numpy(), probs.detach().cpu().numpy(), float(value.item())


def plot_actor_critic_move(
    model: ActorCriticNet,
    env: ConnectFourEnv,
    move_index: int,
    player_id: int,
    action: int | None,
    device_name: str,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot actor probabilities and critic value for the current board."""

    if output_dir is None and not show:
        return

    import matplotlib.pyplot as plt

    device = resolve_device(device_name)
    _, probs, value = actor_critic_snapshot(model, env, device)
    columns = list(range(COLS))

    suffix = f"move_{move_index:03d}_player_{player_id}"
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        actor_path = output_path / f"{suffix}_actor.png"
        critic_path = output_path / f"{suffix}_critic.png"
    else:
        actor_path = None
        critic_path = None

    plt.figure(figsize=(8, 5))
    plt.bar(columns, probs)
    if action is not None:
        plt.axvline(action, linestyle="--", label=f"chosen column {action}")
        plt.legend()
    plt.xticks(columns)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Column")
    plt.ylabel("Actor probability")
    plt.title(f"Actor move probabilities, move {move_index}, player {player_id}")
    plt.tight_layout()
    if actor_path is not None:
        plt.savefig(actor_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.bar(["V(state)"], [value])
    plt.axhline(0.0, linewidth=1)
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Critic value")
    plt.title(f"Critic board score, move {move_index}, player {player_id}")
    plt.tight_layout()
    if critic_path is not None:
        plt.savefig(critic_path, dpi=150)
    if show:
        plt.show()
    plt.close()



def evaluate_actor_critic_vs_random(
    model: ActorCriticNet,
    games: int,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = resolve_device(device_name)
    ac_player = ActorCriticPlayer(model, device=device, sample=False)
    random_player = RandomPlayer()
    wins = 0
    losses = 0
    draws = 0

    for game in range(1, games + 1):
        env = ConnectFourEnv(observation_mode="current_player")
        agent_player = PLAYER_ONE if game % 2 else PLAYER_TWO
        env.reset(starting_player=PLAYER_ONE)

        while not env.done:
            if env.current_player == agent_player:
                action = ac_player.choose_action(env)
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def save_actor_critic_model(model: ActorCriticNet, path: str | Path) -> None:
    payload = {
        "rows": ROWS,
        "cols": COLS,
        "observation_mode": "current_player",
        "model_state_dict": model.cpu().state_dict(),
    }
    torch.save(payload, Path(path))


def load_actor_critic_model(path: str | Path, device_name: str = "auto") -> ActorCriticNet:
    device = resolve_device(device_name)
    payload = torch.load(Path(path), map_location=device)
    if payload.get("rows") != ROWS or payload.get("cols") != COLS:
        raise ValueError("Actor-critic model board dimensions do not match this environment")
    model = ActorCriticNet().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model

def make_player(
    kind: PlayerKind,
    ac_model: ActorCriticNet | None = None,
    device_name: str = "auto",
):
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer()
    if kind == "uniform_random":
        return UniformRandomPlayer()
    if kind == "ac":
        if ac_model is None:
            raise ValueError("ac player requires a loaded actor-critic model")
        return ActorCriticPlayer(ac_model, device=resolve_device(device_name))
    raise ValueError(f"unknown player kind: {kind}")


def play_game(
    player_one: PlayerKind,
    player_two: PlayerKind,
    observation_mode: ObservationMode = "current_player",
    seed: int | None = None,
    verbose: bool = True,
    ac_model: ActorCriticNet | None = None,
    device_name: str = "auto",
    show_ac_plots: bool = False,
    save_ac_plots_dir: str | Path | None = None,
) -> int | None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = ConnectFourEnv(observation_mode=observation_mode)
    players = {
        PLAYER_ONE: make_player(player_one, ac_model=ac_model, device_name=device_name),
        PLAYER_TWO: make_player(player_two, ac_model=ac_model, device_name=device_name),
    }

    env.reset()

    if verbose:
        print(env.render_text())
        print()

    move_index = 0
    while not env.done:
        player_id = env.current_player
        player = players[player_id]
        action = player.choose_action(env)
        if ac_model is not None and (show_ac_plots or save_ac_plots_dir is not None) and isinstance(player, ActorCriticPlayer):
            plot_actor_critic_move(
                model=ac_model,
                env=env,
                move_index=move_index,
                player_id=player_id,
                action=action,
                device_name=device_name,
                output_dir=save_ac_plots_dir,
                show=show_ac_plots,
            )
        result = env.step(action)
        move_index += 1

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
    ac_model: ActorCriticNet | None = None,
    seed: int | None = None,
    verbose: bool = False,
    device_name: str = "auto",
) -> dict[str, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    player = make_player(player_kind, ac_model=ac_model, device_name=device_name)
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
    parser = argparse.ArgumentParser(description="Minimal Connect Four actor-critic environment")
    parser.add_argument(
        "--p1",
        choices=["human", "random", "uniform_random", "ac"],
        default="human",
        help="Player 1 controller. random means heuristic-random; uniform_random is pure random.",
    )
    parser.add_argument(
        "--p2",
        choices=["human", "random", "uniform_random", "ac"],
        default="random",
        help="Player 2 controller. random means heuristic-random; uniform_random is pure random.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["absolute", "current_player"],
        default="current_player",
        help="Observation encoding used by the environment",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--random-smoke-test",
        type=int,
        default=0,
        help="Run N random-vs-random games without printing boards",
    )
    parser.add_argument(
        "--train-ac",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Train an actor-critic agent against the heuristic-random opponent using symbolic observations",
    )
    parser.add_argument(
        "--eval-ac",
        type=int,
        default=0,
        metavar="GAMES",
        help="Evaluate a loaded or newly trained actor-critic model against the heuristic-random opponent",
    )
    parser.add_argument(
        "--eval-tactics",
        choices=["human", "random", "uniform_random", "ac"],
        default=None,
        metavar="PLAYER",
        help="Evaluate one player against fixed tactical positions",
    )
    parser.add_argument(
        "--ac-model",
        default="connect_four_actor_critic.pt",
        help="Path used to save/load an actor-critic model",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimiser learning rate")
    parser.add_argument("--value-loss-weight", type=float, default=0.5, help="Weight on critic/value loss")
    parser.add_argument("--entropy-weight", type=float, default=0.01, help="Weight on entropy exploration bonus")
    parser.add_argument(
        "--training-plot",
        default=None,
        help="Optional PNG path for a matplotlib plot of actor-critic training losses",
    )
    parser.add_argument(
        "--training-metrics-csv",
        default=None,
        help="Optional CSV path for raw per-update training metrics",
    )
    parser.add_argument(
        "--show-ac-plots",
        action="store_true",
        help="Show matplotlib actor/critic plots for each actor-critic move during single-game play",
    )
    parser.add_argument(
        "--save-ac-plots-dir",
        default=None,
        help="Optional directory to save actor/critic PNG plots for each actor-critic move during single-game play",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, etc.")
    parser.add_argument(
        "--tactical-training-ratio",
        type=float,
        default=0.0,
        help="Fraction of training episodes that start from fixed tactical positions",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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

    ac_model: ActorCriticNet | None = None

    if args.random_smoke_test > 0:
        run_random_smoke_tests(args.random_smoke_test, seed=args.seed)
        return

    if args.train_ac > 0:
        ac_model = train_actor_critic_vs_random(
            episodes=args.train_ac,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            value_loss_weight=args.value_loss_weight,
            entropy_weight=args.entropy_weight,
            tactical_training_ratio=args.tactical_training_ratio,
            seed=args.seed,
            device_name=args.device,
            training_plot_path=args.training_plot,
            training_metrics_csv=args.training_metrics_csv,
        )
        save_actor_critic_model(ac_model, args.ac_model)
        print(f"Saved actor-critic model to {args.ac_model}")

    if args.eval_ac > 0:
        if ac_model is None:
            ac_model = load_actor_critic_model(args.ac_model, device_name=args.device)
        results = evaluate_actor_critic_vs_random(ac_model, games=args.eval_ac, seed=args.seed, device_name=args.device)
        print(f"Actor-critic vs heuristic-random over {args.eval_ac} games")
        print(f"Wins:   {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Draws:  {results['draws']}")

    if args.eval_tactics is not None:
        if args.eval_tactics == "ac" and ac_model is None:
            ac_model = load_actor_critic_model(args.ac_model, device_name=args.device)
        evaluate_tactical_positions(
            player_kind=args.eval_tactics,
            ac_model=ac_model,
            seed=args.seed,
            verbose=args.verbose,
            device_name=args.device,
        )
        return

    if args.train_ac > 0 or args.eval_ac > 0:
        return

    if args.p1 == "ac" or args.p2 == "ac":
        if ac_model is None:
            ac_model = load_actor_critic_model(args.ac_model, device_name=args.device)

    play_game(
        player_one=args.p1,
        player_two=args.p2,
        observation_mode=args.observation_mode,
        seed=args.seed,
        verbose=not args.quiet_board,
        ac_model=ac_model,
        device_name=args.device,
        show_ac_plots=args.show_ac_plots,
        save_ac_plots_dir=args.save_ac_plots_dir,
    )


if __name__ == "__main__":
    main()
