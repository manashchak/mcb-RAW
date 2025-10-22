# %% Imports
import os
import math
import pickle
from collections import defaultdict, deque
import numpy as np
import matplotlib.pylab as plt
import gymnasium as gym



# %% Utilities
def seed_everything(seed: int):
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def reset_env(env, seed=None):
    # Gymnasium returns (obs, info); older gym returns obs
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        obs = env.reset()
    return obs


def unpack_step(step_result):
    # Gymnasium: (obs, reward, terminated, truncated, info)
    # Gym:       (obs, reward, done, info)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result
    return obs, reward, done, info

def _as_char(x):
    """
    Normalize a tile value from env.unwrapped.desc to a Python str: 'S','F','H','G'.
    Handles bytes/np.bytes_, str, and integer codepoints robustly.
    """
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    if isinstance(x, (np.integer, int)):
        return chr(int(x))
    # Fallback
    return str(x)



# %% Reward shaping wrappers

# -----------------------------
# Reward Shaping Wrappers (fixed)
# -----------------------------
class PotentialBasedRewardWrapper(gym.Wrapper):
    """
    Potential-based shaping:
      r' = r + gamma * Phi(s') - Phi(s)
    with Phi(s) = -kappa * ManhattanDistance(s, goal).
    Adds dense guidance without changing the optimal policy.
    Also forwards the original env reward as info['orig_reward'].
    """
    def __init__(self, env, gamma=0.99, kappa=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.kappa = kappa
        self._desc = getattr(env.unwrapped, "desc", None)
        if self._desc is None:
            # Fallback: infer square grid
            self.rows = self.cols = int(math.sqrt(env.observation_space.n))
            self.goal_pos = (self.rows - 1, self.cols - 1)
        else:
            self.rows, self.cols = self._desc.shape
            # Find goal 'G' robustly
            gp = (self.rows - 1, self.cols - 1)
            found = False
            for r in range(self.rows):
                for c in range(self.cols):
                    if _as_char(self._desc[r, c]) == "G":
                        gp = (r, c)
                        found = True
                        break
                if found:
                    break
            self.goal_pos = gp
        self._last_s = None

    def _to_pos(self, s):
        return (s // self.cols, s % self.cols)

    def _phi(self, s):
        r, c = self._to_pos(s)
        gr, gc = self.goal_pos
        manhattan = abs(r - gr) + abs(c - gc)
        return -self.kappa * float(manhattan)

    def reset(self, **kwargs):
        s = reset_env(self.env, seed=kwargs.get("seed", None))
        self._last_s = s
        return s, {}

    def step(self, action):
        if self._last_s is None:
            self._last_s, _ = self.reset()
        s = self._last_s
        sp, r, done, info = unpack_step(self.env.step(action))
        info = dict(info) if info else {}
        info["orig_reward"] = r
        shaped = r + self.gamma * (0.0 if done else self._phi(sp)) - self._phi(s)
        self._last_s = None if done else sp
        return sp, shaped, done, info




class DenseRewardWrapper(gym.Wrapper):
    """
    Dense shaping (pragmatic):
    shaped = r + step_penalty (+ hole_penalty or goal_bonus at terminal)
    Note: can change the optimal policy slightly but speeds up learning.
    Adds info['orig_reward'] for convenience.
    """
    def __init__(self, env, step_penalty=-0.01, hole_penalty=-1.0, goal_bonus=0.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        self.goal_bonus = goal_bonus
        self._desc = getattr(env.unwrapped, "desc", None)

    def reset(self, **kwargs):
        return reset_env(self.env, seed=kwargs.get("seed", None)), {}

    def step(self, action):
        sp, r, done, info = unpack_step(self.env.step(action))
        info = dict(info) if info else {}
        info["orig_reward"] = r
        shaped = r + self.step_penalty
        if done and self._desc is not None:
            cols = self._desc.shape[1]
            rr, cc = sp // cols, sp % cols
            tile = _as_char(self._desc[rr, cc])
            if tile == "H":
                shaped += self.hole_penalty
            elif tile == "G":
                shaped += self.goal_bonus

        return sp, shaped, done, info



# %% Environment factory
def make_env(
    map_name="8x8",
    is_slippery=False,
    reward_mode="potential",  # 'potential' | 'dense' | 'none'
    gamma=0.99,
    kappa=1.0,
    step_penalty=-0.01,
    hole_penalty=-1.0,
    goal_bonus=0.0,
    render=False,
    seed=42,
):
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="human" if render else None,
    )
    if reward_mode == "potential":
        env = PotentialBasedRewardWrapper(env, gamma=gamma, kappa=kappa)
    elif reward_mode == "dense":
        env = DenseRewardWrapper(env, step_penalty=step_penalty, hole_penalty=hole_penalty, goal_bonus=goal_bonus)
    # else 'none' -> leave unshaped
    # Seed env & spaces
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env


# %% Agent (Double Q with options)
class TabularAgent:
    def __init__(
        self,
        nS,
        nA,
        algo="doubleq",          # 'doubleq' | 'q' (vanilla Q-learning)
        gamma=0.99,
        alpha_init=0.8,
        alpha_min=0.05,
        alpha_visit_decay=0.02,  # lr(s,a) = alpha_init / (1 + alpha_visit_decay * visits)
        eps_init=1.0,
        eps_min=0.05,
        eps_decay=0.0025,        # epsilon(t) = max(eps_min, eps_init * exp(-eps_decay * t))
        optimistic_init=0.0,
        seed=42,
    ):
        self.nS, self.nA = nS, nA
        self.algo = algo
        self.gamma = gamma
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_visit_decay = alpha_visit_decay
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

        self.Q = np.full((nS, nA), optimistic_init, dtype=np.float64)
        self.Q2 = np.full((nS, nA), optimistic_init, dtype=np.float64) if algo == "doubleq" else None
        self.visits = np.zeros((nS, nA), dtype=np.int32)

    def epsilon(self, episode_idx: int) -> float:
        return max(self.eps_min, self.eps_init * math.exp(-self.eps_decay * episode_idx))

    def alpha(self, s, a) -> float:
        v = self.visits[s, a]
        lr = self.alpha_init / (1.0 + self.alpha_visit_decay * v)
        return max(self.alpha_min, lr)

    def select_action(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.nA))
        if self.algo == "doubleq" and self.Q2 is not None:
            q_mean = (self.Q[s] + self.Q2[s]) / 2.0
            return int(np.argmax(q_mean))
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, sp, done):
        self.visits[s, a] += 1
        lr = self.alpha(s, a)

        if self.algo == "q" or self.Q2 is None:
            target = r if done else (r + self.gamma * np.max(self.Q[sp]))
            self.Q[s, a] += lr * (target - self.Q[s, a])
            return

        # Double Q-learning (Hasselt)
        if self.rng.random() < 0.5:
            if done:
                target = r
            else:
                a_star = int(np.argmax(self.Q[sp]))
                target = r + self.gamma * self.Q2[sp, a_star]
            self.Q[s, a] += lr * (target - self.Q[s, a])
        else:
            if done:
                target = r
            else:
                a_star = int(np.argmax(self.Q2[sp]))
                target = r + self.gamma * self.Q[sp, a_star]
            self.Q2[s, a] += lr * (target - self.Q2[s, a])

    def greedy_policy(self):
        if self.Q2 is not None:
            return np.argmax((self.Q + self.Q2) / 2.0, axis=1)
        return np.argmax(self.Q, axis=1)

    def q_values(self):
        if self.Q2 is not None:
            return (self.Q + self.Q2) / 2.0
        return self.Q


# %% Evaluation
def evaluate_policy(env_eval, policy, episodes=200, seed=123):
    successes, total_reward = 0, 0.0
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        s = reset_env(env_eval, seed=int(rng.integers(0, 10_000_000)))
        done = False
        ep_reward = 0.0
        while not done:
            a = policy[s]
            s, r, done, _ = unpack_step(env_eval.step(a))
            ep_reward += r
            if done and r > 0.0:
                successes += 1
        total_reward += ep_reward
    return successes / episodes, total_reward / episodes


def render_policy_grid(policy, rows, cols):
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    lines = []
    for r in range(rows):
        lines.append(" ".join(arrows[int(policy[r * cols + c])] for c in range(cols)))
    return "\n".join(lines)


# %% Main training entry (keeps your original run(...) signature)
def run(episodes, is_training=True, render=False):
    """
    Run training or evaluation.

    Usage (same as before):
        run(25000)  # train
        run(1, is_training=False, render=True)  # load and render greedy play
    """
    # ---- Config (tweak here) ----
    cfg = dict(
        map_name="8x8",              # your original default
        is_slippery=False,           # set True for tougher, stochastic dynamics
        reward_mode="potential",     # 'potential' | 'dense' | 'none'
        gamma=0.99,
        kappa=1.0,                   # potential strength
        step_penalty=-0.01,          # only for 'dense' mode
        hole_penalty=-1.0,           # only for 'dense' mode
        goal_bonus=0.0,              # only for 'dense' mode
        algo="doubleq",              # 'doubleq' (default) or 'q'
        alpha_init=0.8,
        alpha_min=0.05,
        alpha_visit_decay=0.02,
        eps_init=1.0,
        eps_min=0.05,
        eps_decay=0.0025,
        optimistic_init=0.0,
        max_steps=200,
        eval_every=500,
        eval_episodes=200,
        early_stop_target=0.90,      # target success rate on unshaped env
        early_stop_window=3,         # consecutive evals above target
        anti_stagnation_patience=1500,  # steps without success -> temporarily boost ε
        seed=8,
        # Paths
        q_path="Learnings/RL/Gymnasium/frozen_lake_8x8.pkl",
        plot_path_train="Learnings/RL/Gymnasium/frozen_lake_8x8_training.png",
        plot_path_eval="Learnings/RL/Gymnasium/frozen_lake_8x8_eval.png",
    )
    seed_everything(cfg["seed"])

    # ---- Environments ----
    env_train = make_env(
        map_name=cfg["map_name"],
        is_slippery=cfg["is_slippery"],
        reward_mode=("none" if not is_training else cfg["reward_mode"]),
        gamma=cfg["gamma"],
        kappa=cfg["kappa"],
        step_penalty=cfg["step_penalty"],
        hole_penalty=cfg["hole_penalty"],
        goal_bonus=cfg["goal_bonus"],
        render=False,  # training renders are slow; use eval render instead
        seed=cfg["seed"],
    )

    env_eval = make_env(
        map_name=cfg["map_name"],
        is_slippery=cfg["is_slippery"],
        reward_mode="none",   # never shape the evaluation env
        render=render,
        seed=cfg["seed"] + 1,
    )

    nS, nA = env_train.observation_space.n, env_train.action_space.n
    rows = cols = int(math.sqrt(nS))  # FrozenLake is square
    # ---- Agent / Q-table ----
    if is_training:
        agent = TabularAgent(
            nS=nS, nA=nA,
            algo=cfg["algo"], gamma=cfg["gamma"],
            alpha_init=cfg["alpha_init"], alpha_min=cfg["alpha_min"], alpha_visit_decay=cfg["alpha_visit_decay"],
            eps_init=cfg["eps_init"], eps_min=cfg["eps_min"], eps_decay=cfg["eps_decay"],
            optimistic_init=cfg["optimistic_init"], seed=cfg["seed"],
        )
    else:
        # Load and evaluate
        with open(cfg["q_path"], "rb") as f:
            Q = pickle.load(f)
        # Minimal agent wrapper for action selection:
        class Greedy:
            def __init__(self, Q): self.Q = np.asarray(Q)
            def greedy_policy(self): return np.argmax(self.Q, axis=1)
        agent = Greedy(Q)

    # ---- Tracking for plots ----
    eval_points = []
    eval_successes = []
    eval_avg_rewards = []

    # ---- Training loop ----
    if is_training:
        steps_since_success = 0
        success_streak = 0

        for ep in range(1, episodes + 1):
            s = reset_env(env_train, seed=cfg["seed"] + ep)
            eps = agent.epsilon(ep)
            # Anti-stagnation: if stuck too long, temporarily boost exploration
            if steps_since_success > cfg["anti_stagnation_patience"]:
                eps = max(eps, 0.5)
                steps_since_success = 0

            for t in range(cfg["max_steps"]):
                a = agent.select_action(s, eps)
                sp, r, done, info = unpack_step(env_train.step(a))

                # Use shaped reward 'r' for training; we also look at orig_reward for success bookkeeping
                agent.update(s, a, r, sp, done)
                s = sp
                steps_since_success += 1
                if done:
                    # If the underlying env gave reward (goal), reset stagnation counter
                    if isinstance(info, dict) and info.get("orig_reward", 0.0) > 0.0:
                        steps_since_success = 0
                    break

            # Periodic evaluation on the *unshaped* env with greedy policy
            if ep % cfg["eval_every"] == 0:
                policy = agent.greedy_policy()
                succ_rate, avg_r = evaluate_policy(env_eval, policy, episodes=cfg["eval_episodes"], seed=cfg["seed"] + 1234 + ep)
                print(f"[Episode {ep:5d}/{episodes}] success={succ_rate:.2%} avgR={avg_r:.3f} eps={eps:.3f}")
                eval_points.append(ep)
                eval_successes.append(succ_rate)
                eval_avg_rewards.append(avg_r)

                # Early stopping logic
                if succ_rate >= cfg["early_stop_target"]:
                    success_streak += 1
                else:
                    success_streak = 0
                if success_streak >= cfg["early_stop_window"]:
                    print(f"Early stop at episode {ep}: success ≥ {cfg['early_stop_target']:.0%} "
                            f"for {cfg['early_stop_window']} consecutive evals.")
                    break

        # Save learned Q
        ensure_dir(cfg["q_path"])
        with open(cfg["q_path"], "wb") as f:
            pickle.dump(agent.q_values(), f)

        # Plot eval success over time
        if eval_points:
            plt.figure(figsize=(7, 4))
            plt.plot(eval_points, eval_successes, label="Success rate (eval)")
            plt.ylim(0, 1.05)
            plt.xlabel("Episodes")
            plt.ylabel("Success rate")
            plt.grid(True, alpha=0.3)
            plt.legend()
            ensure_dir(cfg["plot_path_train"])
            plt.tight_layout()
            plt.savefig(cfg["plot_path_train"])
            plt.close()

        # Final evaluation & policy print
        policy = agent.greedy_policy()
        succ_rate, avg_r = evaluate_policy(env_eval, policy, episodes=max(400, cfg["eval_episodes"]), seed=cfg["seed"] + 2025)
        print(f"\nFinal evaluation (unshaped): success={succ_rate:.2%}, avg_reward={avg_r:.3f}")
        print("\nGreedy policy (arrows):")
        print(render_policy_grid(policy, rows, cols))

    else:
        # Pure evaluation/rendering path
        policy = agent.greedy_policy()
        succ_rate, avg_r = evaluate_policy(env_eval, policy, episodes=400, seed=cfg["seed"] + 2025)
        print(f"Loaded Q. Evaluation (unshaped): success={succ_rate:.2%}, avg_reward={avg_r:.3f}")
        print("\nGreedy policy (arrows):")
        print(render_policy_grid(policy, rows, cols))

        # Plot (single point, for compatibility)
        plt.figure(figsize=(5, 3))
        plt.bar(["success"], [succ_rate])
        plt.ylim(0, 1.05)
        plt.tight_layout()
        ensure_dir(cfg["plot_path_eval"])
        plt.savefig(cfg["plot_path_eval"])
        plt.close()


# %% Main
if __name__ == "__main__":
    # Same call as your original script
    run(250000)