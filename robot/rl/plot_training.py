"""
plot_training.py -- reinforcement-learning training curves (from SB3 progress.csv).

    python -m robot.rl.plot_training --out docs/evidence/fig_rl_training.png

Left: success rate of the final residual PPO (upright / lying) vs the first attempt that learned
from scratch (v1, 0 %). Right: mean episode return of the final run.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    def col(k):
        return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    return {k: col(k) for k in ("time/total_timesteps", "rollout/success", "rollout/success_upright",
                                "rollout/success_lying", "rollout/ep_rew_mean")}


def smooth(y, k=9):
    y = np.asarray(y, float)
    out = np.copy(y)
    for i in range(len(y)):
        w = y[max(0, i - k + 1): i + 1]
        out[i] = np.nanmean(w) if np.any(~np.isnan(w)) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join("output", "rl", "ppo_can", "progress.csv"))
    ap.add_argument("--v1", default=os.path.join("output", "rl", "ppo_can_v1_plain", "progress.csv"))
    ap.add_argument("--out", default=os.path.join("docs", "evidence", "fig_rl_training.png"))
    a = ap.parse_args()
    f, v1 = load(a.final), load(a.v1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = f["time/total_timesteps"] / 1e6
    ax[0].plot(x, smooth(f["rollout/success_lying"]) * 100, label="누운 캔 (잔차 PPO)", color="#2a9d8f", lw=2)
    ax[0].plot(x, smooth(f["rollout/success_upright"]) * 100, label="서 있는 캔 (잔차 PPO)", color="#e76f51", lw=2)
    ax[0].plot(v1["time/total_timesteps"] / 1e6, smooth(v1["rollout/success"]) * 100,
               label="처음부터 학습 (1차 시도)", color="#6c757d", lw=2, ls="--")
    ax[0].axhline(42, color="#e76f51", lw=1, ls=":")
    ax[0].text(3.0, 44, "서 있는 캔: 스크립트만 쓸 때 42%", ha="center", fontsize=8, color="#e76f51")
    ax[0].set_xlabel("학습 스텝 (백만)"); ax[0].set_ylabel("성공률 (%)"); ax[0].set_ylim(-3, 103)
    ax[0].set_title("학습 중 성공률 (탐색 잡음 포함)"); ax[0].legend(loc="lower right", fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, smooth(f["rollout/ep_rew_mean"]), color="#264653", lw=2)
    ax[1].set_xlabel("학습 스텝 (백만)"); ax[1].set_ylabel("에피소드 평균 보상")
    ax[1].set_title("평균 보상"); ax[1].grid(alpha=.3)
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("->", a.out, f"({len(x)} points, final {x.max():.1f}M steps)")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    main()
