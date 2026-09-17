"""
ILC+CBF - Dynamic Obstacle Avoidance Demo.

Large robot (4.6m x 1.6m) navigates from [10,10] to [40,40] through
dynamic (moving) obstacles using:
- Straight-line reference path
- ILC trajectory learning + default steer-first StaticCBFFilter
- DynamicCBFFilter: CPA-based collision prediction + perpendicular avoidance

Visualization:
- Speed-colored trail: blue (slow) → red (fast), colorbar 0-5 m/s
"""

import argparse
import json
import random
import sys

if "--headless" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import irsim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ilc_planner import ILCController


def generate_straight_path(robot_start, goal_xy, n=100):
    """Generate a straight-line reference path from start to goal."""
    ref_path = []
    for i in range(n):
        a = i / (n - 1)
        x = robot_start[0] + a * (goal_xy[0] - robot_start[0])
        y = robot_start[1] + a * (goal_xy[1] - robot_start[1])
        theta = np.arctan2(goal_xy[1] - robot_start[1],
                           goal_xy[0] - robot_start[0])
        ref_path.append(np.array([[x], [y], [theta]]))
    return ref_path


def main(headless=False, seed=None, max_steps=600, output=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if headless:
        plt.switch_backend("Agg")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    output_dir = Path(output).resolve() if output is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(Path(__file__).with_name('dynamic_obs.yaml'))
    env = irsim.make(yaml_path, save_ani=False, display=not headless,
                     full=False, seed=seed)

    robot_start = env.robot.state.flatten()[:3]
    goal_xy = env.robot.goal.flatten()[:2]

    # ---- Straight-line reference path ----
    ref_path_list = generate_straight_path(robot_start, goal_xy, n=100)
    print(f"[Path] Straight line: {len(ref_path_list)} waypoints")

    # Draw start point and goal marker
    ax = plt.gca()
    ax.plot(robot_start[0], robot_start[1], 'o', color='green',
            markersize=10, zorder=10)
    ax.plot(goal_xy[0], goal_xy[1], marker='*', color='red', markersize=18,
            markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)
    ax.text(goal_xy[0] + 1.5, goal_xy[1] + 1.5, 'GOAL', fontsize=14,
            color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ---- ILC+CBF Controller ----
    controller = ILCController(
        ref_path_list,
        sample_time=env.step_time,
        v_init=4.5,
        v_max=8.0,
        k_yaw=1.2,
        goal_tolerance=0.5,
        ilc_iterations=5,
    )
    # Retain the original default static and dynamic filters.

    # ---- Record positions + speeds + obstacle trails ----
    speeds_recorded = []
    positions_xy = []
    obs_trails = []

    # ---- Main loop ----
    print("\n[ILC+CBF] Starting simulation...")
    for i in range(max_steps):
        lidar_scan = env.get_lidar_scan()
        obs_list = env.get_obstacle_info_list()

        if i == 0:
            obs_trails = [[] for _ in range(len(obs_list))]
        for j, obs in enumerate(obs_list):
            if j < len(obs_trails):
                oc = np.asarray(obs.center).flatten()
                r = float(getattr(obs, 'radius', 1.0) or 1.0)
                obs_trails[j].append((float(oc[0]), float(oc[1]), r))

        opt_vel, info = controller.control(
            env.robot.state, ref_speed=4.5,
            obstacle_list=obs_list, lidar_scan=lidar_scan,
        )

        # Record position and speed for trail coloring
        pos = env.robot.state.flatten()
        speeds_recorded.append(opt_vel[0, 0])
        positions_xy.append((pos[0], pos[1]))

        env.step(opt_vel)
        env.render(show_traj=True, show_trail=True)

        if i % 50 == 0:
            p = env.robot.state.flatten()
            d = np.sqrt((p[0] - goal_xy[0]) ** 2
                        + (p[1] - goal_xy[1]) ** 2)
            n_dyn = sum(1 for obs in obs_list
                        if np.linalg.norm(np.asarray(obs.velocity).flatten()) > 0.02)
            print(f'  Step {i}: ({p[0]:.1f},{p[1]:.1f}) '
                  f'v={opt_vel[0,0]:.2f} d={d:.1f}m dyn_obs={n_dyn}')

        if env.done():
            print(f'[ILC+CBF] Done at step {i}')
            break
        if info['arrive']:
            print(f'[ILC+CBF] Goal reached at step {i}!')
            break

    # ---- Overlay speed-colored trail (blue=slow, red=fast) ----
    if len(positions_xy) > 1:
        _ax = plt.gca()
        px_trail = [p[0] for p in positions_xy]
        py_trail = [p[1] for p in positions_xy]
        speeds = np.array(speeds_recorded)
        norm = plt.Normalize(0, 5.0)
        cmap = plt.cm.jet

        for j in range(len(px_trail) - 1):
            c = cmap(norm(speeds[j]))
            _ax.plot(px_trail[j:j + 2], py_trail[j:j + 2], color=c,
                     linewidth=3, alpha=0.85, zorder=5)

        # Colorbar (speed legend only)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=_ax, shrink=0.5, pad=0.02)
        cbar.set_label('Speed (m/s)', fontsize=10)

        # ---- Ghost trails: dynamic obstacle movement ----
        ghost_colors = ['dodgerblue', 'royalblue', 'steelblue',
                        'deepskyblue', 'lightblue',
                        'cornflowerblue', 'lightskyblue', 'skyblue',
                        'paleturquoise', 'powderblue']
        for j, trail in enumerate(obs_trails):
            if len(trail) > 3:
                color = ghost_colors[j % len(ghost_colors)]
                n_samples = min(6, len(trail))
                stride = max(1, len(trail) // n_samples)
                obs_r = trail[-1][2]
                for k in range(0, len(trail), stride):
                    frac = k / max(len(trail) - 1, 1)
                    alpha = 0.04 + 0.30 * frac
                    r = obs_r
                    circle = plt.Circle((trail[k][0], trail[k][1]), r,
                                        color=color, alpha=alpha,
                                        zorder=2, ec='none')
                    _ax.add_patch(circle)

        plt.draw()

        # Save only when explicitly requested.
        if output_dir is not None:
            fig_path = output_dir / 'speed_trail_overlay_ilc_cbf.png'
            plt.gcf().savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f'[Visual] Speed trail + legend saved: {fig_path}')

    p = env.robot.state.flatten()
    d = np.sqrt((p[0] - goal_xy[0]) ** 2 + (p[1] - goal_xy[1]) ** 2)
    print(f'Final: ({p[0]:.1f},{p[1]:.1f}) d={d:.2f}m '
          f'collision={env.robot.collision}')

    result = {
        "steps": i + 1, "goal_distance_m": float(d),
        "collision": bool(env.robot.collision),
        "arrived": bool(info['arrive'] or env.robot.arrive),
        "seed": seed,
    }
    if output_dir is not None:
        (output_dir / 'result.json').write_text(
            json.dumps(result, indent=2) + "\n", encoding='utf-8')
    env.end(show_traj=True, show_trail=True, ending_time=0 if headless else 10)
    plt.close('all')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--output', type=Path, default=None,
                        help='Optionally save PNG and JSON to this directory')
    args = parser.parse_args()
    main(**vars(args))
