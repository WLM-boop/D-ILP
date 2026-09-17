"""
ILC+CBF (Optimized) - Static Obstacle Avoidance Demo.

Large robot (4.6m x 1.6m) navigates from [10,10] to [40,40] through
static obstacles using:
- RRT* global path planning (obstacle-aware reference path)
- ILC trajectory learning + First-Order CBF-QP safety filter
  (StaticCBFQPFilter: v-constraint only, no steer-first weaving)

Visualization:
- Smoothed RRT* reference path used internally (not plotted)
- Speed-colored trail: blue (slow) → red (fast), colorbar 0-6 m/s
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
from ilc_planner.cbf_filter import scan_box, StaticCBFQPFilter
from irsim.lib.path_planners import RRTStar


def smooth_path(x, y, window=5):
    """Smooth a 2D path with moving average."""
    x = np.array(x); y = np.array(y)
    kernel = np.ones(window) / window
    xs = np.convolve(x, kernel, mode='same')
    ys = np.convolve(y, kernel, mode='same')
    xs[:window] = x[:window]; ys[:window] = y[:window]
    xs[-window:] = x[-window:]; ys[-window:] = y[-window:]
    return xs, ys


def convert_rrt_path(trajectory):
    """Convert RRT* (2,N) output to ref_path format with smoothing."""
    if trajectory is None:
        return None
    xp = np.asarray(trajectory[0]).flatten()[::-1]
    yp = np.asarray(trajectory[1]).flatten()[::-1]
    if len(xp) < 2:
        return None
    xp, yp = smooth_path(xp, yp, window=4)
    ref_path = []
    prev_theta = 0.0
    for i in range(len(xp)):
        x, y = xp[i], yp[i]
        if i < len(xp) - 1:
            theta = np.arctan2(yp[i + 1] - yp[i], xp[i + 1] - xp[i])
        else:
            theta = prev_theta
        ref_path.append(np.array([[x], [y], [theta]]))
        prev_theta = theta
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
    yaml_path = str(Path(__file__).with_name('static_obs.yaml'))
    env = irsim.make(yaml_path, save_ani=False, display=not headless,
                     full=False, seed=seed)

    # ---- RRT* global path planning ----
    print("[RRT*] Planning global path...")
    env_map = env.get_map(resolution=0.1)
    planner = RRTStar(env_map, robot_radius=1.8, expand_dis=3.0, max_iter=3000)
    robot_start = env.robot.state.flatten()[:3]
    goal_xy = env.robot.goal.flatten()[:2]
    trajectory = planner.planning(robot_start, goal_xy, show_animation=False)
    if trajectory is None:
        print("[RRT*] Failed, using straight line")
        n = 100
        ref_path_list = []
        for i in range(n):
            a = i / (n - 1)
            x = robot_start[0] + a * (goal_xy[0] - robot_start[0])
            y = robot_start[1] + a * (goal_xy[1] - robot_start[1])
            ref_path_list.append(np.array([[x], [y], [
                np.arctan2(goal_xy[1] - robot_start[1],
                           goal_xy[0] - robot_start[0])]]))
    else:
        ref_path_list = convert_rrt_path(trajectory)
        print(f"[RRT*] Path: {len(ref_path_list)} waypoints")

    # Draw start point and goal marker
    ax = plt.gca()
    ax.plot(robot_start[0], robot_start[1], 'o', color='green',
            markersize=10, zorder=10)
    ax.plot(goal_xy[0], goal_xy[1], marker='*', color='red', markersize=18,
            markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)
    ax.text(goal_xy[0] + 1.5, goal_xy[1] + 1.5, 'GOAL', fontsize=14,
            color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ---- ILC+CBF Controller (optimized) ----
    controller = ILCController(
        ref_path_list,
        sample_time=env.step_time,
        v_init=7.0,
        v_max=8.0,
        k_yaw=1.2,
        goal_tolerance=1.0,
        ilc_iterations=5,
    )
    # Replace default steer-first CBF with first-order CBF-QP filter
    controller.static_cbf = StaticCBFQPFilter(
        R_safe=2.0, alpha_cbf=2.5, k_steer=2.0,
        fov_half=1.05, beam_step=4, emergency_rho=0.5)
    controller.dynamic_cbf = None  # no dynamic obstacles in this map

    # ---- Record positions + speeds for color-coded trail ----
    speeds_recorded = []
    positions_xy = []

    # ---- Main loop ----
    print("\n[ILC+CBF] Starting simulation...")
    for i in range(max_steps):
        lidar_scan = env.get_lidar_scan()
        lidar_obs = scan_box(env.robot.state, lidar_scan)
        obs_list = env.get_obstacle_info_list()

        opt_vel, info = controller.control(
            env.robot.state, ref_speed=5.0,
            obstacle_list=obs_list, lidar_scan=lidar_scan,
        )

        for obs in lidar_obs:
            env.draw_box(obs['vertex'], refresh=True)

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
            print(f'  Step {i}: ({p[0]:.1f},{p[1]:.1f}) '
                  f'v={opt_vel[0,0]:.2f} d={d:.1f}m')

        if env.done():
            print(f'[ILC+CBF] Done at step {i}')
            break
        if info['arrive']:
            print(f'[ILC+CBF] Goal reached at step {i}!')
            break
        if info['replan']:
            print(f'[ILC+CBF] Replanning at step {i}...')
            robot_start = env.robot.state.flatten()[:3]
            trajectory = planner.planning(
                robot_start, goal_xy, show_animation=False)
            if trajectory is not None:
                new_path = convert_rrt_path(trajectory)
                if new_path is not None:
                    ref_path_list = new_path
                    controller.update_ref_path(ref_path_list)
                    print(f'  New path: {len(ref_path_list)} waypoints')

    # ---- RRT* reference path: NOT displayed ----

    # ---- Overlay speed-colored trail (blue=slow, red=fast) ----
    if len(positions_xy) > 1:
        _ax = plt.gca()
        px_trail = [p[0] for p in positions_xy]
        py_trail = [p[1] for p in positions_xy]
        speeds = np.array(speeds_recorded)
        norm = plt.Normalize(0, 6.0)
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
