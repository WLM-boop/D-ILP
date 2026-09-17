"""
ILC (Iterative Learning Control) Local Planner Controller.

Ported from planner_node.cpp (ROS move_base local planner plugin).
Implements Spatial ILC with FPUR update rule, BLF lateral control,
and interfaces with the CBF safety filter for obstacle avoidance.

Usage pattern (mirrors RDA-planner's MPC):
    controller = ILCController(ref_path, sample_time=env.step_time)
    for i in range(500):
        obs_list = env.get_obstacle_info_list()
        opt_vel, info = controller.control(env.robot.state, ref_speed=0.5,
                                           obstacle_list=obs_list)
        env.step(opt_vel)
        env.render()
"""

import numpy as np
from math import sqrt, sin, cos, atan2, pi, log, fabs
from .cbf_filter import StaticCBFFilter, DynamicCBFFilter


class ILCController:
    """
    Spatial Iterative Learning Control (ILC) local planner with CBF safety filter.

    Parameters
    ----------
    ref_path : list of (3,1) numpy arrays
        Reference path waypoints: [[x], [y], [theta]] for each waypoint.
    sample_time : float
        Control and simulation time step (must match env.step_time). Default 0.1.
    v_init : float
        Initial/fallback linear velocity. Default 0.5.
    v_max : float
        Maximum linear velocity. Default 1.2.
    k_yaw : float
        Proportional yaw error gain. Default 1.2.
    goal_tolerance : float
        Goal arrival threshold in meters. Default 0.3.
    ilc_iterations : int
        Number of ILC learning iterations per plan. Default 5.
    """

    def __init__(self, ref_path, sample_time=0.1, v_init=0.5, v_max=1.2,
                 k_yaw=1.2, goal_tolerance=0.3, ilc_iterations=5):
        if ref_path is None or len(ref_path) < 2:
            raise ValueError("ref_path must contain at least 2 waypoints")

        self.ref_path = ref_path
        self.dt = sample_time
        self.v_init = v_init
        self.v_max = v_max
        self.k_yaw = k_yaw
        self.goal_tolerance = goal_tolerance
        self.ilc_iterations = ilc_iterations

        # ---- Learned profiles (populated by _initial_learn) ----
        self.vel_list = []           # Per-waypoint learned velocity
        self.omega_bias_list = []    # Per-waypoint learned omega bias

        # ---- Runtime state ----
        self.last_closest_index = 0
        self.last_v_cmd = 0.0
        self.last_omega_cmd = 0.0
        self.stuck_counter = 0
        self.goal_reached = False

        # ---- CBF safety filters ----
        self.static_cbf = StaticCBFFilter(emergency_rho=1.5, steer_zone=7.0, brake_zone=1.8, steer_gain=2.0)
        self.dynamic_cbf = DynamicCBFFilter(robot_radius=1.0, turn_gain=0.6, margin_base=0.5)

        # Run initial ILC learning
        self._initial_learn()

    # =====================================================================
    #  Public API (mirrors RDA MPC interface)
    # =====================================================================

    def control(self, state, ref_speed=0.5, obstacle_list=None,
                lidar_scan=None):
        """
        Main control interface.

        Parameters
        ----------
        state : (3,) or (3,1) ndarray
            Robot pose [x, y, theta].
        ref_speed : float
            Reference speed (used as fallback; ILC profile takes precedence).
        obstacle_list : list or None
            ObstacleInfo namedtuples for dynamic CBF.
        lidar_scan : dict or None
            LiDAR scan dict for static CBF (raw ranges).

        Returns
        -------
        opt_vel : (2,1) ndarray
            [linear_velocity, angular_velocity].
        info : dict
            Keys: 'arrive', 'stuck', 'cur_index', 'vel_list', 'omega_bias_list'.
        """
        if obstacle_list is None:
            obstacle_list = []

        # Ensure state is flat (3,)
        state = np.asarray(state).flatten()
        x, y, yaw = state[0], state[1], state[2]

        # ---- 1. Find closest waypoint index ----
        cur_index = self._closest_index(x, y)
        self.last_closest_index = cur_index

        # ---- 2. Compute desired yaw and yaw error ----
        yaw_des, diff_yaw = self._compute_yaw_error(x, y, yaw, cur_index)

        # ---- 3. Determine ILC velocity ----
        di = self._distance_to_end(x, y)
        is_near_end = (di < 0.4 or cur_index >= len(self.vel_list) - 5)
        is_at_end = (di < self.goal_tolerance * 1.5 or
                     cur_index >= len(self.vel_list) - 2)

        abs_diff_yaw = fabs(diff_yaw)
        v_ilc = self._get_ilc_velocity(cur_index, abs_diff_yaw, is_near_end)

        # ---- 4. Compute ILC omega ----
        omega_ilc = self._get_ilc_omega(cur_index, diff_yaw,
                                        is_near_end, is_at_end)

        # ---- 4b. Pure-pursuit style path tracking ----
        lookahead = max(5, int(2.0 / self.dt / 20))
        target_idx = min(cur_index + lookahead, len(self.ref_path) - 1)
        target_x = float(self.ref_path[target_idx][0, 0])
        target_y = float(self.ref_path[target_idx][1, 0])

        lookahead_yaw = atan2(target_y - y, target_x - x)
        yaw_err = lookahead_yaw - yaw
        while yaw_err > pi: yaw_err -= 2*pi
        while yaw_err < -pi: yaw_err += 2*pi

        pp_weight = min(1.0, fabs(yaw_err) / 0.5)
        omega_ilc = (1.0 - pp_weight) * omega_ilc + pp_weight * (self.k_yaw * yaw_err)

        # ---- 4c. Goal-attraction: pure goal tracking when close ----
        goal_x = float(self.ref_path[-1][0, 0])
        goal_y = float(self.ref_path[-1][1, 0])
        if di < 3.0:
            goal_yaw = atan2(goal_y - y, goal_x - x)
            goal_diff = goal_yaw - yaw
            while goal_diff > pi: goal_diff -= 2*pi
            while goal_diff < -pi: goal_diff += 2*pi
            omega_ilc = self.k_yaw * goal_diff
            v_ilc = min(v_ilc, max(0.15, di * 0.5))
        elif di < 8.0:
            goal_yaw = atan2(goal_y - y, goal_x - x)
            goal_diff = goal_yaw - yaw
            while goal_diff > pi: goal_diff -= 2*pi
            while goal_diff < -pi: goal_diff += 2*pi
            blend = 0.5 * (1.0 - (di - 3.0) / 5.0)
            omega_ilc = (1.0 - blend) * omega_ilc + blend * self.k_yaw * goal_diff

        if di < 0.5:
            v_ilc = 0.0
            omega_ilc = 0.0

        # ---- 5. Hard emergency stop ----
        # When a CBF filter is active, it handles emergency stop internally
        # (single LiDAR pass).  Only run the standalone emergency scan when
        # NO CBF is present, avoiding redundant full-array np.isfinite + np.min.
        hard_emergency = False
        if self.static_cbf is None and lidar_scan is not None:
            ranges = lidar_scan.get('ranges', [])
            if len(ranges) > 0:
                min_d = float(np.min(ranges))
                if np.isfinite(min_d) and min_d < 0.5:
                    v_ilc = 0.0
                    omega_ilc = 0.0
                    self.last_v_cmd = 0.0
                    self.last_omega_cmd = 0.0
                    self.stuck_counter = 999
                    hard_emergency = True

        # ---- 5b. Static CBF safety filter ----
        if not hard_emergency and self.static_cbf is not None:
            v_ilc, omega_ilc = self.static_cbf.apply(
                state, v_ilc, omega_ilc, lidar_scan, self.v_max
            )

        # ---- 6. Dynamic CBF safety filter ----
        if not hard_emergency and self.dynamic_cbf is not None:
            v_ilc, omega_ilc = self.dynamic_cbf.apply(
                state, v_ilc, omega_ilc, obstacle_list
            )

        # ---- 6b. BLF Safety: enforced in _spatial_ilc() during ILC learning ----
        # Runtime BLF removed: original code had undefined ct_error bug making it
        # unreachable. Any fix triggers it at ka=0.45 causing spin at start.

        # ---- 6c. Clamp omega + deadband to prevent oscillation ----
        omega_ilc = max(-1.0, min(1.0, omega_ilc))
        if fabs(omega_ilc) < 0.08:
            omega_ilc = 0.0

        # ---- 7. Low-pass filter ----
        v_cmd, omega_cmd = self._low_pass_filter(v_ilc, omega_ilc)

        # ---- 8. Stuck detection -> request replan ----
        found_obstacle = self.static_cbf._last_found if self.static_cbf is not None else False
        stuck = self._check_stuck(v_cmd, found_obstacle)

        # ---- 9. Goal proximity / goal check ----
        arrived = False
        if di < 0.5:
            omega_cmd *= 0.15

        if di < self.goal_tolerance:
            v_cmd = 0.0
            omega_cmd = 0.0
            self.last_v_cmd = 0.0
            self.last_omega_cmd = 0.0
            if not self.goal_reached:
                self.goal_reached = True
            arrived = True

        opt_vel = np.array([[v_cmd], [omega_cmd]])
        info = {
            'arrive': arrived,
            'stuck': stuck,
            'replan': stuck,
            'cur_index': cur_index,
            'vel_list': self.vel_list,
            'omega_bias_list': self.omega_bias_list,
        }
        return opt_vel, info

    def update_ref_path(self, ref_path):
        """Update reference path and re-run ILC learning."""
        if ref_path is None or len(ref_path) < 2:
            raise ValueError("ref_path must contain at least 2 waypoints")
        self.ref_path = ref_path
        self.last_closest_index = 0
        self.goal_reached = False
        self._initial_learn()

    def reset(self):
        """Reset controller runtime state."""
        self.last_closest_index = 0
        self.last_v_cmd = 0.0
        self.last_omega_cmd = 0.0
        self.stuck_counter = 0
        self.goal_reached = False

    # =====================================================================
    #  ILC Learning (spatialILC + iterate from planner_node.cpp)
    # =====================================================================

    def _initial_learn(self):
        """Equivalent of setPlan() + spatialILC(): run ILC iterations."""
        n = len(self.ref_path)

        # Initialize velocity profile with v_init
        self.vel_list = [self.v_init] * n
        self.omega_bias_list = [0.0] * n

        self._spatial_ilc()

    def _spatial_ilc(self):
        """
        Core spatial ILC simulation loop.

        Ported from LocalPlanner::spatialILC() (planner_node.cpp lines 732-942).
        Runs `ilc_iterations` passes of kinematic simulation along the path,
        accumulating signed cross-track error per waypoint, then applying
        the FPUR update rule.
        """
        path_length = len(self.ref_path)

        # Extract path x, y coordinates
        pathx = [float(self.ref_path[i][0, 0]) for i in range(path_length)]
        pathy = [float(self.ref_path[i][1, 0]) for i in range(path_length)]

        # Compute start path yaw
        start_path_yaw = 0.0
        if path_length >= 2:
            start_path_yaw = atan2(pathy[1] - pathy[0], pathx[1] - pathx[0])

        # Generate repetitive disturbance pattern for robust learning
        disturb = np.zeros(path_length)
        for i in range(path_length):
            t = i / path_length
            disturb[i] = 0.08 * (np.sin(t * 3 * np.pi) + 0.4 * np.sin(t * 5 * np.pi))

        # Initial velocity magnitude
        v_curr_mag = max(self.v_init, 0.4)

        # BLF parameters for the internal learning rollout
        ka_sim = 2.0
        k_blf_sim = 3.0

        for iteration in range(self.ilc_iterations):
            # Reset simulation state
            px = pathx[0]
            py = pathy[0]
            vx = v_curr_mag * cos(start_path_yaw)
            vy = v_curr_mag * sin(start_path_yaw)
            current = 0
            error_list = [0.0] * path_length
            current_all = path_length - 1

            k = 0
            while current < current_all - 1 and k < path_length * 3:
                prev_idx = current

                # Find closest path point + signed error
                current, signed_error = self._getpoint_sim(
                    px, py, pathx, pathy, current
                )

                abs_err = fabs(signed_error)

                # Record error at current index
                error_list[current] = signed_error
                if current > prev_idx:
                    for m in range(prev_idx, current):
                        error_list[m] = signed_error

                # BLF lateral feedback control
                safe_error = min(abs_err, ka_sim - 0.02)
                denom = ka_sim * ka_sim - safe_error * safe_error
                if denom < 0.005:
                    denom = 0.005

                blf_feedback_mag = k_blf_sim * (safe_error / denom)
                if blf_feedback_mag > 3.0:
                    blf_feedback_mag = 3.0

                # Get direction at current index
                md_x, md_y = self._path_direction(pathx, pathy, current)
                mt_x = pathx[current] - px
                mt_y = pathy[current] - py

                # Parallel component: follow path at learned speed
                v_learned = self.vel_list[current]
                v_para_x = md_x * v_learned
                v_para_y = md_y * v_learned

                # Perpendicular component: BLF feedback toward path
                scale = blf_feedback_mag / (abs_err + 1e-6)
                v_perp_x = mt_x * scale
                v_perp_y = mt_y * scale

                # Desired velocity (with disturbance injection for robust learning)
                v_des_x = v_para_x + v_perp_x
                v_des_y = v_para_y + v_perp_y

                # Inject repetitive disturbance
                d = disturb[current]
                perp_x = -md_y
                perp_y = md_x
                v_des_x += d * perp_x * v_learned * 2.0
                v_des_y += d * perp_y * v_learned * 2.0

                # Saturate
                v_des_mag = sqrt(v_des_x * v_des_x + v_des_y * v_des_y)
                if v_des_mag > self.v_max:
                    v_des_x = v_des_x / v_des_mag * self.v_max
                    v_des_y = v_des_y / v_des_mag * self.v_max

                # Kinematic integration
                ax = 5.0 * (v_des_x - vx)
                ay = 5.0 * (v_des_y - vy)
                vx += self.dt * ax
                vy += self.dt * ay
                px += self.dt * vx
                py += self.dt * vy

                k += 1

            # Fill remaining error
            for m in range(current, path_length):
                error_list[m] = error_list[max(0, current - 1)]

            # Apply FPUR update
            self._iterate(error_list)

        # Log summary
        avg_v = sum(self.vel_list) / len(self.vel_list)
        avg_err = sum(abs(e) for e in error_list) / len(error_list)
        print(f"[ILC] Learning complete: Avg_V={avg_v:.2f}, Avg_Err={avg_err:.3f}")

    def _getpoint_sim(self, px, py, pathx, pathy, current):
        """
        Find closest path point and compute signed cross-track error.
        """
        path_length = len(pathx)
        min_dist_sq = 1e10
        best_idx = current

        start_search = max(0, current - 5)

        for i in range(start_search, path_length - 1):
            dx = pathx[i] - px
            dy = pathy[i] - py
            dist_sq = dx * dx + dy * dy

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = i
            elif dist_sq > min_dist_sq + 0.8:
                break

        next_idx = min(best_idx + 1, path_length - 1)
        md_x = pathx[next_idx] - pathx[best_idx]
        md_y = pathy[next_idx] - pathy[best_idx]
        md_norm = sqrt(md_x * md_x + md_y * md_y)
        if md_norm > 1e-6:
            md_x /= md_norm
            md_y /= md_norm
        elif best_idx > 0:
            md_x = pathx[best_idx] - pathx[best_idx - 1]
            md_y = pathy[best_idx] - pathy[best_idx - 1]
            md_norm = sqrt(md_x * md_x + md_y * md_y)
            if md_norm > 1e-6:
                md_x /= md_norm
                md_y /= md_norm
            else:
                md_x, md_y = 1.0, 0.0
        else:
            md_x, md_y = 1.0, 0.0

        mt_x = pathx[best_idx] - px
        mt_y = pathy[best_idx] - py

        cross = md_x * mt_y - md_y * mt_x
        abs_err = sqrt(min_dist_sq)
        signed_error = abs_err if cross >= 0 else -abs_err

        return best_idx, signed_error

    def _path_direction(self, pathx, pathy, idx):
        """Compute unit tangent direction at path index."""
        path_length = len(pathx)
        next_idx = min(idx + 1, path_length - 1)
        dx = pathx[next_idx] - pathx[idx]
        dy = pathy[next_idx] - pathy[idx]
        norm = sqrt(dx * dx + dy * dy)
        if norm > 1e-6:
            return dx / norm, dy / norm
        if idx > 0:
            dx = pathx[idx] - pathx[idx - 1]
            dy = pathy[idx] - pathy[idx - 1]
            norm = sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                return dx / norm, dy / norm
        return 1.0, 0.0

    def _iterate(self, error_list):
        """
        FPUR (Fractional Power Update Rule) learning update.
        """
        total_size = len(self.vel_list)
        if total_size < 2:
            return

        if len(self.omega_bias_list) != total_size:
            self.omega_bias_list = [0.0] * total_size

        # ---- Velocity parameters ----
        alpha_v = 0.05
        beta_v = 0.10
        gamma_v = 0.6
        e_ref_v = 0.4

        # ---- Omega parameters ----
        alpha_w = 0.15
        beta_w = 0.1
        gamma_w = 0.5

        slow_down_index = int(total_size * 0.92)

        for i in range(total_size):
            signed_err = error_list[i]

            if i > 0 and fabs(signed_err) < 1e-5:
                signed_err = error_list[i - 1]
                error_list[i] = signed_err

            if signed_err > 0.9:
                signed_err = 0.9
            if signed_err < -0.9:
                signed_err = -0.9

            abs_err = fabs(signed_err)
            err_sign = 1.0 if signed_err >= 0 else -1.0

            if abs_err > 1.9:
                continue

            # ---- Longitudinal velocity FPUR ----
            ek_v = abs_err - e_ref_v
            sign_ev = 1.0 if ek_v > 0 else -1.0

            v_correction = (alpha_v * ek_v +
                            beta_v * (abs(ek_v) ** gamma_v) * sign_ev)

            if v_correction > 0.10:
                v_correction = 0.10
            if v_correction < -0.08:
                v_correction = -0.08

            v_new = self.vel_list[i] - v_correction + 0.02

            # ---- Coupling constraint ----
            coupling_v_limit = self.v_max
            if abs_err > 0.4:
                coupling_v_limit = self.v_max * 0.85

            # ---- End-of-path deceleration window ----
            current_v_max = coupling_v_limit
            if i > slow_down_index:
                ratio = (i - slow_down_index) / (total_size - slow_down_index)
                current_v_max = coupling_v_limit * (1.0 - 0.6 * ratio)
                if current_v_max < 0.5:
                    current_v_max = 0.5

            self.vel_list[i] = max(0.8, min(current_v_max, v_new))

            # ---- Lateral omega bias FPUR ----
            if abs_err > 0.02:
                w_correction = (alpha_w * abs_err +
                                beta_w * (abs_err ** gamma_w)) * err_sign
                forget_factor = 0.95
                learning_rate = 0.15
                self.omega_bias_list[i] = (forget_factor *
                                           self.omega_bias_list[i] +
                                           learning_rate * w_correction)
            else:
                self.omega_bias_list[i] *= 0.98

            self.omega_bias_list[i] = max(-0.6, min(0.6,
                                                    self.omega_bias_list[i]))

        if len(self.vel_list) > 1:
            self.vel_list[-1] = 0.5

    # =====================================================================
    #  Runtime helpers (control loop)
    # =====================================================================

    def _closest_index(self, x, y, search_window=50):
        """Find index of nearest waypoint, with local window search."""
        plan_size = len(self.ref_path)
        if plan_size == 0:
            return 0

        start_idx = max(0, self.last_closest_index - 10)
        end_idx = min(plan_size - 1, self.last_closest_index + search_window)

        min_dist_sq = 1e10
        best_idx = start_idx

        for i in range(start_idx, end_idx + 1):
            dx = x - float(self.ref_path[i][0, 0])
            dy = y - float(self.ref_path[i][1, 0])
            d2 = dx * dx + dy * dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_idx = i

        return best_idx

    def _compute_yaw_error(self, x, y, yaw, cur_index):
        """Compute desired yaw (path tangent) and yaw error."""
        plan_size = len(self.ref_path)

        if cur_index >= plan_size - 1:
            dx = float(self.ref_path[-1][0, 0]) - float(
                self.ref_path[-2][0, 0])
            dy = float(self.ref_path[-1][1, 0]) - float(
                self.ref_path[-2][1, 0])
        else:
            dx = float(self.ref_path[cur_index + 1][0, 0]) - float(
                self.ref_path[cur_index][0, 0])
            dy = float(self.ref_path[cur_index + 1][1, 0]) - float(
                self.ref_path[cur_index][1, 0])

        yaw_des = atan2(dy, dx)
        diff_yaw = yaw_des - yaw

        while diff_yaw > pi:
            diff_yaw -= 2.0 * pi
        while diff_yaw < -pi:
            diff_yaw += 2.0 * pi

        return yaw_des, diff_yaw

    def _get_ilc_velocity(self, cur_index, abs_diff_yaw, is_near_end):
        """
        Determine velocity from learned profile, with heading-dependent
        speed modulation and end-of-path override.
        """
        idx = min(cur_index, len(self.vel_list) - 1)

        if abs_diff_yaw > 2.09:
            v_ilc = 3.0
        elif abs_diff_yaw > 0.785:
            v_ilc = 2.5
        else:
            v_ilc = self.vel_list[idx]

        return v_ilc

    def _get_ilc_omega(self, cur_index, diff_yaw, is_near_end, is_at_end):
        """Compute omega from P-controller + learned bias."""
        idx = min(cur_index, len(self.omega_bias_list) - 1)
        learned_bias = self.omega_bias_list[idx]

        if is_near_end:
            learned_bias *= 0.2
            if fabs(diff_yaw) > 1.57:
                diff_yaw = 0.0

        if is_at_end:
            learned_bias = 0.0

        current_k_yaw = self.k_yaw * 0.5 if is_at_end else self.k_yaw
        omega_ilc = current_k_yaw * diff_yaw + learned_bias

        return omega_ilc

    def _distance_to_end(self, x, y):
        """Euclidean distance from (x,y) to last waypoint."""
        if not self.ref_path:
            return 0.0
        dx = x - float(self.ref_path[-1][0, 0])
        dy = y - float(self.ref_path[-1][1, 0])
        return sqrt(dx * dx + dy * dy)

    def _cross_track_error(self, x, y, cur_index):
        """Signed cross-track error from robot to reference path segment."""
        n = len(self.ref_path)
        if n < 2 or cur_index >= n - 1:
            return 0.0

        p0 = self.ref_path[cur_index]
        p1 = self.ref_path[cur_index + 1]
        dx = float(p1[0, 0] - p0[0, 0])
        dy = float(p1[1, 0] - p0[1, 0])
        seg_len = sqrt(dx * dx + dy * dy)
        if seg_len < 1e-6:
            return 0.0

        nx = -dy / seg_len
        ny = dx / seg_len

        rx = x - float(p0[0, 0])
        ry = y - float(p0[1, 0])

        return rx * nx + ry * ny

    def _low_pass_filter(self, v_target, omega_target):
        """First-order low-pass filter for command smoothing."""
        alpha_v = 0.85
        alpha_omega = 0.75

        v_target = max(0.0, min(v_target, self.v_max))
        omega_target = max(-1.5, min(1.5, omega_target))

        v_cmd = alpha_v * v_target + (1.0 - alpha_v) * self.last_v_cmd
        omega_cmd = (alpha_omega * omega_target +
                     (1.0 - alpha_omega) * self.last_omega_cmd)

        self.last_v_cmd = v_cmd
        self.last_omega_cmd = omega_cmd

        return v_cmd, omega_cmd

    def _check_stuck(self, v_cmd, found_obstacle):
        """
        Stuck detection: if speed < 0.05 for 20+ cycles with obstacles
        present, trigger re-plan.
        """
        if not self.goal_reached and v_cmd < 0.05 and found_obstacle:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.stuck_counter = 0
                self.last_v_cmd = 0.0
                self.last_omega_cmd = 0.0
                return True
        else:
            self.stuck_counter = 0
        return False
