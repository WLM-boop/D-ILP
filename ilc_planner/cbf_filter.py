"""Obstacle filters used by the iterative-learning controller.

Includes heuristic steering and analytical velocity bounds.
The controller smooths commands after filtering; this implementation does
not enforce a strict final-command projection onto the velocity bounds.
"""

import numpy as np
from math import sqrt, sin, cos, atan2, pi, fabs, exp, log


class StaticCBFFilter:
    """
    CBF safety filter with active steering for static obstacle avoidance.

    Uses LiDAR to find nearest obstacle, then applies:
    - Steering: turn away from obstacle (with hysteresis to prevent oscillation)
    - Speed reduction: proportional to proximity
    - Emergency stop: when obstacle is critically close

    Parameters
    ----------
    emergency_rho : float
        Hard stop distance. Default 1.5m.
    steer_zone : float
        Start steering below this distance. Default 5.0m.
    detection_angle : float
        Half-angle LiDAR FOV. Default 1.2 rad.
    steer_gain : float
        Avoidance turn magnitude. Default 0.8.
    """

    def __init__(self, emergency_rho=1.5, steer_zone=5.0, brake_zone=2.5,
                 detection_angle=1.2, steer_gain=0.8):
        self.emergency_rho = emergency_rho
        self.steer_zone = steer_zone
        self.brake_zone = brake_zone
        self.detection_angle = detection_angle
        self.steer_gain = steer_gain
        self._last_found = False
        self._last_steer_dir = 0.0
        self._steer_persistence = 0

    def apply(self, state, v_ilc, omega_ilc, lidar_scan, v_max):
        """
        CBF: steer-first, brake-later (matching original C++ priority).

        - Steer zone (brake_zone ~ steer_zone): active turning, NO braking
        - Brake zone (emergency_rho ~ brake_zone): turning + gentle braking
        - Emergency (rho < emergency_rho): hard stop
        """
        if lidar_scan is None:
            self._last_found = False
            return v_ilc, omega_ilc

        rho, theta = self._find_nearest_obstacle(lidar_scan, v_ilc)
        self._last_found = (rho is not None)

        if rho is None:
            return v_ilc, omega_ilc

        # ---- Emergency stop ----
        if rho < self.emergency_rho:
            return 0.0, omega_ilc

        # ---- Steer zone: turn away, NO speed reduction ----
        if rho < self.steer_zone:
            steer_weight = 1.0 - (rho - self.brake_zone) / (self.steer_zone - self.brake_zone)
            steer_weight = max(0.0, min(1.0, steer_weight))

            # Hysteresis on steer direction
            if self._steer_persistence <= 0:
                self._last_steer_dir = -1.0 if theta > 0 else 1.0
                self._steer_persistence = 6
            else:
                self._steer_persistence -= 1

            omega_ilc += self._last_steer_dir * self.steer_gain * steer_weight

        # ---- Brake zone: gentle speed reduction (only when very close) ----
        if rho < self.brake_zone:
            brake_weight = (1.0 - (rho - self.emergency_rho) / (self.brake_zone - self.emergency_rho)) ** 2
            v_safe = max(0.5, v_ilc * (1.0 - 0.5 * brake_weight))
            v_ilc = min(v_ilc, v_safe)

        return v_ilc, omega_ilc

    def _find_nearest_obstacle(self, lidar_scan, v_ilc):
        """Scan LiDAR ranges for nearest obstacle (original C++ pattern)."""
        ranges = lidar_scan['ranges']
        angle_min = lidar_scan['angle_min']
        angle_increment = lidar_scan['angle_increment']

        look_dist = 0.7 + fabs(v_ilc) * 2.0
        min_dist = look_dist
        best_rho, best_theta = None, None

        for i in range(0, len(ranges), 8):
            angle = angle_min + i * angle_increment
            if fabs(angle) < self.detection_angle:
                r = ranges[i]
                if np.isfinite(r) and r > 0.15 and r < min_dist:
                    min_dist = r
                    best_rho = r
                    best_theta = angle

        return best_rho, best_theta


class StaticCBFQPFilter:
    """
    First-order CBF-inspired filter — velocity-constraint only (no steer-first).

    Unlike the original StaticCBFFilter (which steers away from obstacles first
    and brakes later, causing weaving), this filter uses the same formulation
    as a first-order velocity constraint:

        For each LiDAR beam at angle φ, range d:
            CBF:  v · cos(φ) ≤ α · (d - R_safe)   (cos(φ) > 0)

    The QP decouples → analytical solution:
        v* = min(v_des, min_i{α·(d_i - R_safe) / cos(φ_i)})

    When v is heavily reduced, a mild steering bias turns the robot away from
    the nearest obstacle — but only as needed, not preemptively.

    Parameters
    ----------
    R_safe : float
        CBF safety radius (m).  Default 2.0.
    alpha_cbf : float
        CBF decay rate.  Larger = less conservative.  Default 2.5.
    k_steer : float
        Obstacle-avoidance steering gain.  Default 2.0.
    fov_half : float
        Half-angle FOV for CBF consideration (rad).  Default 1.05 (~60°).
    beam_step : int
        LiDAR subsampling step.  Default 4.
    emergency_rho : float
        Hard stop distance (m).  Default 0.5.
    """

    def __init__(self, R_safe=2.0, alpha_cbf=2.5, k_steer=2.0,
                 fov_half=1.05, beam_step=4, emergency_rho=0.5):
        self.R_safe = R_safe
        self.alpha_cbf = alpha_cbf
        self.k_steer = k_steer
        self.fov_half = fov_half
        self.beam_step = beam_step
        self.emergency_rho = emergency_rho
        self._last_found = False

    def apply(self, state, v_ilc, omega_ilc, lidar_scan, v_max):
        """
        Apply first-order CBF velocity constraint + mild steering bias.

        Returns (v_safe, omega_safe).

        NOTE: The ILC controller already handles the hard emergency stop
        (< 0.5 m) before calling this filter, so we do NOT repeat the
        full-array np.isfinite + np.min scan here.  Emergency detection
        is folded into the single beam loop below.
        """
        if lidar_scan is None:
            self._last_found = False
            return v_ilc, omega_ilc

        ranges = lidar_scan.get('ranges', [])
        if len(ranges) == 0:
            self._last_found = False
            return v_ilc, omega_ilc

        angle_min = lidar_scan['angle_min']
        angle_increment = lidar_scan['angle_increment']
        range_max = lidar_scan['range_max']

        # ---- Single-pass: CBF constraints + emergency detection ----
        v_bound = v_ilc
        nearest_phi = None
        nearest_d = range_max
        min_d_all = range_max  # track global min for emergency stop

        for i in range(0, len(ranges), self.beam_step):
            d = ranges[i]
            if not np.isfinite(d) or d >= range_max - 0.01:
                continue

            # Emergency stop: track minimum range across all beams
            if d < min_d_all:
                min_d_all = d

            phi = angle_min + i * angle_increment
            if fabs(phi) > self.fov_half:
                continue

            cos_phi = cos(phi)
            if cos_phi <= 0.05:
                continue

            # Track nearest obstacle for steering bias
            if d < nearest_d:
                nearest_d = d
                nearest_phi = phi

            # CBF: v ≤ α·(d - R_safe) / cos(φ)
            margin = d - self.R_safe
            if margin <= 0.0:
                v_upper = 0.0
            else:
                v_upper = self.alpha_cbf * margin / cos_phi

            if v_upper < v_bound:
                v_bound = v_upper

        # ---- Hard emergency stop (from single-pass min_d) ----
        if min_d_all < self.emergency_rho:
            self._last_found = True
            return 0.0, omega_ilc

        self._last_found = (nearest_phi is not None)

        if not self._last_found:
            return v_ilc, omega_ilc

        # Clamp v
        v_safe = max(0.3, min(v_bound, v_max))

        # ---- Steering bias: turn away from nearest obstacle when speed is cut ----
        omega_safe = omega_ilc
        if v_safe < v_ilc * 0.6 and v_ilc > 0.1 and nearest_phi is not None:
            steer_dir = -1.0 if nearest_phi > 0 else 1.0
            reduction_ratio = 1.0 - v_safe / max(v_ilc, 0.1)
            omega_bias = steer_dir * self.k_steer * reduction_ratio
            omega_safe = omega_ilc + omega_bias

        # Clamp omega
        omega_safe = max(-1.5, min(1.5, omega_safe))
        if fabs(omega_safe) < 0.08:
            omega_safe = 0.0

        return v_safe, omega_safe


class NullspaceCBFFilter:
    """
    Nullspace CBF + Steer Zone — O(m) with active avoidance.

    Two-layer safety:
      1. **v-constraint** (nullspace): CBF limits v only, ω preserved — O(m)
      2. **steer zone** (7m): same as StaticCBFFilter's steer-first heuristic
         — only ω modified, v untouched.  Hysteresis prevents oscillation.

    This gives the best of both: analytical velocity bounds + heuristic active steering,
    without the QP overhead.

    Parameters
    ----------
    R_safe : float
        Safety radius for CBF v-constraint (m).  Default 2.0.
    alpha_cbf : float
        CBF decay rate.  Default 2.5.
    fov_half : float
        Half-angle LiDAR FOV (rad).  Default 1.05.
    steer_zone : float
        Distance below which active steering engages (m).  Default 7.0.
    steer_gain : float
        Avoidance turn magnitude.  Default 2.0.
    T_pred : float
        Prediction horizon for dynamic obstacles (s).  0 = disabled.
    """

    def __init__(self, R_safe=2.0, alpha_cbf=2.5, fov_half=1.05,
                 steer_zone=7.0, steer_gain=2.0, T_pred=1.0):
        self.R_safe = R_safe
        self.alpha_cbf = alpha_cbf
        self.fov_half = fov_half
        self.steer_zone = steer_zone
        self.steer_gain = steer_gain
        self.T_pred = T_pred
        self._last_found = False
        self._last_steer_dir = 0.0
        self._steer_persistence = 0

    def apply(self, state, v_ilc, omega_ilc, lidar_scan=None, v_max=8.0,
              obstacle_list=None):
        """
        Apply nullspace CBF-inspired filter — O(m) single-pass.

        Dual-interface (auto-detects calling convention):
          - static_cbf  slot → apply(state, v, ω, lidar, v_max)
          - dynamic_cbf slot → apply(state, v, ω, obstacle_list)

        Parameters
        ----------
        state : (3,) ndarray
            Robot pose [x, y, yaw].
        v_ilc : float
            ILC desired linear velocity.
        omega_ilc : float
            ILC desired angular velocity (preserved in nullspace).
        lidar_scan : dict, list, or None
            LiDAR scan dict (static_cbf slot) or obstacle_list (dynamic_cbf slot).
        v_max : float
            Maximum velocity.
        obstacle_list : list or None
            Obstacle info (only used when called via static_cbf slot with
            explicit keyword, or when used as dual-filter).

        Returns
        -------
        v_cmd : float
            Safe linear velocity.
        omega_cmd : float
            Safe angular velocity (≈ omega_ilc, unless escape mode).
        """
        x, y, yaw = state[0], state[1], state[2]

        # ---- Auto-detect calling convention ----
        # ILCController calls:
        #   static_cbf.apply(s, v, ω, lidar_scan, v_max)     → lidar_scan is dict
        #   dynamic_cbf.apply(s, v, ω, obstacle_list)         → lidar_scan is list
        _lidar = lidar_scan
        _obs_list = obstacle_list

        if isinstance(lidar_scan, list):
            # Called via dynamic_cbf slot — lidar_scan is actually obstacle_list
            _lidar = None
            _obs_list = lidar_scan

        # ---- Step 1: Collect v-bounds from all constraints ----
        v_bound = v_max
        has_obstacle = False

        # 1a. LiDAR constraints
        if _lidar is not None and isinstance(_lidar, dict):
            v_ld = self._lidar_v_bound(_lidar)
            if v_ld is not None and v_ld < v_max:
                v_bound = min(v_bound, v_ld)
                has_obstacle = True

        # 1b. Dynamic obstacle predicted constraints
        if _obs_list is not None and self.T_pred > 0:
            v_dyn = self._dynamic_v_bound(x, y, yaw, _obs_list)
            if v_dyn is not None and v_dyn < v_max:
                v_bound = min(v_bound, v_dyn)
                has_obstacle = True

        self._last_found = has_obstacle

        # ---- Step 2: Nullspace — apply only v to constraint ----
        v_safe = max(0.0, min(v_ilc, v_bound))
        omega_cmd = omega_ilc

        # ---- Step 3: Steer zone — active avoidance (same as StaticCBFFilter) ----
        # Only ω is modified; v is governed by the CBF nullspace constraint.
        min_lidar, phi_nearest = self._find_nearest_obstacle(_lidar)

        if min_lidar is not None and min_lidar < self.steer_zone:
            # Steer weight: linear from 0 at steer_zone edge to 1 at R_safe
            steer_range = self.steer_zone - self.R_safe
            if steer_range > 0.01:
                steer_weight = 1.0 - (min_lidar - self.R_safe) / steer_range
                steer_weight = max(0.0, min(1.0, steer_weight))
            else:
                steer_weight = 1.0

            # Hysteresis: lock steer direction to prevent oscillation
            if self._steer_persistence <= 0:
                self._last_steer_dir = -1.0 if phi_nearest > 0 else 1.0
                self._steer_persistence = 6  # hold 0.6s
            else:
                self._steer_persistence -= 1

            omega_cmd += self._last_steer_dir * self.steer_gain * steer_weight

        # ---- Step 4: Emergency stop ----
        if min_lidar is not None and min_lidar < 0.5:
            v_safe = 0.0
            omega_cmd = 0.0

        return v_safe, omega_cmd

    # ------------------------------------------------------------------
    #  Constraint builders
    # ------------------------------------------------------------------

    def _lidar_v_bound(self, lidar_scan):
        """
        Compute most restrictive v-bound from LiDAR.

        For each beam:  v ≤ α·(d − R_safe) / cos(φ)
        Returns the minimum safe v, or None if unconstrained.
        """
        ranges = lidar_scan.get('ranges', [])
        if len(ranges) == 0:
            return None

        angle_min = lidar_scan['angle_min']
        angle_inc = lidar_scan['angle_increment']
        range_max = lidar_scan['range_max']

        best = None
        for i in range(len(ranges)):
            d = ranges[i]
            if not np.isfinite(d) or d >= range_max - 0.01:
                continue

            phi = angle_min + i * angle_inc
            if fabs(phi) > self.fov_half:
                continue

            cos_phi = cos(phi)
            if cos_phi <= 0.05:        # tangent beam — no v constraint
                continue

            margin = d - self.R_safe
            if margin <= 0.0:
                return 0.0             # inside safety zone → must stop

            # v ≤ α·(d − R_safe) / cos(φ)
            v_lim = self.alpha_cbf * margin / cos_phi
            if best is None or v_lim < best:
                best = v_lim

        return best

    def _dynamic_v_bound(self, x, y, yaw, obstacle_list):
        """
        Compute v-bound from predicted dynamic obstacle positions.

        Constant-velocity prediction, same CBF formula as LiDAR.
        """
        best = None

        for obs in obstacle_list:
            obs_vel = np.asarray(obs.velocity).flatten()
            vx = float(obs_vel[0]) if len(obs_vel) > 0 else 0.0
            vy = float(obs_vel[1]) if len(obs_vel) > 1 else 0.0
            speed = sqrt(vx * vx + vy * vy)
            if speed < 0.02:
                continue

            obs_center = np.asarray(obs.center).flatten()
            ox = float(obs_center[0]) + vx * self.T_pred
            oy = float(obs_center[1]) + vy * self.T_pred
            obs_radius = float(getattr(obs, 'radius', 0.5) or 0.5)

            dx = ox - x
            dy = oy - y
            dist = sqrt(dx * dx + dy * dy)
            d_eff = dist - obs_radius
            if d_eff > 15.0:
                continue

            world_angle = atan2(dy, dx)
            phi = world_angle - yaw
            while phi > pi:   phi -= 2 * pi
            while phi < -pi:  phi += 2 * pi

            if fabs(phi) > self.fov_half:
                continue
            cos_phi = cos(phi)
            if cos_phi <= 0.05:
                continue

            margin = d_eff - self.R_safe
            if margin <= 0.0:
                return 0.0

            v_lim = self.alpha_cbf * margin / cos_phi
            if best is None or v_lim < best:
                best = v_lim

        return best

    def _find_nearest_obstacle(self, lidar_scan):
        """
        Scan LiDAR for nearest obstacle within detection zone.

        Returns (min_range, phi) or (None, None) if no obstacle found.
        Matches StaticCBFFilter's scanning pattern.
        """
        if lidar_scan is None or not isinstance(lidar_scan, dict):
            return None, None

        ranges = lidar_scan.get('ranges', [])
        if len(ranges) == 0:
            return None, None

        angle_min = lidar_scan['angle_min']
        angle_inc = lidar_scan['angle_increment']
        range_max = lidar_scan['range_max']

        best_rho, best_phi = None, None

        for i in range(0, len(ranges), 8):  # subsample (same as StaticCBFFilter)
            phi = angle_min + i * angle_inc
            if fabs(phi) > self.fov_half:
                continue
            d = ranges[i]
            if np.isfinite(d) and d > 0.15:
                if best_rho is None or d < best_rho:
                    best_rho = d
                    best_phi = phi

        return best_rho, best_phi


class DynamicCBFFilter:
    """
    Smart dynamic obstacle avoidance based on obstacle motion direction.

    Key improvements over original:
    - Avoidance direction chosen based on obstacle's velocity, not just cross product
    - Only reacts to obstacles approaching us (dot product < 0)
    - Smaller safety margins for more realistic behavior
    - Per-obstacle speed limit + steering based on predicted collision geometry

    Parameters
    ----------
    robot_radius : float
        Robot collision radius. Default 1.5 (half-width for 4.6x1.6m car).
    turn_gain : float
        Avoidance turn strength. Default 0.8.
    margin_base : float
        Base safety margin. Default 1.0.
    """

    def __init__(self, robot_radius=1.5, turn_gain=0.8, margin_base=1.0):
        self.robot_radius = robot_radius
        self.turn_gain = turn_gain
        self.margin_base = margin_base
        self._last_steer_dir = 0.0
        self._steer_persistence = 0  # frames remaining to hold direction

    def apply(self, state, v_ilc, omega_ilc, obstacle_list):
        """
        Smart dynamic avoidance using obstacle velocity direction.

        For each dynamic obstacle:
        1. Compute relative velocity — is it approaching us?
        2. Predict collision geometry (CPA time and position)
        3. Choose avoidance direction PERPENDICULAR to obstacle's approach
        4. Apply speed limit if collision is imminent
        """
        x, y, yaw = state[0], state[1], state[2]
        robot_pos = np.array([x, y])
        robot_vel_vec = np.array([v_ilc * cos(yaw), v_ilc * sin(yaw)])

        # Filter: only obstacles with non-zero velocity
        dyn_obs_list = []
        for obs in obstacle_list:
            vel = np.asarray(obs.velocity).flatten()
            if np.linalg.norm(vel) > 0.02:
                dyn_obs_list.append(obs)

        if not dyn_obs_list:
            return v_ilc, omega_ilc

        total_omega_offset = 0.0
        total_weight = 0.0
        v_limit = v_ilc

        for obs in dyn_obs_list:
            obs_center = np.asarray(obs.center).flatten()
            obs_pos = obs_center[:2]
            obs_vel = np.asarray(obs.velocity).flatten()[:2]
            obs_radius = getattr(obs, 'radius', 0.5) or 0.5

            # Relative state
            p_rel = robot_pos - obs_pos      # vector from obstacle to robot
            v_rel = robot_vel_vec - obs_vel  # relative velocity
            rho = float(np.linalg.norm(p_rel))
            if rho < 0.01:
                rho = 0.01

            # React to all moving obstacles in range (no direction filter)

            # Safety radius
            r_safe = obs_radius + self.robot_radius + self.margin_base

            # Compute CPA
            tau = 0.0
            v_rel_sq = float(v_rel.dot(v_rel))
            if v_rel_sq > 0.01:
                t_cpa = -float(p_rel.dot(v_rel)) / v_rel_sq
                if t_cpa > 0.0:
                    tau = min(t_cpa, 2.0)

            # Predicted collision position
            p_pred = p_rel + tau * v_rel
            rho_pred = float(np.linalg.norm(p_pred))
            if rho_pred < 0.01:
                rho_pred = 0.01

            if rho_pred > r_safe + 2.0:
                continue
            threat = max(0.0, 1.0 - (rho_pred - r_safe) / 2.0)

            # --- Smart avoidance direction ---
            # The obstacle is approaching. We want to steer PERPENDICULAR
            # to its approach direction (i.e., move sideways out of its path).
            # The approach direction (from robot toward obstacle) is p_rel.
            # Perpendicular directions are +90 or -90 degrees from p_rel.

            # Direction from robot to predicted obstacle position
            dir_to_obs = atan2(p_pred[1], p_pred[0])

            # Choose left or right perpendicular based on which side
            # of the obstacle's velocity we are on.
            # If obstacle is passing to our LEFT, steer RIGHT (away from it).
            # If obstacle is passing to our RIGHT, steer LEFT.
            obs_vel_dir = atan2(obs_vel[1], obs_vel[0])
            dir_from_obs = atan2(-p_rel[1], -p_rel[0])  # from obstacle to robot

            # Angle from obstacle's velocity to our direction
            angle_diff = dir_from_obs - obs_vel_dir
            while angle_diff > pi:
                angle_diff -= 2 * pi
            while angle_diff < -pi:
                angle_diff += 2 * pi

            # Steer away with hysteresis — hold direction to prevent oscillation
            if self._steer_persistence <= 0:
                if angle_diff > 0:
                    self._last_steer_dir = 1.0
                else:
                    self._last_steer_dir = -1.0
                self._steer_persistence = 8  # hold for 0.8 seconds
            else:
                self._steer_persistence -= 1
            steer_dir = self._last_steer_dir

            omega_offset = steer_dir * self.turn_gain * threat

            # Speed limit: slow down when obstacle is close ahead
            angle_to_obs = dir_to_obs - yaw
            while angle_to_obs > pi: angle_to_obs -= 2*pi
            while angle_to_obs < -pi: angle_to_obs += 2*pi

            # Gentle speed reduction — only when obstacle is right ahead
            if fabs(angle_to_obs) < 0.5 and rho_pred < r_safe + 1.0:
                obs_v_limit = rho_pred * 0.8
                v_limit = min(v_limit, max(0.8, obs_v_limit))

            total_omega_offset += omega_offset * threat
            total_weight += threat

        if total_weight > 0.01:
            avg_omega_offset = total_omega_offset / total_weight
            omega_ilc += avg_omega_offset
            v_ilc = min(v_ilc, v_limit)

        return v_ilc, omega_ilc


# RDA-derived visualization helper; see LICENSES/RDA-planner-MIT.txt.
def scan_box(state, scan_data):
    """
    Convert LiDAR scan to polygon obstacles via DBSCAN clustering.
    Ported from RDA-planner lidar_path_track.py scan_box().
    """
    try:
        import cv2
        from sklearn.cluster import DBSCAN
    except ImportError:
        return []

    ranges = np.array(scan_data['ranges'])
    angle_min = scan_data['angle_min']
    angle_max = scan_data['angle_max']
    range_max = scan_data['range_max']

    angles = np.linspace(angle_min, angle_max, len(ranges))
    point_list = []

    for i in range(len(ranges)):
        r = ranges[i]
        if r < (range_max - 0.01) and np.isfinite(r):
            a = angles[i]
            point_list.append(np.array([[r * cos(a)], [r * sin(a)]]))

    if len(point_list) < 4:
        return []

    point_array = np.hstack(point_list).T
    labels = DBSCAN(eps=2.0, min_samples=6).fit_predict(point_array)

    state_flat = np.asarray(state).flatten()
    trans = state_flat[0:2].reshape(2, 1)
    rot = state_flat[2]
    R = np.array([[cos(rot), -sin(rot)], [sin(rot), cos(rot)]])

    obstacle_list = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_pts = point_array[labels == label]
        if len(cluster_pts) < 4:
            continue
        rect = cv2.minAreaRect(cluster_pts.astype(np.float32))
        box = cv2.boxPoints(rect)
        vertices = box.T
        global_vertices = trans + R @ vertices

        obstacle_list.append({
            'vertex': global_vertices,
            'center': None, 'radius': None,
            'cone_type': 'Rpositive', 'velocity': 0,
        })

    return obstacle_list
