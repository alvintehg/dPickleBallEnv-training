import numpy as np
import cv2
import os
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True, serve_training_mode=False, run_probe=False):
        # —— Unity env setup ——
        self.env = UnityParallelEnv(unity_env)

        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]       # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0] # opponent (left)
        self.agent_obs = self.env.possible_agents[0]   # camera index used for observations (left)

        # —— pixel-frame settings ——
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Final obs shape after manual preprocessing
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])  # (stack, H, W)
        else:
            obs_shape = (frame_stack * c, img_size[1], img_size[0])  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.env.action_spaces[self.agent]

        # —— ACTION SPACE DIAGNOSTICS ——
        print(f"\n{'='*60}")
        print(f"ACTION SPACE ANALYSIS:")
        print(f"{'='*60}")
        print(f"Action space type: {type(self.action_space)}")
        print(f"Action space: {self.action_space}")
        if hasattr(self.action_space, 'nvec'):
            print(f"Number of discrete branches: {len(self.action_space.nvec)}")
            for i, n in enumerate(self.action_space.nvec):
                print(f"  Branch {i}: {n} actions")
        print(f"{'='*60}\n")

        # —— reward-shaping parameters ——
        self.prev_game_state = None
        self.steps_since_serve = 0
        self.max_fast = 100
        self.hold_streak = 0
        self.break_streak = 0
        self.violation_occurred = False
        self.phi = {0: 0.0, 1: -0.1, 2: +0.1, 3: +0.5, 4: -0.5, 5: -1.2, 6: +1.2}

        # —— logging control ——
        self.total_steps = 0
        self.log_frequency = 100  # Log detailed step info every N steps only

        # —— Serve training mode parameters ——
        self.serve_training_mode = serve_training_mode
        self.is_serving = False
        self.steps_in_serve = 0
        self.serve_position_reward = 0.0
        self.prev_action = None
        self.serve_count = 0

        # Load pretrained left-agent (opponent) model
        # Note: This is optional - if it fails, we'll use random opponent actions (which is fine)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if opponent model is compatible with current frame_stack
        # Try multiple possible paths for the opponent model
        possible_paths = [
            f"./leftmodel_fs{frame_stack}.zip",  # Frame-stack specific version (preferred)
            "./leftmodel.zip",  # Original version (may be incompatible)
            "./Images/leftmodel",  # Extracted version in Images folder
            "./leftmodel",  # Extracted version in root
        ]

        self.opponent_model = None
        self.opponent_model_compatible = False
        self.opponent_prediction_warning_shown = False

        for left_model_path in possible_paths:
            if os.path.exists(left_model_path):
                try:
                    print(f"Attempting to load opponent model from {left_model_path}...")
                    # Try to load without env parameter (let SB3 handle it)
                    # We'll catch any errors and fall back to random actions
                    self.opponent_model = A2C.load(
                        left_model_path,
                        device=device
                    )
                    # Try to set policy to device
                    if hasattr(self.opponent_model, 'policy'):
                        self.opponent_model.policy.to(device)

                    # Check observation space compatibility
                    if hasattr(self.opponent_model, 'observation_space'):
                        expected_shape = self.opponent_model.observation_space.shape
                        current_shape = (frame_stack, img_size[1], img_size[0])

                        if expected_shape == current_shape:
                            self.opponent_model_compatible = True
                            print(f"✓ Successfully loaded opponent model from {left_model_path}")
                            print(f"  Observation shape matches: {current_shape}")
                        else:
                            print(f"⚠️ Opponent model loaded but observation shape MISMATCH:")
                            print(f"   Expected by model: {expected_shape}")
                            print(f"   Current env:       {current_shape}")
                            print(f"   → Will use RANDOM opponent actions instead")
                            self.opponent_model = None  # Disable incompatible model
                            continue
                    else:
                        # If we can't check compatibility, assume it might work
                        self.opponent_model_compatible = True
                        print(f"✓ Loaded opponent model from {left_model_path} (compatibility unknown)")

                    break
                except (MemoryError, ValueError, KeyError, OSError) as e:
                    print(f"⚠️ Could not load opponent model from {left_model_path}")
                    print(f"   Reason: {type(e).__name__}: {e}")
                    print(f"   → Continuing without opponent model (will use random actions)")
                    continue
                except Exception as e:
                    print(f"⚠️ Unexpected error loading opponent model: {type(e).__name__}: {e}")
                    print(f"   → Continuing without opponent model")
                    continue

        if self.opponent_model is None or not self.opponent_model_compatible:
            print("\n" + "="*60)
            print("ℹ️ OPPONENT MODEL STATUS: Not loaded or incompatible")
            print("="*60)
            print("Training will use RANDOM opponent actions (this is normal for initial training)")
            print(f"If you want opponent model support with frame_stack={frame_stack}:")
            print(f"  1. Train a left-side agent with frame_stack={frame_stack}")
            print(f"  2. Save it as ./leftmodel_fs{frame_stack}.zip")
            print("="*60 + "\n")

        # —— STAGE 1: Run branch mapping probe if requested ——
        if run_probe:
            print("\n" + "="*60)
            print("⚠️ IMPORTANT: DIRECTIONAL SHAPING IS DISABLED")
            print("="*60)
            print("Until the probe successfully maps action branches, all")
            print("directional rewards (LEFT/RIGHT/FWD/BACK) are set to 0.")
            print("This prevents reinforcing wrong behaviors if branch meanings")
            print("are backwards or unknown.")
            print("\nBall-centering shaping is ACTIVE to prevent edge drift.")
            print("="*60 + "\n")
            self.branch_mapping = self._probe_action_branches()
        else:
            # Default mapping (will be overridden if probe runs)
            self.branch_mapping = {}
            print("\n" + "="*60)
            print("ℹ️ Probe disabled - using default action mapping")
            print("="*60 + "\n")

    def _mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Horizontally flip a (F,H,W) or (1,H,W) tensor.
        """
        mirrored = obs[..., ::-1]
        return mirrored

    def _detect_paddle_centroid_rgb(self, raw_rgb, use_roi=True, save_debug=False, debug_name="paddle"):
        """
        Detect orange/red paddle centroid from raw RGB frame (H, W, C) or (C, H, W).

        Args:
            raw_rgb: Raw RGB frame
            use_roi: If True, only search right half of frame (for Player 2 paddle)
            save_debug: If True, save debug images when detection fails
            debug_name: Name prefix for debug images

        Returns:
            (cx, cy) normalized to [0,1], or None if not detected.
        """
        # Handle both (C, H, W) and (H, W, C) formats
        if raw_rgb.shape[0] == 3:  # (C, H, W)
            frame = raw_rgb.transpose(1, 2, 0)  # → (H, W, C)
        else:
            frame = raw_rgb.copy()

        h, w = frame.shape[:2]
        roi_offset_x = 0

        # Apply ROI: only search right half for Player 2 paddle
        if use_roi:
            roi_offset_x = w // 2
            frame_roi = frame[:, roi_offset_x:, :]  # Right half
        else:
            frame_roi = frame

        # Convert RGB to HSV
        frame_hsv = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2HSV)

        # Orange/Red paddle detection in HSV (WIDENED THRESHOLDS)
        # Wider range to handle different lighting conditions:
        # H: 0-35 (covers red-orange spectrum)
        # S: 80-255 (lowered from 100 to catch less saturated orange)
        # V: 80-255 (lowered from 100 to catch darker orange)
        lower_orange = np.array([0, 80, 80])
        upper_orange = np.array([35, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_orange, upper_orange)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or len(contours) == 0:
            # Save debug frames if requested
            if save_debug:
                debug_dir = "./debug_probe"
                os.makedirs(debug_dir, exist_ok=True)

                # Save original frame
                cv2.imwrite(f"{debug_dir}/{debug_name}_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Save ROI frame if used
                if use_roi:
                    cv2.imwrite(f"{debug_dir}/{debug_name}_roi.png", cv2.cvtColor(frame_roi, cv2.COLOR_RGB2BGR))

                # Save mask (binary threshold result)
                cv2.imwrite(f"{debug_dir}/{debug_name}_mask.png", mask)

                # Save HSV visualization
                cv2.imwrite(f"{debug_dir}/{debug_name}_hsv.png", frame_hsv)

            return None

        # Get largest contour (paddle)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] == 0:
            if save_debug:
                debug_dir = "./debug_probe"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/{debug_name}_zero_moment.png", mask)
            return None

        # Centroid in pixel coordinates (relative to ROI)
        cx_roi = M["m10"] / M["m00"]
        cy_roi = M["m01"] / M["m00"]

        # Adjust for ROI offset
        cx = cx_roi + roi_offset_x
        cy = cy_roi

        # Normalize to [0, 1] (relative to full frame)
        return (cx / w, cy / h)

    def _detect_paddle_centroid(self, img):
        """
        Detect orange/red paddle centroid from preprocessed frame (1, H, W) in [0,1].
        Returns (cx, cy) normalized to [0,1], or None if not detected.
        Note: This won't work well with grayscale images. Use _detect_paddle_centroid_rgb for color detection.
        """
        # Convert to uint8 for OpenCV
        frame = (img[0] * 255).astype(np.uint8)

        # Convert grayscale to BGR for color detection
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Orange/Red paddle detection in HSV
        # Orange typically: H=10-25, S=100-255, V=100-255
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_orange, upper_orange)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (paddle)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] == 0:
            return None

        # Centroid in pixel coordinates
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Normalize to [0, 1]
        h, w = frame.shape[:2]
        return (cx / w, cy / h)

    def _detect_ball_centroid(self, img):
        """
        Detect yellow ball centroid from preprocessed frame (1, H, W) in [0,1].
        Returns (bx, by) normalized to [0,1], or None if not detected.
        """
        # Convert to uint8 for OpenCV
        frame = (img[0] * 255).astype(np.uint8)

        # Convert grayscale to BGR for color detection
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Yellow ball detection in HSV
        # Yellow typically: H=20-35, S=100-255, V=100-255
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (ball)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] == 0:
            return None

        # Centroid in pixel coordinates
        bx = M["m10"] / M["m00"]
        by = M["m01"] / M["m00"]

        # Normalize to [0, 1]
        h, w = frame.shape[:2]
        return (bx / w, by / h)

    def _probe_action_branches(self):
        """
        Stage 1: Branch mapping probe - test each action to determine what it controls.
        Tests each branch independently and measures paddle centroid movement.
        """
        print(f"\n{'='*60}")
        print(f"STAGE 1: ACTION BRANCH MAPPING PROBE")
        print(f"{'='*60}")
        print("Testing each action branch to identify:")
        print("  - Which controls VERTICAL movement (forward/back)")
        print("  - Which controls LATERAL movement (left/right)")
        print("  - Which triggers HIT/SERVE action")
        print(f"{'='*60}\n")

        test_cases = [
            # Test branch 0: (0,1,1) vs (2,1,1)
            ("Branch 0", [(0,1,1), (1,1,1), (2,1,1)], "value 0", "value 1", "value 2"),
            # Test branch 1: (1,0,1) vs (1,2,1)
            ("Branch 1", [(1,0,1), (1,1,1), (1,2,1)], "value 0", "value 1", "value 2"),
            # Test branch 2: (1,1,0) vs (1,1,2)
            ("Branch 2", [(1,1,0), (1,1,1), (1,1,2)], "value 0", "value 1", "value 2"),
        ]

        results = {}

        for branch_name, actions, *labels in test_cases:
            print(f"\nTesting {branch_name}:")
            print(f"-" * 40)

            branch_results = []

            for action_tuple in actions:
                # Reset environment
                self.env.reset()
                self.frames.clear()
                for _ in range(self.frame_stack):
                    self.frames.append(np.zeros((1, self.img_size[1], self.img_size[0]), dtype=np.float32))

                # Record initial paddle position
                positions = []

                # Step 5 times with this action
                for step in range(5):
                    action = np.array(action_tuple, dtype=np.int32)
                    actions_dict = {self.agent: action, self.agent_other: np.array([1,1,1], dtype=np.int32)}

                    obs_dict, _, _, _ = self.env.step(actions_dict)
                    raw_img = obs_dict[self.agent_obs]['observation'][0]

                    # Use raw RGB image for color detection (before grayscale conversion)
                    raw_rgb = np.asarray(raw_img, dtype=np.uint8)

                    # Detect paddle centroid from raw RGB with debug enabled
                    debug_name = f"{branch_name.replace(' ', '_')}_action_{action_tuple}_step_{step}"

                    # Try FULL FRAME detection first (no ROI restriction)
                    centroid = self._detect_paddle_centroid_rgb(
                        raw_rgb,
                        use_roi=False,  # Try full frame first
                        save_debug=(step == 0),  # Save debug on first step only
                        debug_name=debug_name + "_fullframe"
                    )

                    # If that fails, try with ROI (right half only)
                    if centroid is None:
                        centroid = self._detect_paddle_centroid_rgb(
                            raw_rgb,
                            use_roi=True,
                            save_debug=(step == 0),
                            debug_name=debug_name + "_roi"
                        )

                    if centroid:
                        positions.append(centroid)

                # Calculate average position
                if positions:
                    avg_x = np.mean([p[0] for p in positions])
                    avg_y = np.mean([p[1] for p in positions])
                    branch_results.append((action_tuple, avg_x, avg_y))
                    print(f"  Action {action_tuple}: avg position ({avg_x:.3f}, {avg_y:.3f})")
                else:
                    print(f"  Action {action_tuple}: ⚠️ Paddle not detected")
                    branch_results.append((action_tuple, None, None))

            # Analyze movement
            if len(branch_results) >= 2 and all(r[1] is not None for r in branch_results):
                dx_01 = branch_results[2][1] - branch_results[0][1]  # value 2 - value 0
                dy_01 = branch_results[2][2] - branch_results[0][2]

                print(f"\n  📊 Movement Analysis:")
                print(f"    Δx (value 2 - value 0): {dx_01:+.3f}")
                print(f"    Δy (value 2 - value 0): {dy_01:+.3f}")

                # Determine what this branch controls
                if abs(dy_01) > abs(dx_01) * 2:
                    if dy_01 > 0.05:
                        control = "VERTICAL (value 2=DOWN/BACK, value 0=UP/FORWARD)"
                    elif dy_01 < -0.05:
                        control = "VERTICAL (value 2=UP/FORWARD, value 0=DOWN/BACK)"
                    else:
                        control = "VERTICAL (minimal movement)"
                elif abs(dx_01) > abs(dy_01) * 2:
                    if dx_01 > 0.05:
                        control = "LATERAL (value 2=RIGHT, value 0=LEFT)"
                    elif dx_01 < -0.05:
                        control = "LATERAL (value 2=LEFT, value 0=RIGHT)"
                    else:
                        control = "LATERAL (minimal movement)"
                else:
                    control = "HIT/ROTATION (minimal centroid movement)"

                print(f"    ✓ {branch_name} controls: {control}")
                results[branch_name] = (control, dx_01, dy_01)
            else:
                print(f"    ⚠️ Could not analyze - paddle detection failed")
                results[branch_name] = ("UNKNOWN", 0, 0)

        # Print final mapping
        print(f"\n{'='*60}")
        print(f"BRANCH MAPPING RESULTS:")
        print(f"{'='*60}")
        for branch, (control, dx, dy) in results.items():
            print(f"{branch}: {control}")
        print(f"{'='*60}\n")

        return results

    def _preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # Resize
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Grayscale or keep color
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # Normalize to [0,1]
        obs = obs.astype(np.float32) / 255.0
        # Mirror observation for left-agent camera
        obs = self._mirror_obs(obs)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()
        raw = obs_dict[self.agent_obs]['observation'][0]
        img = self._preprocess(np.asarray(raw, dtype=np.float32))

        # Initialize frame stack with mirrored first frame
        for _ in range(self.frame_stack):
            self.frames.append(img)

        # Reset reward shaping variables
        self.prev_game_state = None
        self.steps_since_serve = 0
        self.hold_streak = 0
        self.break_streak = 0
        self.violation_occurred = False

        # Reset serve training variables
        self.is_serving = False
        self.steps_in_serve = 0
        self.serve_position_reward = 0.0
        self.prev_action = None

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def mirror_action(self, action: np.ndarray) -> np.ndarray:
        """Mirror horizontal action component: swap left↔right (1↔2)"""
        mirrored = action.copy()
        if mirrored[1] == 1:
            mirrored[1] = 2
        elif mirrored[1] == 2:
            mirrored[1] = 1
        return mirrored

    def get_opponent_action(self):
        """
        Use the pretrained A2C model (trained on right-court) to act as the left agent.
        Mirror the stacked frames back into right-court view for inference,
        then un-mirror the action output.

        Falls back to random actions if:
        - No opponent model loaded
        - Opponent model incompatible (observation shape mismatch)
        - Prediction fails for any reason
        """
        # Return random action if no opponent model or incompatible
        if self.opponent_model is None or not self.opponent_model_compatible:
            return self.action_space.sample()

        try:
            obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)
            # Flip back to right-court view for the model
            obs_for_model = self._mirror_obs(obs)
            obs_batch = np.expand_dims(obs_for_model, axis=0)  # (1, stack, H, W)
            action, _states = self.opponent_model.predict(obs_batch, deterministic=True)
            # Un-mirror the horizontal component of action
            return self.mirror_action(action[0]).astype(np.int32)

        except (ValueError, RuntimeError, Exception) as e:
            # Prediction failed - likely due to observation shape mismatch
            # Show warning once, then silently fall back to random actions
            if not self.opponent_prediction_warning_shown:
                print(f"\n⚠️ WARNING: Opponent model prediction failed!")
                print(f"   Error: {type(e).__name__}: {str(e)[:100]}")
                print(f"   → Falling back to RANDOM opponent actions for all future steps")
                print(f"   → This is expected if frame_stack changed since opponent was trained\n")
                self.opponent_prediction_warning_shown = True
                # Disable opponent model to avoid repeated errors
                self.opponent_model = None
                self.opponent_model_compatible = False

            return self.action_space.sample()

    def step(self, action):
        # —— ACTION VALIDATION (prevents Unity crashes from invalid actions) ——
        # Ensure action is valid before sending to Unity
        action = np.asarray(action, dtype=np.int32)

        # Validate shape
        if action.shape != (3,):
            print(f"⚠️ Invalid action shape {action.shape}, expected (3,). Clamping to (3,)")
            action = action.flatten()[:3] if action.size >= 3 else np.array([1, 1, 1], dtype=np.int32)

        # Validate values are in range [0, 2]
        action = np.clip(action, 0, 2).astype(np.int32)

        # Get opponent action (also validated)
        opponent_action = self.get_opponent_action()
        opponent_action = np.asarray(opponent_action, dtype=np.int32)
        opponent_action = np.clip(opponent_action, 0, 2).astype(np.int32)

        # Send both right-agent action and left-agent (opponent) action
        actions = {self.agent: action,
                   self.agent_other: opponent_action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        raw_list = obs_dict[self.agent_obs]['observation']
        raw_img = raw_list[0]
        vec_array = raw_list[1]

        # —— EXTRACT GAME STATE EARLY (needed for shaping logic below) ——
        game_state = None
        try:
            if vec_array is not None and len(vec_array) > 0:
                game_state = int(np.array(vec_array, dtype=np.float32)[0])
        except Exception as e:
            print(f"⚠️ Warning: Could not extract game_state: {e}")
            game_state = 0  # Default to RALLY state if extraction fails

        # —— VECTOR OBSERVATION INSPECTION (Stage 4) ——
        if not hasattr(self, '_vec_logged'):
            print(f"\n{'='*60}")
            print(f"VECTOR OBSERVATION ANALYSIS:")
            print(f"{'='*60}")
            print(f"vec_array length: {len(vec_array)}")
            print(f"vec_array contents: {vec_array}")
            print(f"vec_array[0] (game_state): {vec_array[0]}")
            if len(vec_array) > 1:
                print(f"Additional vector observations available!")
                for i in range(1, min(len(vec_array), 10)):
                    print(f"  vec_array[{i}]: {vec_array[i]}")
            print(f"{'='*60}\n")
            self._vec_logged = True

        # Increment step counter for logging control
        self.total_steps += 1

        # Preprocess & mirror
        img = self._preprocess(np.asarray(raw_img, dtype=np.float32))
        self.frames.append(img)
        stacked = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # Initialize shaped reward accumulator
        shaped = 0.0

        # —— BALL-BASED SHAPING (prevents edge drift) ——
        # Detect ball position to encourage ball tracking
        ball_centroid = self._detect_ball_centroid(img)
        ball_centering_reward = 0.0

        if ball_centroid is not None and game_state is not None and game_state in [0, 1, 2]:  # During RALLY, RECEIVE, SERVE
            ball_x, ball_y = ball_centroid

            # Reward keeping ball near horizontal center (prevents left/right edge drift)
            center_x = 0.5
            distance_from_center = abs(ball_x - center_x)

            if distance_from_center < 0.15:  # Ball within 15% of center
                ball_centering_reward = 0.3
            elif distance_from_center < 0.25:  # Ball within 25% of center
                ball_centering_reward = 0.1
            elif distance_from_center > 0.4:  # Ball too far from center
                ball_centering_reward = -0.2

            shaped += ball_centering_reward

        # —— SURVIVAL REWARD (encourages staying in rally) ——
        # Small positive reward for keeping rally alive
        # This smooths value estimates and reduces policy oscillations
        survival_reward = 0.0
        if game_state is not None and game_state in [0, 1, 2]:  # RALLY, RECEIVE, SERVE (ball is in play)
            survival_reward = 0.05  # Increased from 0.01 to provide stronger signal
            shaped += survival_reward

        # Compute shaped reward
        base_r = rewards[self.agent] - rewards[self.agent_other]
        # game_state already extracted above - no need to extract again

        # Potential-based reward shaping (add to accumulated shaped rewards)
        gamma = 0.99
        phi_cur = self.phi.get(self.prev_game_state, 0.0)
        phi_next = self.phi.get(game_state, 0.0)  # Use .get() to handle None case
        shaped += base_r + (gamma * phi_next - phi_cur)

        # Additional state-based bonuses (added, not replacing)
        if game_state is not None:
            if game_state == 0:
                shaped += 0.05
            if game_state == 1:
                shaped -= 0.02
            if game_state == 2:
                shaped += 0.05
            if game_state == 3:
                shaped += 0.8
            if game_state == 4:
                shaped -= 0.4
            if game_state == 5:
                shaped -= 1.0
            if game_state == 6:
                shaped += 2.0

        # Serve-break/lost bonuses
        if self.prev_game_state == 1 and game_state == 6:
            shaped += 0.5  # serve_break_bonus
        elif self.prev_game_state == 2 and game_state == 5:
            shaped -= 0.5  # serve_lost_penalty

        # Speed incentive
        if game_state == 2:
            self.steps_since_serve = 0
        elif self.steps_since_serve is not None:
            self.steps_since_serve += 1

        if game_state == 6 and self.prev_game_state == 2:
            shaped += 0.4 * (1 - self.steps_since_serve / self.max_fast)
        elif game_state == 5 and self.prev_game_state == 2:
            shaped -= 0.3 * (1 - self.steps_since_serve / self.max_fast)

        # Streak tracking
        if self.prev_game_state == 2 and game_state == 6:
            self.hold_streak += 1
            self.break_streak = 0
            shaped += 0.15 * min(self.hold_streak, 10)
        elif self.prev_game_state == 1 and game_state == 5:
            self.break_streak += 1
            self.hold_streak = 0
            shaped -= 0.1 * min(self.break_streak, 5)

        # Violation tracking
        if game_state in (1, 2):
            self.violation_occurred = False
        elif game_state in (3, 4):
            self.violation_occurred = True
        elif game_state in (5, 6) and not self.violation_occurred:
            shaped += 0.2

        # Small penalty for prolonged rallies
        if game_state not in (5, 6):
            shaped -= 0.005

        # —— HIT ACTION (action[2]) SHAPING ——
        # Stage 8: Encourage using hit action at right time, penalize spam
        if len(action) > 2:  # Make sure action[2] exists
            hit_action = action[2]

            # During SERVE state: hitting is required
            if game_state == 2 and hit_action > 0:
                shaped += 2.0 * (3.0 if self.serve_training_mode else 1.0)
            # During RECEIVE/RALLY: hitting is good
            elif game_state in [0, 1] and hit_action > 0:
                shaped += 0.5
            # Spamming hit when not near ball (during transitions)
            elif game_state in [3, 4, 5, 6] and hit_action > 0:
                shaped -= 0.2

        # —— SERVE STATE REWARDS (MINIMAL GUIDANCE ENABLED) ——
        if game_state == 2:  # SERVE STATE
            self.is_serving = True
            self.steps_in_serve += 1

            # MINIMAL SERVE GUIDANCE: Encourage forward movement without assuming full semantics
            # We don't know which branch is "forward", so we reward ANY branch going to value 2
            # This gives just enough signal to learn approach behavior
            vertical_reward = 0.0

            # Encourage exploration of "high" values (likely forward/hit)
            # Count how many branches are set to value 2
            high_value_count = sum(1 for a in action if a == 2)
            if high_value_count >= 1:
                vertical_reward = 0.3 * high_value_count  # Reward for trying "2" values

            shaped += vertical_reward

            # Print serve debug info only occasionally (reduce logging overhead)
            if self.steps_in_serve <= 8 and self.total_steps % self.log_frequency == 0:
                v_names = {0: "action[0]=0", 1: "action[0]=1", 2: "action[0]=2"}
                h_names = {0: "action[1]=0", 1: "action[1]=1", 2: "action[1]=2"}
                print(f"   🎾 SERVE Step {self.steps_in_serve}: V={v_names.get(action[0], '?')} H={h_names.get(action[1], '?')} | Reward: {vertical_reward:+.2f}")

        # —— RECEIVE STATE REWARDS (DISABLED UNTIL BRANCH MAPPING CONFIRMED) ——
        elif game_state == 1:  # RECEIVE STATE
            self.steps_in_serve += 1

            # TEMPORARILY DISABLED: All directional shaping disabled until probe confirms branch mapping
            # Previous shaping had strong FWD penalty (-8) which prevented paddle from approaching ball
            # Also had lateral bias that may have caused right-edge drift
            vertical_reward = 0.0
            horizontal_reward = 0.0

            # Print receive debug info only occasionally (reduce logging overhead)
            if self.steps_in_serve <= 8 and self.total_steps % self.log_frequency == 0:
                v_names = {0: "action[0]=0", 1: "action[0]=1", 2: "action[0]=2"}
                h_names = {0: "action[1]=0", 1: "action[1]=1", 2: "action[1]=2"}
                print(f"   🛡️ RECEIVE Step {self.steps_in_serve}: V={v_names.get(action[0], '?')} H={h_names.get(action[1], '?')} | Shaping DISABLED (awaiting probe)")

        # Reset serve tracking when transitioning out of serve/receive
        if game_state not in [1, 2]:
            if self.is_serving:
                self.serve_position_reward = 0.0
                self.steps_in_serve = 0
                self.is_serving = False

        # —— SUCCESSFUL SERVE (Scaled down 100x) ——
        if self.prev_game_state == 2 and game_state == 0:
            multiplier = 3.0 if self.serve_training_mode else 1.0
            shaped += 1.0 * multiplier  # Success bonus
            if self.prev_action is not None and self.prev_action[0] == 2:
                shaped += 1.0 * multiplier  # Extra bonus for forward movement
            if self.prev_action is not None and self.prev_action[1] == 0:
                shaped += 1.0 * multiplier  # Extra bonus for straight facing
            shaped += self.serve_position_reward
            print(f"✓✓✓ SUCCESSFUL SERVE! Total Reward: {shaped:.2f}")
            self.serve_count += 1
            self.is_serving = False
            self.serve_position_reward = 0.0
            self.steps_in_serve = 0

        # —— FAILED SERVE (Scaled down 100x) ——
        if self.prev_game_state == 2 and game_state in (3, 5):
            multiplier = 3.0 if self.serve_training_mode else 1.0
            shaped -= 2.0 * multiplier  # Penalty for failed serve
            print(f"✗✗✗ FAILED SERVE! Penalty: {shaped:.2f}")
            self.serve_count += 1
            self.is_serving = False
            self.serve_position_reward = 0.0
            self.steps_in_serve = 0

        # State transition logging
        state_names = {0: "RALLY", 1: "RECEIVE", 2: "SERVE", 3: "OUT", 4: "NET", 5: "LOST", 6: "WON"}
        if self.prev_game_state != game_state and game_state in [1, 2]:
            print(f"\n{'='*60}")
            print(f"🎯 STATE CHANGE: {state_names.get(self.prev_game_state, '?')} → {state_names.get(game_state, '?')}")
            print(f"{'='*60}")

        self.prev_game_state = game_state
        self.prev_action = action.copy() if hasattr(action, 'copy') else action
        done = terminations[self.agent]

        # CRITICAL FIX: shaped already includes base_r, don't double-count!
        total_r = shaped  # NOT base_r + shaped (that would add base_r twice)

        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print("Rewards: ", total_r, rewards[self.agent_other])

        return (stacked, total_r, done, False, infos[self.agent])

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
