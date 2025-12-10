import os
import numpy as np
from collections import deque
import cv2
import torch
from stable_baselines3 import A2C
# Build a Python class for your solution, do preprocessing (image processing, frame stacking, etc) here.
# During the competition, only the policy function is called at each time step, providing the observation and reward for that time step only.
# Your agent is expected to return actions to be executed.
class TeamX:
    def __init__(self, frame_stack=64, img_size=(168, 84)):
        """
        Initialize TeamX agent.
        
        Args:
            frame_stack: Number of frames to stack (should match training, default 64)
            img_size: Target image size (H, W) - should match training (168, 84)
            side: "right" (default, matches training) or "left" (mirror obs/actions)
        """
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.frames = deque(maxlen=frame_stack)
        # Side selection: default right (matches training); set AGENT_SIDE=left to flip
        self.side = os.getenv("AGENT_SIDE", "right").lower()
        if self.side not in ("right", "left"):
            self.side = "right"

        # Load your checkpoint for policy network
        # TODO: Update this path to point to your trained model file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "pickleball_agent.zip"  # CHANGE THIS to your model path
        print(f"Loading model from {model_path} on device {device}")
        self.model = A2C.load(model_path, device=device)
        self.model.policy.to(device)
        self.model.set_env(None)  # Set to None for inference only
    
    def _mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """Horizontally flip (C,H,W) observation."""
        return obs[..., ::-1]

    def _mirror_action(self, action: np.ndarray) -> np.ndarray:
        """Swap left/right in action component index 1 (0:none,1:right,2:left)."""
        mirrored = action.copy()
        if mirrored[1] == 1:
            mirrored[1] = 2
        elif mirrored[1] == 2:
            mirrored[1] = 1
        return mirrored

    def _preprocess(self, observation, mirror: bool):
        """
        Preprocess observation to match training format.
        Observation comes in as (C, H, W) from Unity.
        """
        # Transpose from (C, H, W) → (H, W, C)
        obs = observation.transpose(1, 2, 0)
        
        # Resize to match training size
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        
        # Normalize to [0, 1] to match training
        obs = obs.astype(np.float32) / 255.0
        if mirror:
            obs = self._mirror_obs(obs)
        return obs
    
    # Your policy takes only visual representation as input, 
    # and reward is 1 when you score, -1 when your opponent scores
    # Your policy function returns actions
    def policy(self, observation, reward):
        # Implement your solution here

        # Preprocess observation
        mirror_obs = (self.side == "left")  # if on left, mirror input to look like right-side view
        processed_obs = self._preprocess(observation, mirror=mirror_obs)
        
        # Add to frame stack
        self.frames.append(processed_obs)
        
        # Get stacked observation
        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)
        
        # Pad if we don't have enough frames yet (for initial steps)
        if len(self.frames) < self.frame_stack:
            # Repeat the first frame to fill the stack
            while len(self.frames) < self.frame_stack:
                self.frames.appendleft(processed_obs)
            stacked_obs = np.concatenate(list(self.frames), axis=0)

        # Use your policy network here
        obs_batch = np.expand_dims(stacked_obs, axis=0)  # (1, stack, H, W)
        action, _states = self.model.predict(obs_batch, deterministic=True)
        action = action[0].astype(np.int32)  # Remove batch dimension

        # If playing left side, un-mirror horizontal action so environment receives left-oriented control
        if self.side == "left":
            action = self._mirror_action(action)

        return action


