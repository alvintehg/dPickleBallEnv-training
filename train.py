import os
# Disable ML-Agents remote registry to avoid manifest fetch issues
os.environ["MLAGENTS_DISABLE_REGISTRY"] = "1"
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import time
import os

from mylib import SharedObsUnityGymWrapper, CustomCNN

# --- CONFIG ---
MODEL_DIR = "./model"  # Directory to save models
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.zip")  # Path for resuming training
TOTAL_TIMESTEPS = 1_000_000  # Total timesteps to train (or continue training)
# Note: If model already trained 1M steps, increase this to continue (e.g., 2_000_000 for 2M total)
CHECKPOINT_FREQ = 10_000  # Save checkpoint every N steps (lower = more frequent saves, safer but uses more disk space)

# —— STAGE 1: ACTION BRANCH PROBE ——
RUN_ACTION_PROBE = True  # Set to True to run branch mapping probe at startup

# ⚠️ IMPORTANT: Update this path to point to your Unity build location!
# You can also set the UNITY_BUILD_PATH environment variable instead
UNITY_BUILD_PATH = os.getenv(
    "UNITY_BUILD_PATH",
    r'/Users/Justin/Desktop/dpickleball/dPickleball BuildFiles/Training/Windows/dp.exe'  # DEFAULT - UPDATE THIS!
)

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Setup side channels
string_channel = StringSideChannel()
channel = CustomDataChannel()

# Game config: 213 = match point 21, random serve
# The last digit is the serve mode:
#     1 = left serve only
#     2 = right serve only
#     3 = random serve
# The remaining digits (before the last one) indicate the match point.
# Example: 213 → match point = 21, serve = 3 (random serve)
reward_cum = [0, 0]
channel.send_data(serve=213, p1=reward_cum[0], p2=reward_cum[1])

print("Creating Unity environment...")
print(f"Unity build path: {UNITY_BUILD_PATH}")

# Check if Unity build exists
if not os.path.exists(UNITY_BUILD_PATH):
    print("\n" + "=" * 60)
    print("❌ ERROR: Unity build file not found!")
    print("=" * 60)
    print(f"Path: {UNITY_BUILD_PATH}")
    print("\nPlease update the UNITY_BUILD_PATH in train.py (line 20) or set the")
    print("UNITY_BUILD_PATH environment variable to point to your Unity build.")
    print("\nExample:")
    print('  UNITY_BUILD_PATH = r"C:\\Path\\To\\Your\\Unity\\Build\\dp.exe"')
    print("\nOr set environment variable:")
    print('  set UNITY_BUILD_PATH="C:\\Path\\To\\Your\\Unity\\Build\\dp.exe"')
    print("=" * 60)
    exit(1)

try:
    unity_env = UnityEnvironment(
        UNITY_BUILD_PATH,
        side_channels=[string_channel, channel]
    )
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR: Failed to launch Unity environment!")
    print("=" * 60)
    print(f"Error: {e}")
    print(f"\nPath used: {UNITY_BUILD_PATH}")
    print("\nPlease check:")
    print("  1. The path is correct and points to dp.exe")
    print("  2. The Unity build exists at that location")
    print("  3. You have permission to run the executable")
    print("\nUpdate UNITY_BUILD_PATH in train.py (line 20) or set environment variable.")
    print("=" * 60)
    raise

print("Wrapping environment with serve training wrapper...")
print("🎯 SERVE TRAINING MODE ENABLED - Using 3x reward multiplier for serve behavior")
print("📊 STAGE 6: Using reduced frame_stack=4 for faster learning")
env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, img_size=(168, 84), grayscale=True, serve_training_mode=True, run_probe=RUN_ACTION_PROBE)

print("Creating A2C model...")
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Training on device: {device}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🔥 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected, using CPU")
# --- CHECKPOINT CALLBACK (saves during training) ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=MODEL_DIR,
    name_prefix="checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
    verbose=1,
)

# --- LOAD EXISTING MODEL OR CREATE NEW ONE ---
# Check if we can safely load the existing model
can_load_model = False
print(f"\nChecking for existing model at: {LAST_MODEL_PATH}")
if os.path.isfile(LAST_MODEL_PATH):
    print(f"✓ Model file found: {LAST_MODEL_PATH}")
    try:
        # Load model data to check observation space compatibility
        import zipfile
        import json
        print("  Checking observation space compatibility...")
        with zipfile.ZipFile(LAST_MODEL_PATH, 'r') as zip_file:
            with zip_file.open('data') as data_file:
                saved_data = json.loads(data_file.read().decode('utf-8'))
                saved_obs_space = saved_data.get('observation_space', {})

                # Compare observation space shapes
                if 'shape' in saved_obs_space:
                    saved_shape = tuple(saved_obs_space['shape'])
                    current_shape = env.observation_space.shape

                    if saved_shape == current_shape:
                        can_load_model = True
                        print(f"  ✓ Observation space matches: {current_shape}")
                    else:
                        print(f"  ⚠️ Observation space mismatch!")
                        print(f"    Saved model:   {saved_shape}")
                        print(f"    Current env:   {current_shape}")
                        print(f"    → Cannot load model (likely due to frame_stack change)")
                        print(f"    → Will create NEW model instead")
                        # Backup old model
                        backup_path = LAST_MODEL_PATH.replace('.zip', '_backup_old_framestack.zip')
                        if not os.path.exists(backup_path):
                            import shutil
                            shutil.copy(LAST_MODEL_PATH, backup_path)
                            print(f"    → Backed up old model to: {backup_path}")
                else:
                    print(f"  ⚠️ No observation space shape found in saved model")
                    print(f"    → Will try to load anyway (older model format)")
                    can_load_model = True  # Try to load even if shape not found
    except Exception as e:
        print(f"  ⚠️ Could not check model compatibility: {type(e).__name__}: {e}")
        print(f"    → Will try to load anyway")
        can_load_model = True  # Try to load even if compatibility check fails
else:
    print(f"✗ No model file found at: {LAST_MODEL_PATH}")
    # Try to find the latest checkpoint as fallback
    checkpoint_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('checkpoint_') and f.endswith('.zip')]
    if checkpoint_files:
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
        latest_checkpoint = os.path.join(MODEL_DIR, checkpoint_files[0])
        print(f"  → Found checkpoint: {checkpoint_files[0]}")
        print(f"  → Will try to load from checkpoint instead")
        LAST_MODEL_PATH = latest_checkpoint  # Override to use checkpoint
        can_load_model = True  # Try to load it
    else:
        print(f"  → No checkpoints found either")
        print(f"  → Will create NEW model")

if can_load_model:
    print(f"Loading existing model from {LAST_MODEL_PATH}")
    print("Resuming training...")
    # Don't specify policy_kwargs when loading - let it use stored kwargs
    model = A2C.load(
        LAST_MODEL_PATH,
        env=env,
        device=device,
        # policy_kwargs=policy_kwargs,  # Removed - causes mismatch error
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    reset_timesteps = False  # Don't reset timestep counter when resuming
else:
    print("Creating new model...")
    model = A2C(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        ent_coef=0.01,  # Entropy coefficient - encourages exploration, prevents policy collapse
        tensorboard_log="./tensorboard_logs/"
    )
    reset_timesteps = True  # Reset timestep counter for new model

print("\n" + "="*60)
print("Starting training...")
print("="*60)
print("Your agent (RIGHT paddle - Player 2) will play against random opponent (LEFT)")
print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
print("Watch the terminal for detailed serve/receive logs!")
print("="*60 + "\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        tb_log_name="dpickleball_run01"
    )

    # Save the final model
    model.save(LAST_MODEL_PATH)
    print(f"\nTraining complete! Final model saved as {LAST_MODEL_PATH}")
    
    # Also save with timestamp
    model_name = f"pickleball_agent_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)
    print(f"Also saved as {model_name}.zip")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    # Save current progress
    try:
        model.save(LAST_MODEL_PATH)
        model_name = f"pickleball_agent_interrupted_{time.strftime('%Y%m%d-%H%M%S')}"
        model.save(model_name)
        print(f"Model saved as {LAST_MODEL_PATH} and {model_name}.zip")
        print("You can resume training by running this script again!")
    except Exception as save_error:
        print(f"Warning: Could not save model: {save_error}")

except Exception as e:
    print(f"\nTraining error occurred: {e}")
    print(f"Error type: {type(e).__name__}")

    # Check if this is a Unity communicator crash
    if "UnityCommunicatorStoppedException" in str(type(e).__name__) or "Communicator has exited" in str(e):
        print("\n" + "="*60)
        print("⚠️ UNITY COMMUNICATOR CRASHED")
        print("="*60)
        print("Unity build disconnected during training. This can happen due to:")
        print("  1. Unity build crashed (out of memory, internal error)")
        print("  2. Invalid actions sent to Unity (now validated, shouldn't happen)")
        print("  3. ML-Agents version mismatch between Python and Unity build")
        print("\nTo diagnose the root cause, check Unity's log file:")
        import os as os_module
        username = os_module.environ.get('USERNAME', os_module.environ.get('USER', 'unknown'))
        unity_log_paths = [
            f"C:\\Users\\{username}\\AppData\\LocalLow\\DefaultCompany\\dPickleball\\Player.log",
            f"C:\\Users\\{username}\\AppData\\LocalLow\\*\\*\\Player.log",
            "./Player.log"  # Sometimes in build directory
        ]
        print("\nPossible Unity log locations:")
        for log_path in unity_log_paths:
            print(f"  - {log_path}")
        print("\nThe log file will show the exact error that caused Unity to crash.")
        print("="*60 + "\n")

    # Try to save progress if possible
    try:
        print("Attempting to save current progress...")
        model.save(LAST_MODEL_PATH)
        model_name = f"pickleball_agent_crashed_{time.strftime('%Y%m%d-%H%M%S')}"
        model.save(model_name)
        print(f"Progress saved to {LAST_MODEL_PATH} and {model_name}.zip")
        print("You can resume training by running this script again!")
    except Exception as save_error:
        print(f"Could not save progress: {save_error}")
        print("Check checkpoint files in ./model/ directory")

    # Re-raise the error so user knows what happened
    raise

finally:
    try:
        env.close()
        print("Environment closed")
    except:
        pass  # Ignore errors when closing
