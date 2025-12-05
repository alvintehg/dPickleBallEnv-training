# Collaborative Training Guide

This guide explains how multiple friends can take turns training the model and sharing progress via GitHub.

## 🔄 Workflow Overview

```
Friend A trains → Commits model → Friend B pulls → Friend B trains → Commits model → Friend C pulls → ...
```

Each friend continues training from where the previous friend left off!

---

## 👤 For Each Friend (Training Turn)

### Step 1: Pull Latest Model from GitHub

```bash
git pull origin main
```

This downloads the latest trained model from the previous friend.

### Step 2: Verify Model Exists

Check that `model/last_model.zip` exists:
```bash
# Windows PowerShell
Test-Path model/last_model.zip

# Should return: True
```

### Step 3: Update Unity Build Path (if needed)

Edit `train.py` line 37 - Update to your Unity build path:
```python
unity_env = UnityEnvironment(
    r"C:\Your\Path\To\dp.exe",  # Update this!
    side_channels=[string_channel, channel]
)
```

### Step 4: Train for Your Turn

```bash
conda activate dpickleball
python train.py
```

**Training will automatically:**
- ✅ Load `model/last_model.zip` (from previous friend)
- ✅ Resume from their timestep
- ✅ Continue training for `TOTAL_TIMESTEPS` (default: 1,000,000)
- ✅ Save checkpoints every 10,000 steps
- ✅ Save final model as `model/last_model.zip`

### Step 5: Commit and Push Your Progress

After training completes (or when you want to hand off):

```bash
# Add the updated model files
git add model/last_model.zip model/checkpoint_*.zip

# Commit with a message showing your progress
git commit -m "Training progress: [X] timesteps completed"

# Push to GitHub
git push origin main
```

**Example commit messages:**
- `"Training progress: 1,000,000 timesteps completed"`
- `"Training progress: 2,000,000 timesteps completed"`
- `"Training checkpoint at 500,000 timesteps"`

---

## 📋 Detailed Example: 3 Friends Training

### Friend A (First Training Session)

1. **Clone repository:**
   ```bash
   git clone https://github.com/alvintehg/dPickleBallEnv-training.git
   cd dPickleBallEnv-training
   ```

2. **Train from scratch:**
   ```bash
   conda activate dpickleball
   python train.py
   ```
   - Trains from 0 → 1,000,000 steps
   - Saves `model/last_model.zip`

3. **Commit and push:**
   ```bash
   git add model/last_model.zip model/checkpoint_*.zip
   git commit -m "Initial training: 1,000,000 timesteps"
   git push origin main
   ```

---

### Friend B (Second Training Session)

1. **Pull latest model:**
   ```bash
   git pull origin main
   ```
   - Downloads `model/last_model.zip` (1,000,000 steps from Friend A)

2. **Train:**
   ```bash
   conda activate dpickleball
   python train.py
   ```
   - Automatically loads Friend A's model
   - Continues from 1,000,000 → 2,000,000 steps
   - Saves updated `model/last_model.zip`

3. **Commit and push:**
   ```bash
   git add model/last_model.zip model/checkpoint_*.zip
   git commit -m "Training progress: 2,000,000 timesteps completed"
   git push origin main
   ```

---

### Friend C (Third Training Session)

1. **Pull latest model:**
   ```bash
   git pull origin main
   ```
   - Downloads `model/last_model.zip` (2,000,000 steps from Friend B)

2. **Train:**
   ```bash
   conda activate dpickleball
   python train.py
   ```
   - Continues from 2,000,000 → 3,000,000 steps

3. **Commit and push:**
   ```bash
   git add model/last_model.zip model/checkpoint_*.zip
   git commit -m "Training progress: 3,000,000 timesteps completed"
   git push origin main
   ```

---

## ⚙️ Configuration Options

### Change Training Duration Per Turn

Edit `train.py` line 14:
```python
TOTAL_TIMESTEPS = 500_000  # Train for 500k steps per turn
# or
TOTAL_TIMESTEPS = 2_000_000  # Train for 2M steps per turn
```

### Change Checkpoint Frequency

Edit `train.py` line 16:
```python
CHECKPOINT_FREQ = 50_000  # Save checkpoint every 50k steps
```

---

## ✅ Pre-Flight Checklist (Before Each Training Session)

- [ ] Pulled latest changes: `git pull origin main`
- [ ] Verified `model/last_model.zip` exists
- [ ] Updated Unity build path in `train.py` (if needed)
- [ ] Conda environment activated: `conda activate dpickleball`
- [ ] All dependencies installed
- [ ] Ready to train!

---

## 📤 Post-Training Checklist (Before Pushing)

- [ ] Training completed or reached desired checkpoint
- [ ] `model/last_model.zip` is updated
- [ ] Checkpoint files saved (optional but recommended)
- [ ] Committed changes: `git add model/ && git commit -m "..." `
- [ ] Pushed to GitHub: `git push origin main`

---

## 🚨 Important Notes

### 1. **Always Pull Before Training**
   - Always run `git pull` before starting your training session
   - This ensures you're starting from the latest model

### 2. **Model Files Are Large**
   - Model files (`.zip`) can be 10-50 MB each
   - GitHub has file size limits (100 MB per file)
   - If you hit limits, consider using Git LFS or only committing `last_model.zip`

### 3. **Checkpoint Safety**
   - The script saves `last_model.zip` automatically
   - Checkpoints (`checkpoint_*_steps.zip`) are backups
   - You can commit just `last_model.zip` if needed

### 4. **Training Interruption**
   - If training is interrupted (crash, power loss), the last checkpoint is saved
   - Just run `python train.py` again to resume
   - No need to commit until you're ready to hand off

### 5. **Multiple Friends Training Simultaneously**
   - ⚠️ **Don't train at the same time!**
   - Only one person should train at a time
   - Always pull before training to avoid conflicts

---

## 🔍 Verifying Training Progress

### Check Current Timestep

When you load a model, the training output will show:
```
Loading existing model from ./model/last_model.zip
Resuming training...
Starting training...
Training for 1,000,000 timesteps...
```

The model remembers its timestep internally. To see exact progress, check TensorBoard:
```bash
tensorboard --logdir ./tensorboard_logs
```

### Check Model Files

```bash
# List all model files
ls model/

# Should show:
# - last_model.zip (latest checkpoint)
# - checkpoint_10000_steps.zip
# - checkpoint_20000_steps.zip
# - ... (more checkpoints)
```

---

## 🐛 Troubleshooting

### Problem: "Training starts from 0 instead of resuming"
- **Solution**: Check that `model/last_model.zip` exists after `git pull`
- Verify file: `Test-Path model/last_model.zip` (should return `True`)

### Problem: "Git conflict when pulling"
- **Solution**: Someone else pushed while you were training
- Resolve: `git pull --rebase` or `git pull` then resolve conflicts manually

### Problem: "Model file too large for GitHub"
- **Solution**: Only commit `last_model.zip`, not all checkpoints
- Or use Git LFS: `git lfs track "model/*.zip"`

### Problem: "Friend's model doesn't load"
- **Solution**: Make sure they committed `model/last_model.zip` (not just code)
- Check: `git log --oneline` to see their commit
- Verify: `git show HEAD:model/last_model.zip` (should show file exists)

---

## 📊 Training Timeline Example

```
Time    Friend    Action                          Timesteps
─────────────────────────────────────────────────────────────
Day 1   Friend A  Trains from scratch             0 → 1,000,000
Day 1   Friend A  Commits & pushes                 ✅
Day 2   Friend B  Pulls, trains                   1,000,000 → 2,000,000
Day 2   Friend B  Commits & pushes                 ✅
Day 3   Friend C  Pulls, trains                   2,000,000 → 3,000,000
Day 3   Friend C  Commits & pushes                 ✅
Day 4   Friend A  Pulls, trains again             3,000,000 → 4,000,000
...
```

---

## 🎯 Best Practices

1. **Communicate**: Let others know when you're training
2. **Commit regularly**: Don't wait too long between commits
3. **Test before pushing**: Run `test_training.py` if unsure
4. **Use clear commit messages**: Include timestep count
5. **Keep checkpoints**: They're backups if something goes wrong

---

Happy collaborative training! 🎾🤖

