# Training Notification Setup

## 🔔 Automatic Notifications

I've set up automatic notifications that will alert you when training completes.

### Option 1: Background Monitor (Running Now)

A background process is monitoring your training and will send a macOS notification when it finishes.

**Status**: ✅ Active and monitoring

**What it does:**
- Monitors the training process
- Waits for completion
- Extracts final accuracy from results
- Sends macOS notification with results
- Compares with baseline (CNN-LSTM at 51.94%)

### Option 2: Manual Monitor Script

Run this in a separate terminal:

```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
./scripts/monitor_training.sh
```

**Features:**
- Checks every 30 seconds
- Shows progress updates
- Extracts and displays final accuracy
- Sends notification with comparison to baseline
- Maximum wait: 1 hour

### Option 3: Simple Notification Script

For a simpler approach:

```bash
./scripts/notify_when_done.sh
```

This will:
- Wait for training to finish
- Send a notification
- Display final accuracy

## 📊 What You'll See

When training completes, you'll receive a notification like:

**If improved:**
```
🎉 Training Complete!
ImprovedEEGNet: 58.50% (+6.56% improvement over CNN-LSTM)
```

**If similar:**
```
Training Complete
ImprovedEEGNet: 52.30% (+0.36% vs CNN-LSTM)
```

## 🔍 Manual Check

You can also check manually:

```bash
# Check if training is still running
ps aux | grep train_improved_model | grep -v grep

# View latest results
tail -50 results/improved_model_results.txt

# Quick status check
./scripts/check_training.sh
```

## 📝 Notification Details

The notification will include:
- ✅ Training completion status
- 📊 Final validation accuracy
- 📈 Comparison with CNN-LSTM baseline (51.94%)
- 🎯 Improvement percentage (if applicable)

## ⚙️ Customization

To change notification settings, edit `scripts/monitor_training.sh`:
- `CHECK_INTERVAL`: How often to check (default: 30s)
- `MAX_WAIT`: Maximum wait time (default: 1 hour)

---

*Notifications are now active and monitoring your training!*
