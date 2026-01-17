#!/bin/bash
# Wrapper script to ensure auto-load runs when training completes
# This can be called after training starts to set up the monitor

cd "$(dirname "$0")/.."

echo "🔔 Setting up auto-load monitor..."
echo ""

# Check if training is still running
if ps aux | grep "train_improved_model" | grep -v grep > /dev/null; then
    echo "✅ Training is in progress"
    echo "🔍 Monitor will wait for completion..."
    echo ""
    
    # Run the notify script in background
    nohup bash scripts/notify_when_done.sh > /tmp/auto_load_monitor.log 2>&1 &
    MONITOR_PID=$!
    
    echo "✅ Auto-load monitor started (PID: $MONITOR_PID)"
    echo "📝 Logs: /tmp/auto_load_monitor.log"
    echo ""
    echo "The monitor will:"
    echo "  ✓ Wait for training to complete"
    echo "  ✓ Extract accuracy from results"
    echo "  ✓ Send macOS notification"
    echo "  ✓ Automatically load model into API server (if running)"
    echo ""
    echo "To check monitor status:"
    echo "  tail -f /tmp/auto_load_monitor.log"
    echo ""
    echo "To stop monitor:"
    echo "  kill $MONITOR_PID"
else
    echo "⚠️  Training is not running"
    echo "💡 Start training first, then run this script"
    echo ""
    echo "Or run notify script directly:"
    echo "  ./scripts/notify_when_done.sh"
fi
