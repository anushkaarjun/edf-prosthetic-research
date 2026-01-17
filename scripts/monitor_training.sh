#!/bin/bash
# Monitor training progress and send notification when complete

RESULTS_FILE="../results/improved_model_results.txt"
PROCESS_NAME="train_improved_model"
CHECK_INTERVAL=30  # Check every 30 seconds
MAX_WAIT=3600      # Maximum wait time (1 hour)

echo "🔍 Monitoring training progress..."
echo "Results file: $RESULTS_FILE"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Function to send macOS notification
send_notification() {
    local title="$1"
    local message="$2"
    osascript -e "display notification \"$message\" with title \"$title\" sound name \"Glass\""
    echo "✅ Notification sent: $title - $message"
}

# Function to extract accuracy from results
extract_accuracy() {
    if [ -f "$RESULTS_FILE" ]; then
        # Look for validation accuracy in results
        grep -i "validation accuracy\|val acc\|results:" "$RESULTS_FILE" | tail -5 | head -1
    fi
}

# Check if training is running
check_process() {
    ps aux | grep "$PROCESS_NAME" | grep -v grep | grep -v "monitor_training" > /dev/null
    return $?
}

# Wait for training to start (if not already running)
if ! check_process; then
    echo "⚠️  Training process not found. Waiting for it to start..."
    while ! check_process; do
        sleep 5
    done
    echo "✅ Training process detected!"
fi

# Monitor training
START_TIME=$(date +%s)
LAST_SIZE=0
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ELAPSED_MIN=$((ELAPSED / 60))
    
    # Check if process is still running
    if ! check_process; then
        echo ""
        echo "🎉 Training process completed!"
        echo "⏱️  Total time: ${ELAPSED_MIN} minutes"
        
        # Wait a moment for file to be written
        sleep 2
        
        # Check results file
        if [ -f "$RESULTS_FILE" ] && [ -s "$RESULTS_FILE" ]; then
            echo ""
            echo "📊 Training Results:"
            echo "===================="
            tail -30 "$RESULTS_FILE"
            
            # Extract final accuracy
            FINAL_ACC=$(grep -i "validation accuracy\|val acc.*:" "$RESULTS_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
            
            if [ -n "$FINAL_ACC" ]; then
                ACC_PERCENT=$(echo "$FINAL_ACC * 100" | bc -l | xargs printf "%.2f")
                echo ""
                echo "🎯 Final Validation Accuracy: ${ACC_PERCENT}%"
                
                # Compare with baseline
                BASELINE=51.94
                IMPROVEMENT=$(echo "$ACC_PERCENT - $BASELINE" | bc -l | xargs printf "%.2f")
                
                if (( $(echo "$ACC_PERCENT > $BASELINE" | bc -l) )); then
                    send_notification "Training Complete! 🎉" "ImprovedEEGNet: ${ACC_PERCENT}% (${IMPROVEMENT}% improvement over CNN-LSTM)"
                    echo "✅ Improved by ${IMPROVEMENT}% over CNN-LSTM baseline (51.94%)"
                else
                    send_notification "Training Complete" "ImprovedEEGNet: ${ACC_PERCENT}% (${IMPROVEMENT}% vs CNN-LSTM)"
                    echo "📊 Accuracy: ${ACC_PERCENT}% (${IMPROVEMENT}% vs CNN-LSTM baseline)"
                fi
            else
                send_notification "Training Complete" "Check results/improved_model_results.txt for details"
            fi
        else
            send_notification "Training Complete" "Results file not found or empty"
            echo "⚠️  Results file not found or empty"
        fi
        
        break
    fi
    
    # Check if results file has new content
    if [ -f "$RESULTS_FILE" ]; then
        CURRENT_SIZE=$(stat -f%z "$RESULTS_FILE" 2>/dev/null || stat -c%s "$RESULTS_FILE" 2>/dev/null || echo 0)
        if [ "$CURRENT_SIZE" -gt "$LAST_SIZE" ]; then
            echo "[${ELAPSED_MIN}m] Training in progress... (file size: ${CURRENT_SIZE} bytes)"
            LAST_SIZE=$CURRENT_SIZE
        fi
    fi
    
    # Check timeout
    if [ $ELAPSED -gt $MAX_WAIT ]; then
        send_notification "Training Timeout" "Training exceeded ${MAX_WAIT}s. Check manually."
        echo "⏱️  Maximum wait time exceeded"
        break
    fi
    
    # Show progress every 2 minutes
    if [ $((ITERATION % 4)) -eq 0 ]; then
        echo "[${ELAPSED_MIN}m] Still training... (checking every ${CHECK_INTERVAL}s)"
    fi
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "✅ Monitoring complete!"
