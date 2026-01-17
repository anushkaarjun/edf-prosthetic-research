#!/bin/bash
# Simple script to notify when training finishes and automatically load the model
# Run this in a separate terminal: ./scripts/notify_when_done.sh

cd "$(dirname "$0")/.."

echo "🔔 Notification monitor started..."
echo "Watching for training completion..."
echo "Will automatically load improved model once training completes."
echo ""

# Wait for process to finish
while ps aux | grep "train_improved_model" | grep -v grep > /dev/null; do
    sleep 10
done

echo "✅ Training completed!"

# Wait for results file and model to be saved
sleep 5

# Send notification with accuracy
if [ -f "results/improved_model_results.txt" ] && [ -s "results/improved_model_results.txt" ]; then
    ACC=$(grep -i "validation accuracy\|val acc.*:" results/improved_model_results.txt | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    if [ -n "$ACC" ]; then
        ACC_PCT=$(echo "$ACC * 100" | bc -l | xargs printf "%.2f")
        echo "📊 Final Accuracy: ${ACC_PCT}%"
        
        # Compare with baseline
        BASELINE=51.94
        IMPROVEMENT=$(echo "$ACC_PCT - $BASELINE" | bc -l | xargs printf "%.2f")
        
        if (( $(echo "$ACC_PCT > $BASELINE" | bc -l) )); then
            osascript -e "display notification \"ImprovedEEGNet: ${ACC_PCT}% (+${IMPROVEMENT}% improvement)\" with title \"Training Complete! 🎉\" sound name \"Glass\""
            echo "✅ Improved by ${IMPROVEMENT}% over CNN-LSTM baseline (51.94%)"
        else
            osascript -e "display notification \"ImprovedEEGNet: ${ACC_PCT}% (${IMPROVEMENT}% vs CNN-LSTM)\" with title \"Training Complete\" sound name \"Glass\""
            echo "📊 Accuracy: ${ACC_PCT}% (${IMPROVEMENT}% vs CNN-LSTM baseline)"
        fi
    else
        osascript -e "display notification \"Check results/improved_model_results.txt\" with title \"Training Complete\" sound name \"Glass\""
    fi
else
    osascript -e "display notification \"Training finished\" with title \"EEG Model Training\" sound name \"Glass\""
fi

# Check if model file exists and compare with baseline
MODEL_FILE="models/best_improved_eegnet.pth"
BASELINE_ACC=51.94  # CNN-LSTM current best

if [ -f "$MODEL_FILE" ] && [ -n "$ACC_PCT" ]; then
    echo ""
    echo "📦 Model file found: $MODEL_FILE"
    echo "📊 Comparing accuracy with baseline models..."
    echo ""
    
    # Compare with baseline (CNN-LSTM at 51.94%)
    if (( $(echo "$ACC_PCT > $BASELINE_ACC" | bc -l) )); then
        echo "✅ Improved model accuracy (${ACC_PCT}%) is HIGHER than baseline (${BASELINE_ACC}%)"
        echo "🚀 Loading improved model into API server..."
        echo ""
        
        # Check if API server is running
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            # Load the model
            python3 scripts/load_improved_model.py
            
            if [ $? -eq 0 ]; then
                echo ""
                osascript -e "display notification \"Improved model (${ACC_PCT}%) loaded into API server!\" with title \"Better Model Loaded! 🎉\" sound name \"Glass\""
                echo "✅ Improved model is now active in the API server!"
                echo "🎉 You can now use it in the React simulator!"
            else
                echo ""
                echo "⚠️  Failed to load model. Check API server status."
                osascript -e "display notification \"Model loading failed. Check API server.\" with title \"Model Load Error\" sound name \"Basso\""
            fi
        else
            echo ""
            echo "⚠️  API server is not running."
            echo "💡 Start API server first: make api-server"
            echo "   Then load model: make load-improved"
            osascript -e "display notification \"API server not running. Start it to load model.\" with title \"API Server Required\" sound name \"Basso\""
        fi
    else
        DIFF=$(echo "$BASELINE_ACC - $ACC_PCT" | bc -l | xargs printf "%.2f")
        echo "⚠️  Improved model accuracy (${ACC_PCT}%) is LOWER than baseline (${BASELINE_ACC}%)"
        echo "📉 Difference: -${DIFF}%"
        echo ""
        echo "💡 Not loading improved model - keeping CNN-LSTM (${BASELINE_ACC}%) as active model"
        echo "   The improved model underperformed and will not be automatically loaded."
        osascript -e "display notification \"Improved model (${ACC_PCT}%) is worse than CNN-LSTM (${BASELINE_ACC}%). Not loading.\" with title \"Lower Accuracy Detected\" sound name \"Basso\""
        echo ""
        echo "✅ Keeping current best model (CNN-LSTM at ${BASELINE_ACC}%) active"
    fi
elif [ -f "$MODEL_FILE" ]; then
    echo ""
    echo "📦 Model file found: $MODEL_FILE"
    echo "⚠️  Could not extract accuracy from results"
    echo "💡 Model file exists but accuracy not found. Check results manually."
    echo "   Will not auto-load without accuracy verification."
elif [ -n "$ACC_PCT" ]; then
    echo ""
    echo "⚠️  Model file not found: $MODEL_FILE"
    echo "💡 Model may still be saving. Check in a few seconds."
    osascript -e "display notification \"Model file not found yet. Check manually.\" with title \"Model File Missing\" sound name \"Basso\""
else
    echo ""
    echo "⚠️  Could not determine accuracy or find model file"
    echo "💡 Check training results manually: tail -50 results/improved_model_results.txt"
fi

echo ""
echo "✅ Complete!"
