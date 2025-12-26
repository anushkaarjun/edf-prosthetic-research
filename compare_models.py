#!/usr/bin/env python3
"""
Compare CSP+SVM and Neural Network models side-by-side.
"""
import os
import sys
import subprocess

def run_csp_svm(data_path, max_subjects=5):
    """Run CSP+SVM model and extract accuracy."""
    print("="*70)
    print("RUNNING CSP+SVM MODEL")
    print("="*70)
    
    cmd = [
        sys.executable, 'run_csp_svm.py',
        '--data-path', data_path,
        '--max-subjects', str(max_subjects),
        '--no-clean'  # Disable cleaning to avoid digitization errors
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Extract accuracy from output
    accuracy = None
    for line in output.split('\n'):
        if 'Average Test Accuracy' in line:
            try:
                # Extract percentage
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith('%') and '(' in parts[i-1]:
                        accuracy = float(parts[i-1].split('(')[1])
                        break
            except:
                pass
    
    return accuracy, output

def run_neural_network(data_path, max_subjects=5):
    """Run Neural Network model and extract accuracy."""
    print("="*70)
    print("RUNNING NEURAL NETWORK MODEL")
    print("="*70)
    
    cmd = [
        sys.executable, 'train_model.py',
        '--data-path', data_path,
        '--max-subjects', str(max_subjects),
        '--freeze-after', '30',
        '--epochs', '50'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        
        # Extract accuracy from output
        accuracy = None
        for line in output.split('\n'):
            if 'Average Test Accuracy' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('%') and '(' in parts[i-1]:
                            accuracy = float(parts[i-1].split('(')[1])
                            break
                except:
                    pass
        
        return accuracy, output
    except subprocess.TimeoutExpired:
        return None, "Training timed out (neural network takes longer)"
    except Exception as e:
        return None, f"Error: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <data_path> [max_subjects]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_subjects = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Data path: {data_path}")
    print(f"Max subjects: {max_subjects}")
    print()
    
    # Run CSP+SVM
    csp_acc, csp_output = run_csp_svm(data_path, max_subjects)
    
    print("\n")
    
    # Run Neural Network
    nn_acc, nn_output = run_neural_network(data_path, max_subjects)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print()
    print(f"CSP + SVM Model:")
    if csp_acc is not None:
        print(f"  Average Test Accuracy: {csp_acc:.2f}%")
    else:
        print(f"  Could not extract accuracy")
    print()
    print(f"Neural Network Model:")
    if nn_acc is not None:
        print(f"  Average Test Accuracy: {nn_acc:.2f}%")
    else:
        print(f"  {nn_output}")
    print()
    
    if csp_acc is not None and nn_acc is not None:
        if csp_acc > nn_acc:
            print(f"Winner: CSP + SVM ({csp_acc:.2f}% vs {nn_acc:.2f}%)")
            print(f"  Advantage: {csp_acc - nn_acc:.2f}%")
        elif nn_acc > csp_acc:
            print(f"Winner: Neural Network ({nn_acc:.2f}% vs {csp_acc:.2f}%)")
            print(f"  Advantage: {nn_acc - csp_acc:.2f}%")
        else:
            print(f"Tie: Both models achieved {csp_acc:.2f}%")
        
        print()
        if max(csp_acc, nn_acc) >= 80:
            print("✓ TARGET ACHIEVED (80%+)!")
        else:
            gap = 80 - max(csp_acc, nn_acc)
            print(f"⚠ Below target (80%). Gap: {gap:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()

