# Accuracy audit: 3-class CNN-LSTM (~98% test)

Answers to the checklist, based on the current codebase (`train_model_3class.py`, `data_loader_edf.py`).

---

## Fixes applied (post-audit)

- **Train accuracy** is now computed, printed, and saved in `eval_results.json` as `train_accuracy` so you can monitor train–test gap.
- **Small test set warning:** if there are fewer than 5 test subjects, the training script prints a clear warning that test accuracy has high variance.
- **Test confusion matrix** and **n_test_subjects / n_test_trials** are saved in `eval_results.json` for auditability.
- **RANDOM_STATE** env var is supported (e.g. `RANDOM_STATE=123 python3 train_model_3class.py`) so you can run multiple splits for robustness.
- **Sanity check script:** `python3 sanity_check_labels.py` trains with **shuffled labels**; test accuracy should drop to ~33%. If it stays > 50%, the script exits with failure (possible leakage or bug).

---

## 1. Data splitting & leakage

### Are test subjects completely unseen during training?
**Yes.** Split is **subject-independent**:
- `train_test_split(unique_subjects, test_size=0.2)` → 80% subjects for train, 20% for test.
- `mask_train = np.isin(subject_ids, train_subjects)` / `mask_test = np.isin(subject_ids, test_subjects)`.
- No subject appears in both train and test.

### Were trials split before or after chunking?
**Split is by subject; “chunking” is one segment per event.**  
- EDF loader: each event produces **one** segment (one row in `X`). No sliding window; no multiple chunks per trial.
- Split is applied **after** loading: by `subject_ids`. So no segment from a test subject is ever in the training set. **No chunk leakage.**

### Was oversampling done before or after splitting?
**After.**  
- Subject split first → `X_train_pool`, `X_test` (test is untouched).
- Oversampling only on `X_train_pool` / `y_train_pool` (indices from training subjects only).
- Test set is never oversampled or duplicated. **No leakage from oversampling.**

---

## 2. Dataset characteristics

### How many subjects?
- **Up to 20** (`max_subjects=20` in `load_validation_from_edf_dir`).
- Test = 20% → **~4 held-out subjects** (with default 80/20 split).
- So 98% is on a **small number of test subjects**; a few errors change the percentage a lot.

### How many total trials per class?
- Capped at **2500** total trials (`max_total_trials=2500`), **80** per run (`max_trials_per_run=80`).
- Script prints class counts, e.g. `3-class: {0: …, 1: …, 2: …}`. With 20 subjects and 2500 trials, hundreds per class is typical but depends on DATA_DIR.

### Is the dataset publicly benchmarked?
- EDF layout (S001, S002, …, R03–R14, .edf.event with T0–T4) matches **BCI Competition IV 2a** style.
- Literature on BCI IV 2a (and similar 3-class motor imagery) often reports **~70–85%** for subject-independent or cross-subject settings. **98% is above typical published results** for strict subject-independent evaluation, so it’s worth treating as “optimistic until verified” (e.g. same lab/setup, or lucky subject split).

---

## 3. Evaluation method

### Test vs validation accuracy?
- **Validation accuracy** (100% in your run): 25% of **training subjects’** trials, used for early stopping. Can be optimistic.
- **Test accuracy** (~98.9%): held-out **subjects** only. This is the number that counts. The script prints both and writes `test_accuracy` to `eval_results.json`.

### Cross-validation?
- **Single 80/20 subject split** (fixed `random_state=42`). Not LOSO and not k-fold over subjects. So one split only; result can vary with different splits.

### Confusion matrices?
- **Yes.** Script prints confusion matrices for both validation and test (rows = actual, cols = predicted). Worth checking that no single class dominates predictions.

---

## 4. Model behavior

### Training vs test accuracy?
- From your `eval_results.json`: val 100%, test ~98.9%. Train accuracy is not stored; script doesn’t print it. **Worth logging train accuracy** to check for large train–test gap (e.g. 100% train + 98% test with small data can still be plausible but is the main thing to monitor).

### Does performance drop with ablations?
- Not run in the repo. **Suggested checks:** retrain without class weights, without oversampling, or without bandpass (8–30 Hz in loader). If accuracy drops a lot, the model is relying on those.

---

## 5. Robustness checks (not in repo; recommended)

- **Shuffle labels:** Retrain with randomly shuffled `y`. Accuracy should drop to ~33% (3-class). If it stays high → bug.
- **Different seed:** Retrain with another `random_state` (and different subject split). See how much test accuracy varies.
- **New subject:** Add a new subject to DATA_DIR, put them only in test (or do LOSO). Report accuracy on that subject.

---

## Red-flag summary

| Check | Status |
|-------|--------|
| Chunk leakage (same trial in train and test) | **None** – one segment per event; split by subject. |
| Same subject in train and test | **No** – subject-independent split. |
| Oversampling before split | **No** – oversampling only on train pool. |
| Tiny dataset | **Borderline** – up to 20 subjects, ~4 test subjects; 2500 trials total. |
| Only reporting best fold | **N/A** – single split, not k-fold. |

So the **main inflation risks from the checklist are not present** in the code: no chunk leakage, no subject overlap, oversampling is after split. The 98% is still **subject-independent test accuracy**, but with a **small number of test subjects** and **one split**, so it should be validated with more subjects, LOSO, or different seeds before claiming robustness.

---

## The killer question

**“Are any EEG segments from the same subject or same trial present in both training and test sets?”**  

**Answer: No.**  
- Each segment is tagged with `subject_id`. Train/test split is by subject, so no test subject’s data is in the training set.  
- Each event yields one segment; there’s no sliding window, so no “same trial” in both sets.  

So the current setup does **not** have the usual subject/trial leakage that inflates EEG accuracy. The remaining reasons to be cautious are: **small number of test subjects**, **single split**, and **98% being above many published BCI IV 2a–style results** for subject-independent evaluation.
