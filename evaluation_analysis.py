"""
evaluation_analysis.py  —  Part F: Evaluation and Analysis
IT9204 | Student 202507410 Faiza Malik

Professor's guidance implemented:
  - Model trained on a 5000-essay SUBSET (not full dataset).
  - Evaluation also runs on subset for consistency.
  - To evaluate on full dataset: change SAMPLE_SIZE = len(df).

Generates:
  error_analysis.png       → 4-panel chart for report
  failure_cases.csv        → worst 10 predictions for error discussion
  evaluation_results.json  → all metrics for report tables

Requires:
  - models/model.pkl
  - models/feature_names.json
  - models/score_meta.json
  - data/asap_2_0.csv

Run: python evaluation_analysis.py
"""

import re, json, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nltk, textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model and dataset
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model...")
model         = joblib.load("models/model.pkl")
feature_names = json.load(open("models/feature_names.json"))
score_meta    = json.load(open("models/score_meta.json"))

print(f"Model: {score_meta.get('model_name', 'Random Forest')}")
print(f"Training RMSE from notebook: {score_meta.get('rmse', 'N/A')}")
print(f"Training R² from notebook  : {score_meta.get('r2', 'N/A')}")

print("\nLoading ASAP 2.0 dataset...")
df = pd.read_csv("data/asap_train.csv")

# Correct column names from GitHub repository
text_col  = "full_text"
score_col = "score"   # actual column name in ASAP 2.0

if text_col not in df.columns:
    text_col = next((c for c in df.columns if "text" in c.lower()), None)
if score_col not in df.columns:
    score_col = next(
        (c for c in df.columns
         if "score" in c.lower() and "id" not in c.lower()), None
    )

df = df.rename(columns={text_col: "essay_text", score_col: "score"})
df["essay_text"] = df["essay_text"].astype(str)
df["score"]      = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["essay_text", "score"]).reset_index(drop=True)

# Normalise to 0-10 per prompt
# ASAP 2.0 score is uniformly 1-6. Normalise to 0-10.
df["score"] = (df["score"] - 1) / 5 * 10

print(f"Dataset: {len(df)} essays | Score: {df['score'].min():.1f}–{df['score'].max():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Professor's guidance: use a subset for experimentation
# Change EVAL_SIZE to len(df) when ready for full evaluation
# ─────────────────────────────────────────────────────────────────────────────
EVAL_SIZE = min(2000, len(df))

print(f"\nEvaluating on {EVAL_SIZE} essays (subset — professor's guidance).")
print("To evaluate on full dataset: change EVAL_SIZE = len(df)")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — must match training notebook exactly
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    sentences  = sent_tokenize(text)
    words      = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    word_count = len(words)
    sent_lens  = [len(word_tokenize(s)) for s in sentences]
    try:
        fe = textstat.flesch_reading_ease(text)
        fk = textstat.flesch_kincaid_grade(text)
        gf = textstat.gunning_fog(text)
    except:
        fe, fk, gf = 50.0, 8.0, 10.0
    return {
        "word_count"       : word_count,
        "char_count"       : len(text),
        "sentence_count"   : max(len(sentences), 1),
        "unique_word_ratio": len(set(words)) / max(word_count, 1),
        "avg_word_length"  : float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_len" : float(np.mean(sent_lens)) if sent_lens else 0.0,
        "long_sent_ratio"  : sum(1 for l in sent_lens if l > 20) / max(len(sent_lens), 1),
        "flesch_ease"      : fe,
        "flesch_grade"     : fk,
        "gunning_fog"      : gf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extract features and predict
# ─────────────────────────────────────────────────────────────────────────────
print("\nExtracting features...")
sample    = df.sample(n=EVAL_SIZE, random_state=42).reset_index(drop=True)
feat_list = [extract_features(t) for t in sample["essay_text"]]
feat_df   = pd.DataFrame(feat_list)
y_true    = sample["score"].values
y_pred    = np.clip(model.predict(feat_df), 0.0, 10.0)
residuals = y_pred - y_true


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae  = float(mean_absolute_error(y_true, y_pred))
r2   = float(r2_score(y_true, y_pred))

print(f"\n{'='*60}")
print(f"EVALUATION RESULTS  ({EVAL_SIZE} essays)")
print(f"{'='*60}")
print(f"  RMSE : {rmse:.4f}  (avg error in score units, 0-10 scale)")
print(f"  MAE  : {mae:.4f}  (mean absolute error)")
print(f"  R²   : {r2:.4f}  (1.0 = perfect prediction)")
print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-band RMSE — shows where model struggles
# ─────────────────────────────────────────────────────────────────────────────
bands = [
    (0, 2,  "1-2 Emerging"),
    (2, 4,  "3-4 Beginning"),
    (4, 6,  "5-6 Developing"),
    (6, 8,  "7-8 Proficient"),
    (8, 10, "9-10 Exemplary"),
]
band_results = {}
for lo, hi, label in bands:
    mask = (y_true >= lo) & (y_true <= hi)
    if mask.sum() > 5:
        b_rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        band_results[label] = {"count": int(mask.sum()), "rmse": round(b_rmse, 4)}
        print(f"  {label:<22} n={mask.sum():>5}  RMSE={b_rmse:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Error distribution
# ─────────────────────────────────────────────────────────────────────────────
close = int((np.abs(residuals) <= 1.5).sum())
over  = int((residuals >  1.5).sum())
under = int((residuals < -1.5).sum())
print(f"\nWithin ±1.5 points   : {close} ({100*close/EVAL_SIZE:.0f}%)")
print(f"Over-predicted >1.5  : {over}  ({100*over/EVAL_SIZE:.0f}%)")
print(f"Under-predicted >1.5 : {under} ({100*under/EVAL_SIZE:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Failure cases — worst 10 predictions
# ─────────────────────────────────────────────────────────────────────────────
sample["predicted"]     = y_pred
sample["actual"]        = y_true
sample["abs_error"]     = np.abs(residuals)
sample["residual"]      = residuals
sample["word_count"]    = feat_df["word_count"].values
sample["essay_preview"] = sample["essay_text"].str[:200]

failures = sample.nlargest(10, "abs_error")[
    ["essay_preview", "actual", "predicted", "abs_error",
     "residual", "word_count"]
]
failures.to_csv("failure_cases.csv", index=False)
print(f"\nTop 5 worst predictions:")
for _, row in failures.head(5).iterrows():
    direction = "over" if row["residual"] > 0 else "under"
    print(
        f"  Actual:{row['actual']:.1f}  Predicted:{row['predicted']:.1f}  "
        f"Error:{row['abs_error']:.2f}  ({direction})  "
        f"Words:{int(row['word_count'])}"
    )
print("Saved → failure_cases.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
try:
    importances = pd.Series(
        model.named_steps["model"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
except Exception:
    importances = pd.Series(
        [1/len(feature_names)] * len(feature_names),
        index=feature_names
    )
print(f"\nTop 5 features:")
for name, val in importances.head(5).items():
    print(f"  {name:<22} {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4-panel chart
# ─────────────────────────────────────────────────────────────────────────────
PURPLE = "#534AB7"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#BA7517"

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color=PURPLE)
ax1.plot([0, 10], [0, 10], "r--", lw=1.5, label="Perfect")
ax1.set_xlabel("Actual score (0-10)")
ax1.set_ylabel("Predicted score")
ax1.set_title(f"Predicted vs Actual  (R²={r2:.3f})")
ax1.text(0.05, 0.92, f"RMSE={rmse:.3f}", transform=ax1.transAxes,
         fontsize=10, color=PURPLE)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals, bins=40, color=TEAL, edgecolor="white", linewidth=0.4)
ax2.axvline(0,    color="red",   lw=1.5, label="Zero error")
ax2.axvline( 1.5, color=AMBER, lw=1, linestyle="--", label="±1.5 threshold")
ax2.axvline(-1.5, color=AMBER, lw=1, linestyle="--")
ax2.set_xlabel("Residual (predicted − actual)")
ax2.set_ylabel("Count")
ax2.set_title(f"Residuals  (MAE={mae:.3f})")
ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
b_labels = list(band_results.keys())
b_rmses  = [band_results[b]["rmse"] for b in b_labels]
colors3  = [TEAL if r < rmse else CORAL for r in b_rmses]
bars = ax3.bar(range(len(b_labels)), b_rmses, color=colors3, edgecolor="white")
ax3.axhline(rmse, color="gray", lw=1.5, linestyle="--",
            label=f"Overall RMSE ({rmse:.3f})")
ax3.set_xticks(range(len(b_labels)))
ax3.set_xticklabels([b.split(" ")[0] for b in b_labels], fontsize=9)
ax3.set_ylabel("RMSE")
ax3.set_title("RMSE by score band")
ax3.legend(fontsize=9)
for bar, val in zip(bars, b_rmses):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax4 = fig.add_subplot(gs[1, 1])
top = importances.head(8)
ax4.barh(range(len(top)), top.values[::-1], color=PURPLE, edgecolor="white")
ax4.set_yticks(range(len(top)))
ax4.set_yticklabels(top.index[::-1], fontsize=9)
ax4.set_xlabel("Importance score")
ax4.set_title("Feature Importance (Random Forest)")

fig.suptitle(
    "IT9204 Essay Grading — Model Evaluation\n"
    f"Student: 202507410 | Dataset: ASAP 2.0 | "
    f"Trained on 5000-essay subset",
    fontsize=11, fontweight="bold", y=0.99
)
plt.savefig("error_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved → error_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Save results JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "model"           : score_meta.get("model_name", "Random Forest"),
    "eval_sample_size": EVAL_SIZE,
    "training_subset" : score_meta.get("training_subset", 5000),
    "dataset"         : "ASAP 2.0 — https://github.com/scrosseye/ASAP_2.0",
    "metrics"     : {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)},
    "band_results": band_results,
    "error_dist"  : {
        "within_1_5"    : close,
        "over_pred_1_5" : over,
        "under_pred_1_5": under,
        "pct_within_1_5": round(100 * close / EVAL_SIZE, 1),
    },
    "top_features": {k: round(float(v), 4) for k, v in importances.head(5).items()},
    "failure_sample": [
        {
            "actual"   : round(float(r["actual"]), 2),
            "predicted": round(float(r["predicted"]), 2),
            "error"    : round(float(r["abs_error"]), 2),
            "words"    : int(r["word_count"]),
            "preview"  : r["essay_preview"][:150],
        }
        for _, r in failures.head(3).iterrows()
    ],
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved → evaluation_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# Print report summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
SUMMARY — copy these values into your report (Section 8)
{'='*60}
Model          : {score_meta.get('model_name', 'Random Forest')}
Training size  : 5000 essays (subset per professor's guidance)
Evaluation on  : {EVAL_SIZE} essays

METRICS:
  RMSE = {rmse:.4f}  → predictions are ±{rmse:.2f} score pts on avg
  MAE  = {mae:.4f}
  R²   = {r2:.4f}  → model explains {r2*100:.0f}% of score variance

ERROR DISTRIBUTION:
  Within ±1.5 pts : {close} ({100*close/EVAL_SIZE:.0f}% of predictions)
  Over-predicted  : {over} ({100*over/EVAL_SIZE:.0f}%)
  Under-predicted : {under} ({100*under/EVAL_SIZE:.0f}%)

TOP 3 FEATURES:
  1. {list(importances.index)[0]}
  2. {list(importances.index)[1]}
  3. {list(importances.index)[2]}

ARCHITECTURE NOTE FOR REPORT:
  Scoring  → Azure ML Random Forest (deterministic, reproducible)
  Feedback → RAG retrieval from rubric docs + source texts
  LLM      → Explanation only, never scoring

FILES GENERATED:
  error_analysis.png       → insert in report Section 8
  failure_cases.csv        → discuss in error analysis
  evaluation_results.json  → source for all report metrics
{'='*60}
""")
