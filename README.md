# Carvana auction decisions

Estimate the risk that a used car is a bad buy, then use that risk to decide whether to bid. Each root notebook runs both loss variants of one model family.

| Notebook | L2 version | Log-loss version |
| --- | --- | --- |
| [Sparse linear models](glassbox_linear.ipynb) | L1-regularized linear regression | L1-regularized logistic regression |
| [Random forest](random_forest.ipynb) | Squared-error regression trees | Classification trees with log-loss splits |
| [CatBoost](catboost_trees.ipynb) | RMSE regression | Binary log-loss classification |

## Run

Place the labelled competition file at `data/DontGetKicked/training.csv`. From the project root, install the requirements in a Python environment and run the development workflow.

```bash
python -m pip install -r requirements.txt
python -m carvana all
```

For interactive use, open the three notebooks in Jupyter and run each from top to bottom. The following commands are equivalent and can also be run individually.

```bash
python -m carvana glassbox
python -m carvana random_forest
python -m carvana catboost
python -m carvana validate
python -m carvana explain
```

`validate` needs all three families' saved validation scores and settings. It ranks the six models by AUC, checks the winner's calibration and selects a bidding threshold. `explain` produces global and local TreeSHAP explanations on validation cars. To reproduce the final internal test after the policy is frozen, run `python -m carvana holdout`. The development commands skip this step.

## Shared data and validation

| Chronological split | Purchase dates | Cars |
| --- | --- | ---: |
| Training | Through June 8, 2010 | 51,060 after filtering |
| Validation | June 9 – September 29, 2010 | 11,022 |
| Internal test | September 30 – December 30, 2010 | 10,770 |

- All families use the same 15 pre-bid fields: vehicle age, odometer, make, model, size, transmission, auction, purchase state, online-sale flag, trim, submodel and four acquisition-time reference prices.
- For fitting only, remove rows with the exact four-quote pattern `[0, 1, 0, 0]` (131 cars in the full training period). For each fit, replace nonpositive individual quotes with missing values. Fill missing numeric values using training medians. Mark missing categories and group vehicle models seen fewer than 20 times in training. Keep every car in each evaluation period.
- Inside training, the linear models use a whole-date 80/20 split for regularization. The tree models screen candidates over three expanding time folds and compare shortlisted settings on validation. Each notebook fits its encoders on the relevant training rows. All six final fits use the same eligible training cars, evaluation dates and validation AUC.
- Calibration compares raw probabilities with logistic recalibration fitted to out-of-fold training scores. On validation, choose the global skip threshold minimizing estimated dollar loss. Buying a bad car costs 20% of its auction quote. Skipping a good car costs the positive part of its retail quote minus its auction quote minus $900. The test uses the frozen model, calibration and threshold.

## Project layout

| Location | Purpose |
| --- | --- |
| `carvana/preprocessing.py` | Source data, feature list, chronological splits and training-only cleaning |
| `carvana/training.py` | Shared model-family entry point and selected-model refit |
| `carvana/glassbox.py`, `tree_models.py`, `tree_training.py` (under `carvana/`) | Model definitions and family-specific tuning |
| `carvana/evaluation.py`, `calibration.py`, `business_loss.py` | Aligned validation scores, calibration and bid threshold |
| `carvana/tree_shap.py`, `holdout.py` | Tree explanations and frozen-policy test |
| `tests/` | Checks for chronological splitting, leakage, score alignment and decision costs |
| `doc/` | Presentation files and earlier report draft |

Current validation outputs go to `results/validation/drop_training_rows/acquisition_prices/tuned/`. Start with `six_model_auc.csv`, `calibration/thresholds.csv` and `shap/`. Final metrics are in `results/test/drop_training_rows/acquisition_prices/<model>/metrics.csv`. Other existing result folders are historical experiments. Vehicle-level scores can be regenerated from the raw data.

The presentation run selected **CatBoost Logloss** (validation AUC **0.66830**, internal test AUC **0.69791**). Its raw risk threshold was **0.54908**. The reference run used scikit-learn 1.9.0 and CatBoost 1.2.10. Run the automated checks with `python -m unittest discover -s tests -v`.
