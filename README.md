# Carvana: pre-bid risk of a bad buy

The three notebooks compare two sparse linear Glassbox models and four tree models using validation ROC-AUC. The positive class is a recorded bad purchase. The unit of analysis is a vehicle that a dealer historically bought. Download the [competition data from Kaggle](https://www.kaggle.com/competitions/DontGetKicked/data) and extract `training.csv` and `test.csv` into `data/DontGetKicked/`.

## Run from this folder

Use a Python 3.12 environment. Install `python -m pip install -r requirements.txt` and select that environment as your Jupyter kernel.

1. Run `python preprocessing.py` to generate the four shared tables under `data/processed/`.
2. Start `jupyter lab` and run all cells in `glassbox_linear.ipynb`, `random_forest.ipynb` and `catboost_trees.ipynb` (any order). `python glassbox.py` is an equivalent non-notebook entry point for the linear models.
3. Run `python evaluation.py` for the selected fifteen-field validation AUCs in `results/validation/acquisition_prices/auc.csv`. The per-family score files record each vehicle's RefId, outcome and unrounded scores. The combined table contains six rows after all three model families have run.

## Validation experiment

The three model families use the same fifteen pre-bid inputs, including the four acquisition-time MMR quotes. Trim and SubModel are joined by RefId for the training and validation cars. The six configurations differ by model family and training objective. The [currently committed results table](results/validation/acquisition_prices/auc.csv) predates the Glassbox implementation and contains the four preliminary tree results: CatBoost Logloss 0.66693, CatBoost L2 0.66683, RF L2 0.66522 and RF Logloss 0.66427. Those tree estimates use shared within-family settings rather than separate loss-specific tuning and remain preliminary.

### Glassbox models

`Glassbox_L2` is a Lasso linear model trained with squared error. It selects `alpha` from `1e-5, 1e-4, 1e-3, 1e-2, 1e-1`. `Glassbox_Logloss` is L1-penalised logistic regression using the SAGA solver. It selects `C` from `0.01, 0.1, 1, 10`. Each model selects its own hyperparameter by AUC on a complete-date inner split: approximately the first 80% of the prepared training cars are inner training and the final 20% are inner tuning. Neither tuning process sees the formal validation set.

Numeric fields are standardised using the relevant training rows. All categorical fields use one-hot encoding, with levels observed fewer than 20 times grouped into the infrequent level and unknown levels handled without failure. The two versions use the same linear form and L1 regularisation. They do not use resampling, SMOTE or class weights.

The notebook writes four artifacts under `results/validation/acquisition_prices/`: `glassbox_scores.csv`, `glassbox_auc.csv`, `glassbox_coefficients.csv` and `glassbox_top_effects.png`. Row-level score files are ignored by Git. `glassbox_coefficients.csv` records the encoded feature, raw field, reference category, numeric training scale, both coefficients, logistic odds ratio and indicators for nonzero and direction-stable effects.

For a numeric field, a coefficient describes a one-training-standard-deviation increase. The L2 coefficient is a risk-score change; the Logistic coefficient is a log-odds change and its exponential is an odds ratio. Categorical coefficients are relative to their recorded one-hot reference level. Raw coefficient magnitudes are not compared across the two losses. All interpretations describe predictive associations, not causal effects.

This Glassbox stage ends with validation scores and linear interpretation. Calibration, business-loss threshold selection and internal-holdout evaluation are performed after the six-model AUC decision.

### Acquisition-time price comparison

The selected version includes four numeric acquisition-time MMR estimates (auction/retail, average/clean condition). For the earlier eleven-field tree comparison only, set `include_prices = False` in the two tree notebooks. This comparison assumes the historical quotes are available when bidding. All estimator settings and the chronological development split are shared. Generate the paired [validation AUC table](results/validation/acquisition_prices/comparison.csv) with `python evaluation.py --compare-prices` after running both tree feature sets. Price-run outputs have a separate directory. On the same 11,022 validation vehicles, all four tree variants improved; CatBoost Logloss reached 0.66693.

For these predictors only, a missing or nonpositive quote is replaced with the median positive training quote for that field. At least one quote needed replacement for 693 training and 114 validation vehicles. No rows are dropped and no additional missingness indicators are added. All three notebooks use only the training and validation vehicles; the internal holdout and official test are outside this experiment.

## Field decisions

| Role | Raw fields | Reason |
|---|---|---|
| Numeric predictors | VehicleAge, VehOdo | Age and odometer at auction. |
| Categorical predictors | Make, Model, Size, Transmission, Auction, VNST, IsOnlineSale | Vehicle descriptors and the platform, location and channel of the auction. |
| Metadata | RefId, PurchDate, IsBadBuy | Join key, chronological split date, and training target respectively. |
| Selected additional predictors | Trim, SubModel | Vehicle trim and submodel; included in the selected setup. |
| Reserve | Color, VehYear, Nationality, TopThreeAmericanName, VNZIP1 | Potentially known before bidding; omit from the selected feature set to reduce detail or redundancy. |
| Exclude pending provenance | WheelType, WheelTypeID | Training subset: 2,367 missing wheel types, with a 66.33% bad-buy rate versus 9.62% when recorded. The dictionary does not timestamp recording. The association alone does not establish leakage. Both versions and their missingness are omitted. |
| Selected numeric predictors / proposed cost analysis | All four acquisition MMR prices (including source spelling MMRAcquisitonRetailCleanPrice) | Used in the selected fifteen-field model and proposed for separate cost analysis. These are market estimates; actual resale proceeds, repair costs and net margins are unobserved. |
| Exclude | All four MMRCurrent prices, VehBCost, WarrantyCost | Later/current valuations, realised purchase price, or uncertain warranty quote time. |
| Exclude | BYRNO, PRIMEUNIT, AUCGUART | Buyer identifier or sparse flags with uncertain recording times. |

## Shared preparation and evaluation

Split labelled auctions by complete calendar dates: training through 2010-06-08 (51,191), validation through 2010-09-29 (11,022), and internal holdout thereafter (10,770). The official Kaggle test (48,707) has no target. Its dates overlap training; it is prepared only for potential submissions. The three notebooks only load training and validation.

Trim and uppercase categories; standardise missing strings. Vehicle models seen fewer than 20 times in training and unseen models map to `__OTHER__`. Numeric median imputation is fitted on training rows only. All models receive the same fifteen raw inputs for validation. Glassbox and random forest one-hot encode training categories. CatBoost uses native categorical handling. Glassbox tunes each training loss independently; the current tree versions share settings within each family. Training losses are squared error and log loss.

AUC uses continuous scores. A CatBoost RMSE score can fall outside [0,1]. For the later decision stage, `evaluation.threshold_costs(labels, scores)` enumerates thresholds for one selected score series, including buying all or skipping all. The existing helper uses fixed scenario costs (skip a good car = 1; buy a bad car = 5). A per-vehicle cost formula using acquisition-time reference prices is under discussion. Select among validation costs only after model selection and any calibration; freeze the rule before the internal holdout.
