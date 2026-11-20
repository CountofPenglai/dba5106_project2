# Carvana: pre-bid risk of a bad buy

The two notebooks compare four tree models using validation ROC-AUC. The positive class is a recorded bad purchase. The unit of analysis is a vehicle that a dealer historically bought. Download the [competition data from Kaggle](https://www.kaggle.com/competitions/DontGetKicked/data) and extract `training.csv` and `test.csv` into `data/DontGetKicked/`.

**Current training policy.** CatBoost L2 keeps the 15 pre-bid predictors, depth 8 and leaf penalty 3. Before fitting each model, exclude training cars whose four acquisition quotes in source order are exactly (0, 1, 0, 0). This leaves 51,060 of the 51,191 training cars. Every car in a scoring block still receives a score. The comparison of 11 and 15 predictors uses the same eligible training IDs.

The validation figures, bidding threshold and SHAP results linked under `results/validation/acquisition_prices/` describe the original preprocessing. The adopted rule has training-fold results only. Future validation runs write under `results/validation/drop_training_rows/`.

## Run from this folder

Use a Python 3.12 environment. Install `python -m pip install -r requirements.txt` and select that environment as your Jupyter kernel.

1. Run `python preprocessing.py` to generate the four shared tables under `data/processed/`.
2. Start `jupyter lab`, open `random_forest.ipynb` and `catboost_trees.ipynb`, and run all cells in each (either order).
3. Run `python evaluation.py` for the selected fifteen-field validation AUCs in `results/validation/drop_training_rows/acquisition_prices/auc.csv`. The per-family score files record each vehicle's RefId, outcome and unrounded scores.
4. Run `python business_loss.py` after generating baseline scores to select a bidding threshold in the new results directory.

## Training-only comparison of quote treatments

Run `python quote_treatments.py` to compare the original preprocessing, four peer-median quotes, and dropping affected training rows. The affected bundle in source order is (0, 1, 0, 0). Each of the three expanding training windows contained the same 131 affected cars. Each later scoring block contained none. The experiment fixes CatBoost L2 (depth 8, leaf penalty 3, 250 trees, learning rate 0.05, seed 42) and the 15 inputs. It reads only the training split.

Peer medians use training cars with four positive quotes. Match make, age and size with at least 20 donors, then fall back to age and size with the same minimum. All affected cars found a match in these two groups. Original preprocessing is retained for other rows in the imputation comparison. The deletion variant fits the preprocessing and model on the reduced training window. The three methods score identical cars in each fold.

Training-row exclusion was adopted over peer-median replacement on mean ROC-AUC. Across the three equally weighted folds, the mean was 0.655524 for the original treatment, 0.653692 for peer medians, and 0.654083 for dropping the rows. [Fold details](results/training_cv/quote_treatments/folds.csv) · [Comparison](results/training_cv/quote_treatments/comparison.csv)

## Historical high-risk feature comparison

Run `python high_risk_features.py` for the fixed CatBoost L2 experiment (depth 8, leaf penalty 3). All four variants include the original fifteen fields. The three alternatives add relative mileage, the auction condition spread, or both. Relative mileage divides odometer mileage by the training median for the same age and size. A missing or nonpositive peer median falls back to the age median and then to the full training median. The spread is (clean auction quote − average auction quote) / average auction quote, using the same acquisition-time quotes and fold-specific imputation as the baseline.

The [three chronological folds](results/validation/acquisition_prices/high_risk_features/folds.csv) and [validation comparison](results/validation/acquisition_prices/high_risk_features/comparison.csv) report ROC-AUC and precision among the top 1% and 5% of scores. Each review budget rounds up to a whole car, with ascending RefId breaking ties. Cross-validation means give each fold equal weight. These review lists measure ranking quality. The earlier bidding policy has its own separate evaluation.

## Historical validation experiment

The two model families use the same fifteen pre-bid inputs, including the four acquisition-time MMR quotes. Trim and SubModel are joined by RefId for the training and validation cars. The four configurations differ by model family and training objective; compare their validation ROC-AUCs in [the archived results table](results/validation/acquisition_prices/auc.csv). Before tuning, CatBoost with log loss ranked first (0.66693).

### Acquisition-time price comparison

The selected version includes four numeric acquisition-time MMR estimates (auction/retail, average/clean condition). For the earlier eleven-field comparison only, set `include_prices = False` in both notebooks. This comparison assumes the historical quotes are available when bidding. All estimator settings and the chronological development split are shared. Generate the paired [validation AUC table](results/validation/acquisition_prices/comparison.csv) with `python evaluation.py --compare-prices` after running both feature sets. Price-run outputs have a separate directory. On the same 11,022 validation vehicles, all four variants improved; CatBoost Logloss reached 0.66693.

For these predictors only, a missing or nonpositive quote is replaced with the median positive training quote for that field. At least one quote needed replacement for 693 training and 114 validation vehicles. This earlier experiment retained all training rows and used no additional missingness indicators. Both notebooks use only the training and validation vehicles; the internal holdout and official test are outside this experiment.

## Field decisions

| Role | Raw fields | Reason |
|---|---|---|
| Numeric predictors | VehicleAge, VehOdo | Age and odometer at auction. |
| Categorical predictors | Make, Model, Size, Transmission, Auction, VNST, IsOnlineSale | Vehicle descriptors and the platform, location and channel of the auction. |
| Metadata | RefId, PurchDate, IsBadBuy | Join key, chronological split date, and training target respectively. |
| Selected additional predictors | Trim, SubModel | Vehicle trim and submodel; included in the selected setup. |
| Reserve | Color, VehYear, Nationality, TopThreeAmericanName, VNZIP1 | Potentially known before bidding; omit from the selected feature set to reduce detail or redundancy. |
| Exclude pending provenance | WheelType, WheelTypeID | Training subset: 2,367 missing wheel types, with a 66.33% bad-buy rate versus 9.62% when recorded. The dictionary does not timestamp recording. The association alone does not establish leakage. Both versions and their missingness are omitted. |
| Selected numeric predictors and cost inputs | All four acquisition MMR prices (including source spelling MMRAcquisitonRetailCleanPrice) | The model uses all four quotes. The cost calculation uses the two average-condition quotes. Actual resale proceeds, repair costs and net margins are unobserved. |
| Exclude | All four MMRCurrent prices, VehBCost, WarrantyCost | Later/current valuations, realised purchase price, or uncertain warranty quote time. |
| Exclude | BYRNO, PRIMEUNIT, AUCGUART | Buyer identifier or sparse flags with uncertain recording times. |

## Shared preparation and evaluation

Split labelled auctions by complete calendar dates: training through 2010-06-08 (51,191), validation through 2010-09-29 (11,022), and internal holdout thereafter (10,770). The official Kaggle test (48,707) has no target. Its dates overlap training; it is prepared only for potential submissions. The two notebooks only load training and validation.

Trim and uppercase categories; standardise missing strings. Vehicle models seen fewer than 20 times in training and unseen models map to `__OTHER__`. Numeric median imputation is fitted on training rows only. Both models receive the same fifteen raw inputs for validation. Random forest one-hot encodes training categories. CatBoost uses native categorical handling. The two versions within each family share settings; training losses are squared error and log loss.

AUC uses continuous scores. A CatBoost RMSE score can fall outside [0,1].

## Historical business loss and threshold selection

Select the highest-AUC model among the four fifteen-field tree variants, then reuse its saved validation scores. Skip a car if its score is at least the threshold. With A and R denoting the acquisition-time average auction and retail quotes in dollars, count each decision as follows.

- Buying a bad car costs `loss_rate × A`. The main scenario assumes a 20% net loss. The sensitivity check repeats the calculation at 10% and 30%.
- Skipping a good car costs `max(0, R − A − 900)`. The $900 allowance covers routine preparation and selling costs. Correct decisions carry zero loss.

These are estimated decision costs. The same training-median quote substitution used for modelling supplies missing or nonpositive quotes for 114 validation cars. The evaluator scores every distinct cutoff, including buying all and skipping all. Equal scores move together. If several cutoffs tie for minimum dollar loss, it keeps the highest one.

Among the untuned variants, CatBoost Logloss ranked first by validation AUC (0.66693). All three scenarios selected the same cutoff, approximately 0.48522. It skipped four of 11,022 validation cars, all recorded bad buys. At 20%, estimated loss was $1,548,297.40, compared with $1,561,908.00 for buying every car. The CSV retains the full-precision cutoff. These figures use the same validation data that selected the cutoff. The internal holdout and Kaggle test are outside this evaluation.

## Black-box hyperparameter tuning

Run `python tune_trees.py` to tune both model families and select the bidding cutoff for the active preprocessing policy. Each notebook also describes how to run its own family. The [tuned validation comparison](results/validation/acquisition_prices/tuned/comparison.csv) and [tuned decision costs](results/validation/acquisition_prices/tuned/business_loss.csv) sit alongside the per-family fold and finalist tables. The original results remain under `acquisition_prices/`.

1. Screen each version over three expanding training windows. The later blocks cover December 2009–January 2010, February–March 2010, and April 1–June 8, 2010. Train on every earlier development date. Fit price and numeric medians, rare-model grouping and category encoding separately in each fold.
2. Random forest tests depths 8, 14 and 20 with minimum leaf sizes 12 and 30. CatBoost tests depths 4, 6 and 8 with L2 leaf penalties 3 and 10. The output column `regularization` denotes the respective leaf size or penalty. Keep the two highest mean fold AUCs for each model variant plus its baseline, counting duplicate configurations once.
3. Refit each finalist on the eligible training cars (51,060 under the current policy) and select on the 11,022-car development validation set. Within 0.0001 of the best validation AUC for a variant, prefer shallower trees, then lower fold AUC standard deviation, then stronger regularization. Compare the four selected variants on validation AUC and use the highest-ranked variant's continuous scores for bidding.
4. Set a global threshold using the same dollar loss and 10%, 20% and 30% scenarios as the baseline. Validation outcomes inform both model and threshold selection. The scores and decision savings are development results.

With the original preprocessing, the tuned CatBoost squared-error model had the highest validation AUC (0.66849, depth 8 and L2 leaf penalty 3). All three cost scenarios selected a risk-score threshold of 0.4306593054718898. The rule skipped eight bad buys and zero good cars in validation. At 20%, estimated savings against buying every car were $20,510.60, compared with $13,610.60 for the untuned decision rule.

## Calibration and historical bidding results

Run `python calibration.py` after tuning. It takes the validation-AUC winner and its saved validation scores. For the logistic recalibration, the same fixed tree configuration and fold-specific preprocessing produce predictions for 18,774 training-period cars across the three chronological folds. Each car is scored using only earlier dates. A logistic regression maps the log odds of these scores to a bad-buy probability. Input scores are clipped to [0.000001, 0.999999] for finite log odds. All validation scores in this run already fell inside that interval.

Compare both probability estimates on the same 11,022 validation cars. Use the logistic mapping only if its slope is positive, its Brier score improves and its log loss does not increase. The [calibration metrics](results/validation/acquisition_prices/tuned/calibration/metrics.csv) include both fitted coefficients. The [reliability plot](results/validation/acquisition_prices/tuned/calibration/reliability.png) compares mean predicted and observed risk in ten roughly equal-sized groups. Expected calibration error (ECE) weights each group's absolute difference by its size.

With the original preprocessing, the raw CatBoost L2 scores averaged 13.17% against an observed 12.66%. Logistic recalibration raised the mean to 14.51% and increased both Brier score and log loss, so the selected rule uses raw scores. The historical rule skipped bids at or above 0.4306593054718898. The 20% loss scenario saved an estimated $20,510.60 versus buying every validation vehicle. The 10% and 30% scenarios chose the same eight cars. The increasing logistic transformation preserved their order, giving the same decisions at its own cutoff. The [threshold comparison](results/validation/acquisition_prices/tuned/calibration/thresholds.csv) records both scales and the selected method. All figures here describe development validation.

## TreeSHAP and historical explanations

Run `python tree_shap.py` after tuning and calibration. CatBoost's native TreeSHAP values explain the raw score for all 11,022 validation cars in the selected 15-field model. The expected model score is 0.12238. Adding the 15 signed contributions to it reconstructs each prediction. The 20% loss scenario and $900 allowance determine the bidding cutoff separately. [CatBoost SHAP documentation](https://catboost.ai/docs/en/concepts/shap-values)

The [global ranking](results/validation/acquisition_prices/tuned/shap/global_importance.png) uses mean absolute contributions across cars. Age accounted for 20.4% of total absolute attribution, followed by odometer (12.6%) and vehicle model (12.4%). Age contributed −0.0226 on average for cars aged three years or less and +0.0296 for cars aged five years or more. These are patterns in the fitted scores. Correlated vehicle descriptors and price quotes can share attribution. [All 15 features](results/validation/acquisition_prices/tuned/shap/global_importance.csv) · [Numeric feature contrasts](results/validation/acquisition_prices/tuned/shap/numeric_contrasts.csv)

The [local explanations](results/validation/acquisition_prices/tuned/shap/local_explanations.png) compare the lowest-scored car skipped by the rule with the car nearest the median score among bad buys that passed it. RefId breaks ties. Both examples were recorded bad buys. The first reached the 0.430659 cutoff, with age eight contributing +0.0830. The second scored 0.1562. Its age and mileage raised the score, while its average-condition auction quote lowered it. [All contributions and model input values](results/validation/acquisition_prices/tuned/shap/local_explanations.csv)
