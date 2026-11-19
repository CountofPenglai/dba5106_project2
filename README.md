# Carvana: pre-bid risk of a bad buy

The two notebooks compare four tree models using validation ROC-AUC. The positive class is a recorded bad purchase. The unit of analysis is a vehicle that a dealer historically bought. Download the [competition data from Kaggle](https://www.kaggle.com/competitions/DontGetKicked/data) and extract `training.csv` and `test.csv` into `data/DontGetKicked/`.

## Run from this folder

Use a Python 3.12 environment. Install `python -m pip install -r requirements.txt` and select that environment as your Jupyter kernel.

1. Run `python preprocessing.py` to generate the four shared tables under `data/processed/`.
2. Start `jupyter lab`, open `random_forest.ipynb` and `catboost_trees.ipynb`, and run all cells in each (either order).
3. Run `python evaluation.py` for the selected fifteen-field validation AUCs in `results/validation/acquisition_prices/auc.csv`. The per-family score files record each vehicle's RefId, outcome and unrounded scores.
4. Run `python business_loss.py` for the baseline models to select a bidding threshold and reproduce the [validation cost comparison](results/validation/acquisition_prices/business_loss.csv) from the saved scores.

## Validation experiment

The two model families use the same fifteen pre-bid inputs, including the four acquisition-time MMR quotes. Trim and SubModel are joined by RefId for the training and validation cars. The four configurations differ by model family and training objective; compare their validation ROC-AUCs in [the current results table](results/validation/acquisition_prices/auc.csv). Before tuning, CatBoost with log loss ranked first (0.66693).

### Acquisition-time price comparison

The selected version includes four numeric acquisition-time MMR estimates (auction/retail, average/clean condition). For the earlier eleven-field comparison only, set `include_prices = False` in both notebooks. This comparison assumes the historical quotes are available when bidding. All estimator settings and the chronological development split are shared. Generate the paired [validation AUC table](results/validation/acquisition_prices/comparison.csv) with `python evaluation.py --compare-prices` after running both feature sets. Price-run outputs have a separate directory. On the same 11,022 validation vehicles, all four variants improved; CatBoost Logloss reached 0.66693.

For these predictors only, a missing or nonpositive quote is replaced with the median positive training quote for that field. At least one quote needed replacement for 693 training and 114 validation vehicles. No rows are dropped and no additional missingness indicators are added. Both notebooks use only the training and validation vehicles; the internal holdout and official test are outside this experiment.

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

## Business loss and threshold selection

Select the highest-AUC model among the four fifteen-field tree variants, then reuse its saved validation scores. Skip a car if its score is at least the threshold. With A and R denoting the acquisition-time average auction and retail quotes in dollars, count each decision as follows.

- Buying a bad car costs `loss_rate × A`. The main scenario assumes a 20% net loss. The sensitivity check repeats the calculation at 10% and 30%.
- Skipping a good car costs `max(0, R − A − 900)`. The $900 allowance covers routine preparation and selling costs. Correct decisions carry zero loss.

These are estimated decision costs. The same training-median quote substitution used for modelling supplies missing or nonpositive quotes for 114 validation cars. The evaluator scores every distinct cutoff, including buying all and skipping all. Equal scores move together. If several cutoffs tie for minimum dollar loss, it keeps the highest one.

Among the untuned variants, CatBoost Logloss ranked first by validation AUC (0.66693). All three scenarios selected the same cutoff, approximately 0.48522. It skipped four of 11,022 validation cars, all recorded bad buys. At 20%, estimated loss was $1,548,297.40, compared with $1,561,908.00 for buying every car. The CSV retains the full-precision cutoff. These figures use the same validation data that selected the cutoff. The internal holdout and Kaggle test are outside this evaluation.

## Black-box hyperparameter tuning

Run `python tune_trees.py` to reproduce both model families and the bidding decision. Each notebook also describes how to run its own family. The [tuned validation comparison](results/validation/acquisition_prices/tuned/comparison.csv) and [tuned decision costs](results/validation/acquisition_prices/tuned/business_loss.csv) sit alongside the per-family fold and finalist tables. The original results remain under `acquisition_prices/`.

1. Screen each version over three expanding training windows. The later blocks cover December 2009–January 2010, February–March 2010, and April 1–June 8, 2010. Train on every earlier development date. Fit price and numeric medians, rare-model grouping and category encoding separately in each fold.
2. Random forest tests depths 8, 14 and 20 with minimum leaf sizes 12 and 30. CatBoost tests depths 4, 6 and 8 with L2 leaf penalties 3 and 10. The output column `regularization` denotes the respective leaf size or penalty. Keep the two highest mean fold AUCs for each model variant plus its baseline, counting duplicate configurations once.
3. Refit each finalist on all 51,191 training cars and select on the 11,022-car development validation set. Within 0.0001 of the best validation AUC for a variant, prefer shallower trees, then lower fold AUC standard deviation, then stronger regularization. Compare the four selected variants on validation AUC and use the highest-ranked variant's continuous scores for bidding.
4. Set a global threshold using the same dollar loss and 10%, 20% and 30% scenarios as the baseline. Validation outcomes inform both model and threshold selection. The scores and decision savings are development results.

The tuned CatBoost squared-error model had the highest validation AUC (0.66849, depth 8 and L2 leaf penalty 3). All three cost scenarios selected a risk-score threshold of 0.4306593054718898. The rule skipped eight bad buys and zero good cars in validation. At 20%, estimated savings against buying every car were $20,510.60, compared with $13,610.60 for the untuned decision rule.
