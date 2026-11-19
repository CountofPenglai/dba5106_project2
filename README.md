# Carvana: pre-bid risk of a bad buy

The two notebooks compare four tree models using validation ROC-AUC. The positive class is a recorded bad purchase. The unit of analysis is a vehicle that a dealer historically bought. Download the [competition data from Kaggle](https://www.kaggle.com/competitions/DontGetKicked/data) and extract `training.csv` and `test.csv` into `data/DontGetKicked/`.

## Run from this folder

Use a Python 3.12 environment. Install `python -m pip install -r requirements.txt` and select that environment as your Jupyter kernel.

1. Run `python preprocessing.py` to generate the four shared tables under `data/processed/`.
2. Start `jupyter lab`, open `random_forest.ipynb` and `catboost_trees.ipynb`, and run all cells in each (either order).
3. Run `python evaluation.py` for all four validation AUCs in `results/validation/comparison.csv`. The per-family score files record each vehicle's RefId, outcome and unrounded scores.

## Field decisions

| Role | Raw fields | Reason |
|---|---|---|
| Numeric predictors | VehicleAge, VehOdo | Age and odometer at auction. |
| Categorical predictors | Make, Model, Size, Transmission, Auction, VNST, IsOnlineSale | Vehicle descriptors and the platform, location and channel of the auction. |
| Metadata | RefId, PurchDate, IsBadBuy | Join key, chronological split date, and training target respectively. |
| Reserve | Trim, SubModel, Color, VehYear, Nationality, TopThreeAmericanName, VNZIP1 | Potentially known before bidding; omit from the simple baseline to reduce detail or redundancy. |
| Exclude pending provenance | WheelType, WheelTypeID | Training subset: 2,367 missing wheel types, with a 66.33% bad-buy rate versus 9.62% when recorded. The dictionary does not timestamp recording. The association alone does not establish leakage. Both versions and their missingness are omitted. |
| Reserve pending quote time | All four acquisition MMR prices (including source spelling MMRAcquisitonRetailCleanPrice) | Reference estimates at purchase; the dictionary does not establish their availability before bidding. |
| Exclude | All four MMRCurrent prices, VehBCost, WarrantyCost | Later/current valuations, realised purchase price, or uncertain warranty quote time. |
| Exclude | BYRNO, PRIMEUNIT, AUCGUART | Buyer identifier or sparse flags with uncertain recording times. |

## Shared preparation and evaluation

Split labelled auctions by complete calendar dates: training through 2010-06-08 (51,191), validation through 2010-09-29 (11,022), and internal holdout thereafter (10,770). The official Kaggle test (48,707) has no target. Its dates overlap training; it is prepared only for potential submissions. The two notebooks only load training and validation.

Trim and uppercase categories; standardise missing strings. Vehicle models seen fewer than 20 times in training and unseen models map to `__OTHER__`. Numeric median imputation is fitted on training rows only. Both models receive the same nine raw inputs. Random forest one-hot encodes training categories. CatBoost uses native categorical handling. The two versions within each family share settings; training losses are squared error and log loss.

AUC uses continuous scores. A CatBoost RMSE score can fall outside [0,1]. For the later decision stage, `evaluation.threshold_costs(labels, scores)` enumerates thresholds for one selected score series, including buying all or skipping all. The proposed costs (skip a good car = 1; buy a bad car = 5) are assumptions. Select among validation costs only after model selection and any calibration; freeze the rule before the internal holdout.
