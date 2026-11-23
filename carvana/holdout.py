"""Evaluate the frozen validation winner on the labelled, chronological internal test."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .business_loss import mistake_costs
from .calibration import frozen_policy, probabilities
from .evaluation import read_validation_scores, validation_auc
from .preprocessing import ROOT, TARGET, OUT, missing_quote_bundle, read_development_rows, read_holdout_rows
from .training import refit_score


def score_rule(rows, prices, scores, rule):
    y = rows[TARGET]
    if not y.index.equals(scores.index):
        raise ValueError("Probabilities and outcomes have different RefIds")
    bad = y.eq(1).to_numpy()
    skipped = scores.ge(rule.threshold).to_numpy()
    caught = (bad & skipped).sum()
    wrong_buy, wrong_skip = mistake_costs(y, prices, rule.bad_buy_loss_rate, rule.normal_cost_usd)
    buy_all = wrong_buy[bad].sum()
    loss = wrong_buy[bad & ~skipped].sum() + wrong_skip[~bad & skipped].sum()
    return dict(start_date=rows.PurchDate.min().date(), end_date=rows.PurchDate.max().date(),
                vehicles=len(y), bad_buys=bad.sum(), auc=roc_auc_score(y, scores),
                brier=brier_score_loss(y, scores), log_loss=log_loss(y, scores),
                skipped_bids=skipped.sum(), bad_buys_caught=caught, good_cars_skipped=(~bad & skipped).sum(),
                bad_buys_bought=bad.sum()-caught, bad_buy_recall=caught/bad.sum(),
                buy_all_loss_usd=buy_all/100, fixed_rule_loss_usd=loss/100,
                estimated_savings_usd=(buy_all-loss)/100)


def evaluate_holdout():
    train, validation = read_development_rows()
    calibration, rule = frozen_policy()
    saved = read_validation_scores(OUT, validation[TARGET])
    if validation_auc(validation[TARGET], saved).index[0] != rule.model:
        raise ValueError("Validation scores have changed. Regenerate the validation policy first.")
    test = read_holdout_rows()
    later = pd.concat([validation, test], verify_integrity=True)
    model, raw, features = refit_score(rule.model, train, later)
    np.testing.assert_allclose(raw.loc[validation.index], saved[rule.model], atol=1e-12, rtol=0)
    scores = probabilities(raw, calibration.intercept, calibration.slope)
    fit_info = {}
    if rule.model.startswith("CatBoost"):
        fit_info = dict(depth=model.get_param("depth"), leaf_penalty=model.get_param("l2_leaf_reg"))
    reports = []
    for split, rows in [("validation", validation), ("internal_test", test)]:
        metrics = score_rule(rows, features.loc[rows.index], scores.loc[rows.index], rule)
        if split == "validation" and (metrics["skipped_bids"] != rule.skipped_bids or
                                      not np.isclose(metrics["fixed_rule_loss_usd"], rule.loss_usd, rtol=0, atol=1e-6)):
            raise ValueError("Refit does not reproduce the frozen validation decision")
        reports.append(dict(split=split, model=rule.model, training_rows=(~missing_quote_bundle(train)).sum(),
                            **fit_info, calibration=rule.method, threshold=rule.threshold,
                            bad_buy_loss_rate=rule.bad_buy_loss_rate, normal_cost_usd=rule.normal_cost_usd, **metrics))
    directory = ROOT / "results" / "test" / "drop_training_rows" / "acquisition_prices" / rule.model.lower()
    directory.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(reports)
    report.to_csv(directory / "metrics.csv", index=False)
    return report
