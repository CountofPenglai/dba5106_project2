"""Select a global skip-bid threshold on saved validation scores. Run: python business_loss.py"""

import numpy as np
import pandas as pd

from evaluation import _aligned, read_validation_scores, validation_auc
from preprocessing import ROOT, load_development

AUCTION = "MMRAcquisitionAuctionAveragePrice"
RETAIL = "MMRAcquisitionRetailAveragePrice"


def threshold_sensitivity(y, scores, prices, *, loss_rates=(0.10, 0.20, 0.30), normal_cost=900):
    """For each assumed loss rate, minimize total dollars lost across validation cars.

    Skip bids with scores >= threshold. Among equal minima, keep the highest
    threshold (fewer skipped bids). Charge each mistake to the nearest cent.
    """
    _aligned(y, scores)
    if not prices.index.equals(y.index):
        raise ValueError("Quotes must share the validation labels' RefIds in order")
    quotes = prices[[AUCTION, RETAIL]].to_numpy(dtype=float)
    if not np.isfinite(quotes).all() or (quotes <= 0).any():
        raise ValueError("Cost evaluation requires finite, positive acquisition quotes")
    if not np.isfinite(normal_cost) or normal_cost < 0:
        raise ValueError("Normal cost must be finite and nonnegative")

    # Buy all at +inf. As the threshold falls, all cars tied at a score switch to skip.
    order = np.argsort(-scores.to_numpy(), kind="stable")
    risk = scores.to_numpy()[order]
    group_ends = np.r_[np.flatnonzero(risk[:-1] != risk[1:]) + 1, len(risk)]
    cutoffs = np.r_[0, group_ends]
    thresholds = np.r_[np.inf, risk[group_ends - 1]]

    bad = y.to_numpy()[order] == 1
    auction, retail = quotes[order].T
    missed_profit = np.rint(100 * np.maximum(0, retail - auction - normal_cost)).astype(np.int64)

    def prefix_sums(values):
        return np.r_[0, np.cumsum(values)][cutoffs]

    skipped_good = prefix_sums(~bad)
    bought_bad = bad.sum() - prefix_sums(bad)
    skipped_good_loss = prefix_sums(np.where(bad, 0, missed_profit))
    rows = []
    for rate in loss_rates:
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Bad-buy loss rate must be between zero and one")
        bad_loss = np.where(bad, np.rint(auction * rate * 100).astype(np.int64), 0)
        bought_bad_loss = bad_loss.sum() - prefix_sums(bad_loss)
        total = skipped_good_loss + bought_bad_loss
        best = int(total.argmin())
        rows.append({
            "bad_buy_loss_rate": rate, "normal_cost_usd": normal_cost,
            "threshold": thresholds[best], "validation_cars": len(y),
            "skipped_bids": cutoffs[best], "skipped_good_FP": skipped_good[best],
            "bought_bad_FN": bought_bad[best],
            "skipped_good_loss_usd": skipped_good_loss[best] / 100,
            "bought_bad_loss_usd": bought_bad_loss[best] / 100,
            "loss_usd": total[best] / 100, "loss_per_car_usd": total[best] / (100 * len(y)),
            "buy_all_loss_usd": total[0] / 100, "skip_all_loss_usd": total[-1] / 100,
        })
    return pd.DataFrame(rows)


def main():
    directory = ROOT / "results" / "validation" / "acquisition_prices"
    _, _, prices, labels = load_development()
    scores = read_validation_scores(directory, labels)
    ranking = validation_auc(labels, scores)
    model = ranking.index[0]
    report = threshold_sensitivity(labels, scores[model], prices)
    report.insert(0, "model", model)
    report.insert(1, "validation_auc", ranking[model])
    report.to_csv(directory / "business_loss.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
