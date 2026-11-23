"""Run a model family, validate all six candidates, or evaluate the frozen winner."""
import argparse

from .registry import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*MODELS, "validate", "explain", "holdout", "all"],
                        help="'all' runs development and explanations; 'holdout' is separate")
    stage = parser.parse_args().stage
    if stage in (*MODELS, "all"):
        from .training import train_family
        for family in MODELS if stage == "all" else [stage]:
            print(train_family(family).to_string())
    if stage in ("validate", "all"):
        from .calibration import validate
        print(validate().to_string())
    if stage in ("explain", "all"):
        from .tree_shap import explain
        print(explain().to_string())
    if stage == "holdout":
        from .holdout import evaluate_holdout
        print(evaluate_holdout().to_string(index=False))


if __name__ == "__main__":
    main()
