from collections import defaultdict

from sparrow_datums import FrameAugmentedBoxes
from sparrow_tracky import compute_moda_by_class, MODA

from .config import Config


def evaluate_predictions() -> None:
    moda_collector = defaultdict(MODA)
    n_evaluations = 0
    for annotation_path in Config.annotations_directory.glob("*.json.gz"):
        predictions_path = Config.predictions_directory / annotation_path.name
        if annotation_path.exists() and predictions_path.exists():
            n_evaluations += 1
            predicted_boxes = FrameAugmentedBoxes.from_file(predictions_path)
            ground_truth_boxes = FrameAugmentedBoxes.from_file(annotation_path)
            moda_dict = compute_moda_by_class(predicted_boxes, ground_truth_boxes)
            for label, moda in moda_dict.items():
                moda_collector[label] += moda
    print(f"{n_evaluations} evaluations")
    print("=========")
    total = MODA()
    for label in moda_collector.keys():
        name = Config.labels[label]
        sub_moda = moda_collector[label]
        total += sub_moda
        print(
            f"{name + ':':<15}{sub_moda.value:8.3f} moda, "
            f"{sub_moda.n_truth:5} true boxes, "
            f"{sub_moda.false_negatives:5} false negatives, "
            f"{sub_moda.false_positives:5} false positives "
        )
    print("=========")
    print(
        f"{'total moda:':<15}{total.value:8.3f} moda, "
        f"{total.n_truth:5} true boxes, "
        f"{total.false_negatives:5} false negatives, "
        f"{total.false_positives:5} false positives "
    )
