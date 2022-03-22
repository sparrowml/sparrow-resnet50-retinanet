from collections import defaultdict

from sparrow_datums import FrameAugmentedBoxes
from sparrow_tracky import compute_moda, MODA

from .config import DefaultConfig


def evaluate_predictions() -> None:
    moda_collector = defaultdict(MODA)
    n_evaluations = 0
    for annotation_path in DefaultConfig.annotations_directory.glob("*.json.gz"):
        predictions_path = DefaultConfig.predictions_directory / annotation_path.name
        if annotation_path.exists() and predictions_path.exists():
            n_evaluations += 1
            predicted_boxes = FrameAugmentedBoxes.from_file(predictions_path)
            ground_truth_boxes = FrameAugmentedBoxes.from_file(annotation_path)
            all_labels = set(predicted_boxes.labels) | set(ground_truth_boxes.labels)
            for label in all_labels:
                moda_collector[label] += compute_moda(
                    predicted_boxes[predicted_boxes.labels == label],
                    ground_truth_boxes[ground_truth_boxes.labels == label],
                )
    print(f"{n_evaluations} evaluations")
    print("=========")
    total = MODA()
    for label in moda_collector.keys():
        name = DefaultConfig.labels[label]
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
