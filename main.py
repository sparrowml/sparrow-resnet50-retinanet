import fire

from sparrow_resnet50_retinanet import (
    evaluate_predictions,
    import_predictions,
    run_predictions,
    sample_frames,
    save_checkpoint,
    save_pretrained,
    train_model,
    version_annotations,
)

if __name__ == "__main__":
    commands = {
        "evaluate-predictions": evaluate_predictions,
        "import-predictions": import_predictions,
        "run-predictions": run_predictions,
        "sample-frames": sample_frames,
        "save-checkpoint": save_checkpoint,
        "save-pretrained": save_pretrained,
        "train-model": train_model,
        "version-annotations": version_annotations,
    }
    fire.Fire(commands)
