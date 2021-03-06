import fire

from sparrow_resnet50_retinanet import (
    evaluate_predictions,
    export_model,
    import_predictions,
    run_predictions,
    sample_frames,
    save_pretrained,
    save_checkpoint,
    train_model,
    version_annotations,
)

if __name__ == "__main__":
    commands = {
        "evaluate-predictions": evaluate_predictions,
        "export-model": export_model,
        "import-predictions": import_predictions,
        "run-predictions": run_predictions,
        "sample-frames": sample_frames,
        "save-checkpoint": save_checkpoint,
        "save-pretrained": save_pretrained,
        "train-model": train_model,
        "version-annotations": version_annotations,
    }
    fire.Fire(commands)
