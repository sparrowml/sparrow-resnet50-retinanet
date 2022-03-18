import fire

from sparrow_resnet50_retinanet import (
    import_predictions,
    run_predictions,
    sample_frames,
    save_pretrained,
)

if __name__ == "__main__":
    commands = {
        "import-predictions": import_predictions,
        "run-predictions": run_predictions,
        "sample-frames": sample_frames,
        "save-pretrained": save_pretrained,
    }
    fps = fire.Fire(commands)
