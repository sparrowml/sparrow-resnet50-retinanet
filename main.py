import fire

from sparrow_resnet50_retinanet import sample_frames, save_pretrained

if __name__ == "__main__":
    commands = {
        "sample-frames": sample_frames,
        "save-pretrained": save_pretrained,
    }
    fire.Fire(commands)
