import fire

from sparrow_resnet50_retinanet import save_pretrained

if __name__ == "__main__":
    commands = {
        "save-pretrained": save_pretrained,
    }
    fire.Fire(commands)
