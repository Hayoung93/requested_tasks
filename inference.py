import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from config import get_cfg
from models import get_model
from dataset import FaceForensicspp


def main(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testset = FaceForensicspp(args, cfg, "test", transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.run.num_workers)
    model = get_model(cfg, None)
    if (cfg.io.resume is not None) and (cfg.io.resume != "") and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume)
        model.load_state_dict(cp["model"])
    else:
        raise Exception("No checkpoint is provided for inference.")
    model.to(device)
    model.eval()
    pbar = tqdm(testloader)
    with torch.inference_mode():
        classwise_correct = [0] * args.num_classes
        classwise_count = [0] * args.num_classes
        for it, data in enumerate(pbar):
            img, label, fp = data
            inputs = img.to(device)
            label = label.to(device)
            outputs, losses = model(inputs, label)
            pred = outputs.argmax(dim=1)
            for p, l in zip(pred, label):
                classwise_correct[l] += (p == l).item()
                classwise_count[l] += 1
        classwise_acc = torch.tensor(classwise_correct) / torch.tensor(classwise_count)
        for ci in range(1, args.num_classes + 1):
            print("ACC_class-{}: {:.6f}".format(ci, classwise_acc[ci - 1].item()))


if __name__ == "__main__":
    args, cfg = get_cfg()
    main(args, cfg)
