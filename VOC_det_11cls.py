import os
from PIL import Image
from defusedxml.ElementTree import parse as ET_parse

from torchvision import datasets, datapoints


CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDet11cls(datasets.VOCDetection):
    def __init__(self, root, mode, transforms, year="2012"):
        super().__init__(root, year, mode, False, None, None, transforms)
        self.mode = mode
        self.valid_classes = CLASSES[:11]
        self.vc2idx = dict(zip(self.valid_classes, range(1, len(self.valid_classes) + 1)))

        self.rel_path = self.images[0].replace(root, "")
        self.rel_path = self.rel_path[1:] if self.rel_path.startswith("/") else self.rel_path
        self.rel_path = "/".join(self.rel_path.split("/")[:-1])

        # filter out objects with selected classes
        self.anns = []
        for tar in self.targets:
            anns = self.parse_voc_xml(ET_parse(tar).getroot())
            objects = anns["annotation"]["object"]
            new_objects = []
            for obj in objects:
                clsname = obj["name"]
                if clsname in self.valid_classes:
                    new_objects.append(obj)
            if len(new_objects) > 0:
                anns["annotation"]["object"] = new_objects
                self.anns.append(anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]["annotation"]
        img_fp = os.path.join(self.root, self.rel_path, ann["filename"])
        img = Image.open(img_fp).convert("RGB")
        objects = ann["object"]
        bboxes = [(float(obj["bndbox"]["xmin"]), float(obj["bndbox"]["ymin"]), float(obj["bndbox"]["xmax"]), float(obj["bndbox"]["ymax"])) for obj in objects]  # tl_x, tl_y, br_x, br_y format
        bboxes = datapoints.BoundingBox(bboxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(int(ann["size"]["height"]), int(ann["size"]["width"])))  # TODO: check if this is a valid input of v2 transforms
        if self.transforms is not None:
            img, bboxes = self.transforms(img, bboxes)
        labels = [self.vc2idx[obj["name"]] for obj in objects]
        targets = {"boxes": bboxes, "labels": labels}
        return img, targets, img_fp
    
    def __len__(self):
        return len(self.anns)
