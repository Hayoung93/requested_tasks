import os
import numpy as np
from PIL import Image
from defusedxml.ElementTree import parse as ET_parse

from torchvision import datasets, datapoints
import torchvision.transforms.functional as ttf


CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDet11cls(datasets.VOCDetection):
    def __init__(self, root, mode, transforms, year="2012"):
        super().__init__(root, year, mode, False, None, None, transforms)
        self.mode = mode
        self.valid_classes = CLASSES[:11]
        print("Classes: {}".format(" | ".join(self.valid_classes)))
        self.vc2idx = dict(zip(self.valid_classes, range(1, len(self.valid_classes) + 1)))

        self.rel_path = self.images[0].replace(root, "")
        self.rel_path = self.rel_path[1:] if self.rel_path.startswith("/") else self.rel_path
        self.rel_path = "/".join(self.rel_path.split("/")[:-1])

        # adjust train and validation set, reserve only 1000 example for validation

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
        w, h = img.size
        objects = ann["object"]
        bboxes = [(float(obj["bndbox"]["xmin"]), float(obj["bndbox"]["ymin"]), float(obj["bndbox"]["xmax"]), float(obj["bndbox"]["ymax"])) for obj in objects]  # tl_x, tl_y, br_x, br_y format
        labels = [self.vc2idx[obj["name"]] for obj in objects]
        """
        # debugging for degenerate boxes
        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            if (xmax - xmin <= 0) or (ymax - ymin <= 0):
                raise Exception("Degenerate box detected before transform")
        """
        # bboxes_datapoint = datapoints.BoundingBox(bboxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(int(ann["size"]["height"]), int(ann["size"]["width"])))  # TODO: check if this is a valid input of v2 transforms
        if self.transforms is not None:
            # img, bboxes_datapoint = self.transforms(img, bboxes_datapoint)
            transformed = self.transforms(image=np.asarray(img), bboxes=bboxes, class_labels=labels)
            img, bboxes, labels = transformed["image"], transformed["labels"], transformed["class_labels"]
            """
            # debugging for degenerate boxes
            for bi, box in enumerate(bboxes_datapoint.data):
                xmin, ymin, xmax, ymax = box
                if (xmax - xmin <= 0) or (ymax - ymin <= 0):
                    raise Exception("Degenerate box detected after transform\n Before: {} | After: {} | Transforms: {}".format(bboxes[bi], box, self.transforms.transforms))
            """
            # # discard degenerated bboxes, created because of cropping
            # valid_box_idx = []
            # for bi, box in enumerate(bboxes_datapoint.data):
            #     xmin, ymin, xmax, ymax = box
            #     if (xmax - xmin > 0) and (ymax - ymin > 0):
            #         valid_box_idx.append(bi)
            # if len(bboxes) > len(valid_box_idx):
            #     print("degenerate boxes discarded")
        img = ttf.to_tensor(img)
        img = ttf.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        targets = {"boxes": bboxes, "labels": labels}
        return img, targets, img_fp
    
    def __len__(self):
        return len(self.anns)
