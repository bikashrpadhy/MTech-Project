import os
import torch
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import numpy as np
import copy
import cv2

import our_utils

from torch.utils.data import Dataset
from PIL import Image, ImageDraw

def load_voc_instances(dirname, split, class_names):
    
    file_list_path = os.path.join(dirname, "ImageSets", "Main", split + ".txt")
    fileids = np.loadtxt(file_list_path, dtype=str)

    annotation_dirname = os.path.join(dirname, "Annotations/")
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with open(anno_file, "r") as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
 
        hand_annotations = {}
        body_annotations = {}
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls == "hand":
                hand_px = [float(bndbox.find(x).text) for x in ["x1", "x2", "x3", "x4"]]
                hand_py = [float(bndbox.find(x).text) for x in ["y1", "y2", "y3", "y4"]]
                hand_poly = [(x, y) for x, y in zip(hand_px, hand_py)]
            else:
                body_px = [bbox[0], bbox[2], bbox[2], bbox[0]]
                body_py = [bbox[1], bbox[1], bbox[3], bbox[3]]
                body_poly = [(x, y) for x, y in zip(body_px, body_py)]
            body_id = int(obj.find("body_id").text)
            if cls == "hand":
                hand_ann = {
                        "category_id": class_names.index(cls), "bbox": bbox, 
                        "body_id": body_id, "segmentation": [hand_poly],
                    }
                if body_id in hand_annotations:
                    pass
                else:
                    hand_annotations[body_id] = []
                hand_annotations[body_id].append(hand_ann)
            else:
                body_ann = {
                     "category_id": class_names.index(cls), "bbox": bbox, 
                     "body_id": body_id, "segmentation": [body_poly], 
                    }
                if body_id in body_annotations:
                    pass 
                else:
                    body_annotations[body_id] = []
                body_annotations[body_id].append(body_ann)  
        
        instances = []
        for body_id in hand_annotations:
            body_ann = body_annotations[body_id][0]
            for hand_ann in hand_annotations[body_id]:
                hand_ann["body_box"] = body_ann["bbox"]
                instances.append(hand_ann)
            body_ann["body_box"] = body_ann["bbox"]
            instances.append(body_ann)

        r["annotations"] = instances
        dicts.append(r)

    return dicts 

class BodyHandsDataset(Dataset):
    def __init__(self, is_train=True, target_size=(480, 640)):
        self.is_train = is_train
        self.target_size = target_size
        self.root = '../datasets/VOC2007'#_minimal'
        self.split = 'train' if is_train else 'test'
        self.class_names = ('hand', 'body')
        self.data_list = load_voc_instances(self.root, self.split, self.class_names)

    def __len__(self):
        return len(self.data_list)

    def our_transform(self, image, annotations):
        
        # Calculate the aspect ratio of the original images
        original_width, original_height = image.size[0], image.size[1]
        """aspect_ratio = original_width / original_height

        # Calculate the new dimensions while preserving the aspect ratio
        if original_width < original_height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)"""
        
        if self.is_train:
            self.transform = transforms.Compose([
            #transforms.Resize((new_height, new_width)),
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
            ])

        image = self.transform(image)
       
        new_annotations = []
        
        for i, annotation in enumerate(annotations):
            temp = {}
            bbox = copy.deepcopy(annotation['bbox'])
            # Resize bounding box coordinates
            """bbox[0] *= (new_width / original_width)
            bbox[1] *= (new_height / original_height)
            bbox[2] *= (new_width / original_width)
            bbox[3] *= (new_height / original_height)"""
            bbox[0] *= (self.target_size[1] / original_width)
            bbox[1] *= (self.target_size[0] / original_height)
            bbox[2] *= (self.target_size[1] / original_width)
            bbox[3] *= (self.target_size[0] / original_height)
            temp['bbox'] = bbox

            # Resize body box coordinates
            body_box = copy.deepcopy(annotation['body_box'])
            """body_box[0] *= (new_width / original_width)
            body_box[1] *= (new_height / original_height)
            body_box[2] *= (new_width / original_width)
            body_box[3] *= (new_height / original_height)"""
            body_box[0] *= (self.target_size[1] / original_width)
            body_box[1] *= (self.target_size[0] / original_height)
            body_box[2] *= (self.target_size[1] / original_width)
            body_box[3] *= (self.target_size[0] / original_height) 
            temp['body_box'] = body_box
            
            # Resize segmentation coordinates (assuming polygon coordinates)
            seg_annotations = []
            s = []
            for seg in annotation['segmentation']:
                for coord in seg:
                    """ x = coord[0] * new_width/original_width
                    y = coord[1] * new_height/original_height"""
                    x = coord[0] * self.target_size[1]/original_width
                    y = coord[1] * self.target_size[0]/original_height
                    s.append((x, y))
                seg_annotations.append(s)
            temp['segmentation'] = seg_annotations

            temp['category_id'] = annotation['category_id']
            temp['body_id'] = annotation['body_id']
            new_annotations.append(temp)

        return image, new_annotations
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_path = data['file_name']
        annotations = data['annotations']
        image = Image.open(image_path).convert('RGB')
        
        #print("old annotations: ", annotations)
        #print("")
        image, new_annotations = self.our_transform(image, annotations)
        #print("new annotations: ", new_annotations) 
        
                
        num_objs = len(new_annotations)
        labels = []
        bboxes = []
        seg_masks = []
        for i in range(num_objs):
            labels.append(new_annotations[i]['category_id']+1)
            bboxes.append(new_annotations[i]['bbox'])
            seg_masks.append(our_utils.generate_segmentation_mask(new_annotations[i]['segmentation']))
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        seg_masks = torch.stack(seg_masks, dim=0)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = seg_masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target #image, new_annotations, target



if __name__ == '__main__':
    dataset = BodyHandsDataset(is_train=False)
    #d = iter(dataset)
    i, a, t = dataset[10]
    output_path = './img_' + dataset.data_list[10]['image_id'] + '.png'
    #print("Sairam", output_path)
    our_utils.save_image_with_annotations(i, a, output_path)
