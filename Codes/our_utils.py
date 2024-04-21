import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from pycocotools.coco import COCO
import time
import copy
import json
from scipy.optimize import linear_sum_assignment # for bipartite mapping

from typing import Tuple, List
import math
from torch.nn import functional as F

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def loadRes(cocoObj, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in cocoObj.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        #elif type(resFile) == np.ndarray:
        #    anns = cocoObj.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]

        print("set(annsImgIds): ", set(annsImgIds))
        print()
        print("set(cocoObj.getImgIds): ", set(cocoObj.getImgIds()))

        assert set(annsImgIds) == (set(annsImgIds) & set(cocoObj.getImgIds())), \
              'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res





def generate_segmentation_mask(segmentation_annotation, im_size=(480, 640)):
  """Generates a segmentation mask from a list of lists of tuples of x, y coordinates.

  Args:
    segmentation_annotation: A list of lists of tuples of x, y coordinates.
    im_size: a tuple (h, w) denoting mask/image size

  Returns:
    A segmentation mask tensor.
  """
  
  mask = np.zeros(im_size, dtype=np.uint8)
  for polygon in segmentation_annotation:
    polygon = np.array(polygon).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
  return torch.tensor(mask, dtype=torch.uint8)

def save_image_with_annotations(image, annotations, output_path):
    # Create a PIL Image from the torch tensor
    image_pil = transforms.ToPILImage()(image)

    image = image.permute(1, 2, 0).numpy()

    # Create a drawing context to superimpose annotations on the image
    draw = ImageDraw.Draw(image_pil)

    for annotation in annotations:
        # Draw the bounding box on the image
        draw.rectangle(annotation['bbox'], outline="blue", width=2)
        seg_mask = generate_segmentation_mask(annotation['segmentation'], (image.shape[0], image.shape[1]))
        #seg_mask = seg_mask[:, :, None]
        #seg_mask = np.repeat(seg_mask, 3, axis=2)
        
        # Define the color for the overlay (e.g., red: [0, 0, 255] for BGR format)
        overlay_color = [0, 0, 255]

        # Create a white mask for the segmentation area
        white_mask = np.zeros((image_pil.size[1], image_pil.size[0], 3), dtype=np.float32)
        white_mask[:, :, 0][seg_mask == 1] = overlay_color[0]
        white_mask[:, :, 1][seg_mask == 1] = overlay_color[1]
        white_mask[:, :, 2][seg_mask == 1] = overlay_color[2]

        # Superimpose the colored mask on the RGB image
        superimposed_image = cv2.addWeighted(image, 1., white_mask, 0.7, 0)
        image = superimposed_image

    image_with_annotations = np.array(image_pil)
    # Save or display the superimposed image
    cv2.imwrite(output_path, cv2.cvtColor(image_with_annotations, cv2.COLOR_RGB2BGR))
    image = (image*255).astype(np.uint8)
    cv2.imwrite(output_path[:-4] + '_seg.png', image)#cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1) ## idx is a 1-d bitmask of indices of images that crossed the threshold

    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx] ## select the k,v only for those images that satisfied the cutoff score
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    #print("in overlay_boxes, labels: ", labels)
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def overlay_keypoints(image, predictions):
    kps = predictions["keypoints"]
    scores = predictions["keypoints_scores"]
    kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
    for region in kps:
        image = vis_keypoints(image, region.transpose((1, 0)))
    return image

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


CATEGORIES = """BACKGROUND
hand
body
""".split("\n")

class PersonKeypoints(object):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions['boxes']

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (int(x.item()), int(y.item())), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def predict(img, model, device, conf_thresh=0.4, iou_thresh=0.2):
    model.eval()
    cv_img = np.array(img) ##[:, :, [2, 1, 0]] ## this 2,1,0 rearranges the colors
    img_tensor = torchvision.transforms.functional.to_tensor(img)
    with torch.no_grad():
        output = model([img_tensor.to(device)])
       #print("output of the model: ", output)
    top_predictions = select_top_predictions(output[0], conf_thresh)
    

    # Perform NMS to filter out duplicate boxes
    filtered_indices = torchvision.ops.nms(top_predictions['boxes'], top_predictions['scores'], iou_thresh)

    # Create a new dictionary with filtered boxes and scores
    top_predictions = {
        'boxes': top_predictions['boxes'][filtered_indices],
        'labels': top_predictions['labels'][filtered_indices],
        'scores': top_predictions['scores'][filtered_indices],
        'masks': top_predictions['masks'][filtered_indices]
    }
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
    result = cv_img.copy()
    result = overlay_boxes(result, top_predictions)
    if 'masks' in top_predictions:
        result = overlay_mask(result, top_predictions)
    """if 'keypoints' in top_predictions:
        result = overlay_keypoints(result, top_predictions)"""
    result = overlay_class_names(result, top_predictions)
    return result, output, top_predictions


################################# YEE HAW ##################################
from utils import Boxes
from scipy.optimize import linear_sum_assignment # for bipartite mapping

def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:  
    """
    Calculate the pairwise intersection area between two sets of bounding boxes.

    Args:
        boxes1 (Boxes): Bounding boxes represented as Boxes object.
        boxes2 (Boxes): Another set of bounding boxes represented as Boxes object.

    Returns:
        torch.Tensor: A 2D tensor containing the pairwise intersection areas between boxes1 and boxes2.
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection

def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Calculate the pairwise Intersection over Area (IOA) between two sets of bounding boxes.

    Args:
        boxes1 (Boxes): Bounding boxes represented as Boxes object.
        boxes2 (Boxes): Another set of bounding boxes represented as Boxes object.

    Returns:
        torch.Tensor: A 2D tensor containing the pairwise IOA values between boxes1 and boxes2.
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device))
    return ioa

#def extract_handbody_components_inference(pred_box_features, pred_boxes, pred_classes):
def extract_handbody_components_inference(pred_boxes, pred_classes):

    hand_indices = pred_classes == 1
    body_indices = pred_classes == 2
    hand_boxes = pred_boxes[hand_indices]
    body_boxes = pred_boxes[body_indices]
    # hand_features = pred_box_features[hand_indices]
    # body_features = pred_box_features[body_indices]
    gt_ioa = pairwise_ioa(Boxes(body_boxes), Boxes(hand_boxes)).T
    handbody_components = {
    "hand_boxes": hand_boxes,
    "body_boxes": body_boxes,
    "hand_indices": hand_indices,
    "body_indices": body_indices,
    # "hand_features": hand_features,
    # "body_features": body_features,
    "gt_ioa": gt_ioa,
    }
    return handbody_components



@torch.jit.script
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp


    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas


    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return pred_boxes.reshape(deltas.shape) 

def OverlapEstimationInference(handbody_components, pred_instances, device):

    """
    Perform overlap estimation and association of hand and body instances.

    This function takes predicted instances of hands and bodies, along with associated components,
    and estimates the overlap between them to determine the association of hands with bodies.

    Args:
        cfg (CfgNode): Model configuration parameters.
        handbody_components (dict): A dictionary containing various components related to hand and body detection.
            - num_hands (int): Number of detected hand instances.
            - num_bodies (int): Number of detected body instances.
            - hand_indices (Tensor): Indices of hand instances.
            - body_indices (Tensor): Indices of body instances.
            - gt_ioa (Tensor): Ground truth overlap information between hands and bodies.
        pred_instances (list of Instances): Predicted instances containing hand and body detections.
        device (str): The device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        pred_instances (list of Instances): Updated predicted instances with assigned body IDs.

    Notes:
        This function estimates the association of hands with bodies based on overlap scores and positional density.
        It handles cases where there are no hands or no bodies detected and assigns unique IDs accordingly.
    """

    #num_hands = handbody_components["num_hands"]
    num_hands = handbody_components["hand_boxes"].shape[0]
    num_bodies = handbody_components["body_boxes"].shape[0]
    #num_bodies = handbody_components["num_bodies"]
    hand_indices = handbody_components["hand_indices"]
    body_indices = handbody_components["body_indices"]
    gt_overlap = (handbody_components["gt_ioa"] > 0).float()

    ## Here edge cases are being handled
    if num_hands == 0:
        ## if there are no hands, assign body_ids to all body instances are return
        pred_instances[0]["pred_body_ids"] = torch.Tensor([i for i in range(1, num_bodies+1)]).to(device)
        return pred_instances

    if num_bodies == 0:
        ## if there are no bodies, assign unique body_ids to all hand instances and return
        pred_instances[0]["pred_body_ids"] = torch.Tensor([num_bodies] * num_hands).to(device)


    ##non trivial case:
    pred_body_ids = torch.Tensor([-1.0] * (num_hands+num_bodies)).to(device)
    pred_hand_boxes = handbody_components["hand_boxes"]
    pred_body_boxes = handbody_components["body_boxes"]
#    pred_mu = handbody_components["pred_mu"]
#    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    pred_mu = handbody_components["hand_boxes"] # FIXME: for now, we are taking pred_hand_box== pred_mu
    box2box_transform = Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0))    

    ## mean difference bw bbox postions of pred_hand_boxes, pred_mu:
    mu_hand = box2box_transform.get_deltas( pred_hand_boxes, pred_mu )
    ## This is how get_deltas brother looks like:
    #  def get_deltas(self, src_boxes, target_boxes):


    mu_body = [] # A list of length num_hands, contains body candidates for a hand
    scores_positional_density = []
    for hand_no in range(num_hands):
        ## for each hand, do the following:
        hand_boxes_hand_no = pred_hand_boxes[hand_no:hand_no+1] ## get the predicted hand_box tensors of hand[hand_no]
        # new_pred_body_boxes = torch.cat([pred_body_boxes, hand_boxes_hand_no], dim=0) ## add(concatenate) the obtained hand_box tensors to body_box tensors
        new_pred_body_boxes = pred_body_boxes
        hand_boxes_hand_no = hand_boxes_hand_no.repeat(num_bodies, 1) ## Repeats the hands num_bodies+1 times to mach each hand against each body
        ## box deltas (differences) between the hand boxes and body boxes:
        mu_body_hand_no = box2box_transform.get_deltas(hand_boxes_hand_no, new_pred_body_boxes) # (num_bodies+1, 4)
        ## we'll get index error since get_delta expects only 1-d tensor

        ## box deltas (differences) between the hand boxes and hand boxes:
        mu_hand_hand_no = mu_hand[hand_no:hand_no+1].repeat(num_bodies, 1) # (Num_bodies+1, 4)
        ## repetetion will help us use Hungarian algo(linear_sum_assignment) later

        ## confindence_score in ?? mean hand position and mean body position ?? yes. looks like it
        ## but mu_hand_hand_no is a delta, ie a difference, so how does this make sense???
        conf_hand_no = torch.exp(-2.0 * 1e-1 * torch.sum(torch.abs(mu_hand_hand_no - mu_body_hand_no), dim=1))

        ## the confidences is updated as positional density for the hand
        scores_positional_density.append(conf_hand_no.reshape(1, num_bodies))
        mu_body.append(mu_body_hand_no) ## update the mean body position for the hand
    ## loop ends

    ## create a single tensor(scores_positional_density) where the contents of all the tensors in the list are stacked
    ## on top of each other along the first dimension
    
    scores_positional_density = torch.cat(scores_positional_density, dim=0)
    #print("size of the same:",scores_positional_density.size())
    # pred_overlap = handbody_components["pred_overlap"]
    pred_overlap = gt_overlap
    pred_overlap = F.sigmoid(pred_overlap)
    overlap_mask = (pred_overlap > 0.1).float() ## overlap_mask= 0 if pred_overlap<= 0.1
    #print("predoverlap: ",pred_overlap)
    #print("scores-post.desn:", scores_positional_density)
    #print("overlap_mask: ", overlap_mask)

    scores = pred_overlap * scores_positional_density * overlap_mask

    scores = torch.cat([scores, scores], dim=1) ## make it a "2-d square matrix" to use hungrian algo on it
    scores_numpy = scores.detach().to("cpu").numpy() ## transfer to cpu, as numpy-array
    row_ind, col_ind = linear_sum_assignment(-scores_numpy) ## minus to get the max score and not the min score

    """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html"""

    col_ind = (col_ind % (num_bodies+1)) + 1 ## index back to 1 from 0
    row_ind, col_ind = torch.from_numpy(row_ind).to(device),\
    torch.from_numpy(col_ind).to(device) ## transfer back to device(cuda)

    pred_body_ids_for_bodies = torch.arange(1, num_bodies+1).to(device)
    pred_body_ids_for_hands = torch.FloatTensor([num_bodies+1] * num_hands).to(device) ## a row tensor
    pred_body_ids_for_hands[row_ind] = col_ind.float() ## match the hand indices with the respective body indices
    pred_body_ids[hand_indices] = pred_body_ids_for_hands
    pred_body_ids[body_indices] = pred_body_ids_for_bodies.float()

    #pred_instances[0].pred_body_ids = pred_body_ids
    pred_instances[0]["pred_body_ids"] = pred_body_ids
    #print("our_utils: pred_instances[0]: ", pred_instances[0])
    #assert False 

    return pred_instances

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def PositionalDensityInference(device, pred_mu_deltas, pred_instances):
    pred_boxes = []
    pred_classes = []
    for instances_per_image in pred_instances:
        if len(instances_per_image) == 0: ## if there are no instances fro the image, move on to the next one
            continue
        pred_boxes.append(instances_per_image.pred_boxes.tensor) ## note the conversion to tensor
        pred_classes.append(instances_per_image.pred_classes) # Assumes batchsize is 1
    if pred_boxes:
        pred_boxes = cat(pred_boxes, dim=0)
        pred_classes = cat(pred_classes, dim=0) ## concatenate the boxes and classes together

    else: ## there are no predicted boxes
        pred_boxes = torch.empty(0, 4).to(device)
        pred_classes = torch.empty(0).to(device)
    pred_hand_boxes = pred_boxes[pred_classes==0] ## class==0 means hand

    ## if there are no hand boxes, intialise them, create new ones in the device of the model
    if not pred_hand_boxes.shape[0]:
        pred_hand_boxes = torch.empty(0, 4).to(device)
    box2box_transform = Box2BoxTransform(weights=(0.01, 0.01, 0.01, 0.01))

    ## why update the predicted positions here in the inference stage???
    ## is it to take care of the deltas that remain in the last epoch during training?
    pred_mu = box2box_transform.apply_deltas(pred_mu_deltas,pred_hand_boxes,)

    return pred_instances, pred_mu

# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit 

# Annotate boxes as Tensor (but not Boxes) in order to use scripting
@torch.jit.script_if_tracing
def paste_masks_in_image(
    masks: torch.Tensor, boxes: torch.Tensor, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()

