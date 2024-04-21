import os

"""os.system("curl -LO https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("curl -LO https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("curl -LO https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("curl -LO https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("curl -LO https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")"""

from engine import train_one_epoch, evaluate
from coco_eval import CocoEvaluator
import utils

import torch
import torchmetrics.detection.mean_ap as mean_ap
import torch.nn as nn
import cv2

import our_utils

from dataset import BodyHandsDataset
from model import get_model_segmentation
from visualizer import CustomVisualizer


def get_score(device):

    model = get_model_segmentation(num_classes)
    model_path = './checkpoints/best_model_epoch_99.pt'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    dataset_test = BodyHandsDataset(is_train=False)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)

    # Initialize the MeanAveragePrecision metric
    mAP_metric = mean_ap.MeanAveragePrecision(num_classes=2)

    # Compute mAP for each test image
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.no_grad():
            # Generate predictions using the detection model
            outputs = model(images)
        
        # Convert model outputs and targets to the appropriate format
        predictions = [{k: v.cpu().numpy() for k, v in output.items()} for output in outputs]
        targets = [{k: v.cpu().numpy() for k, v in target.items()} for target in targets]
        
        # Update mAP metric with predictions and targets for the current batch
        mAP_metric.update(predictions, targets)

    # Compute final mAP value
    mAP_value = mAP_metric.compute()
    print("mAP:", mAP_value)



def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has three classes only - background, hand, body
    num_classes = 3
    # use our dataset and defined transformations
    dataset = BodyHandsDataset()
    dataset_test = BodyHandsDataset(is_train=False)

    """# split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])"""

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10 
    best_model_ap = float('-inf')

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger= train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        #print("metric_logger: ", metric_logger)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #print("data_loader_test ka type: ", type(data_loader_test)) 

        for img_idx in range(len(data_loader_test)):
            img, targets = data_loader_test.dataset[img_idx]
            #print("targets: ", targets)
            #print("type of targets['image_id']", type(targets['image_id']))

        #coco_evaluator = evaluate(model, data_loader_test, device=device)
        #AP = coco_evaluator.coco_eval['bbox'].stats[0]
        # Get the COCO metrics from the evaluator
        # Save the best model
        #if AP > best_model_ap:
        #    best_model = model
        #    torch.save(best_model.state_dict(), "./checkpoints2/best_model_epoch_" + str(epoch) + ".pt")
        #elif epoch == num_epochs - 1:
        torch.save(model.state_dict(), "./checkpoints2/model_epoch_" + str(epoch) + ".pt")


    print("That's it!")

if __name__ == '__main__':
    #main()


    num_classes = 3
    iou_thresh = 0.2
    conf_thresh = 0.4
    im_size = (640, 480) # this is in (width, height) format

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = get_model_segmentation(num_classes)
    model_path = './checkpoints/best_model_epoch_99.pt'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    data_path = '../datasets/VOC2007/JPEGImages/'#_minimal/JPEGImages/'
    image_files = os.listdir(data_path)

    # Iterate through each image file
    for i, image_file in enumerate(image_files):
        # Check if the file is an image (you can customize the image extensions)
        if image_file.startswith('test') and image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(i)
            # Construct the full path to the image
            image_path = os.path.join(data_path, image_file)

            # Open the image using OpenCV
            image = cv2.imread(image_path)
            image = cv2.resize(image, im_size)
            # Check if the image was successfully loaded
            if image is not None:
                v= CustomVisualizer(image[:, :, ::-1], scale=1.0)
                results, output, top_predictions = our_utils.predict(image, model, device)
                boxes= top_predictions['boxes']
                classes = top_predictions['labels']
                
                """ body_ids = outputs.pred_body_ids """

                masks =top_predictions["masks"]
                hand_indices = classes == 1
                body_indices = classes == 2
                hand_boxes = boxes[hand_indices]
                body_boxes = boxes[body_indices] 

                # hand_masks = masks[hand_indices]
                # body_ids= body_indices # we should have pred_body_ids
                # hand_body_ids = body_ids[hand_indices]
                # body_body_ids = body_ids[body_indices]
                # num_hands, num_bodies = hand_boxes.shape[0], body_boxes.shape[0]
                # body_masks = []
                # for body_no in range(num_bodies):
                #     box = body_boxes[body_no].view(-1).cpu().numpy()
                #     xmin, ymin, xmax, ymax = box
                #     body_poly = [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]
                #     body_masks.append(body_poly)

                # hand_masks = []
                # for hand_no in range(num_hands):
                #     box = hand_boxes[hand_no].view(-1).cpu().numpy()
                #     xmin, ymin, xmax, ymax = box
                #     hand_poly = [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]
                #     hand_masks.append(hand_poly)
                pred_boxes= boxes 
                pred_classes= classes 
                handbody_components= our_utils.extract_handbody_components_inference(pred_boxes, pred_classes)
                # we dont have predicted_instances, pred_body_ids now
                pred_body_ids= {"hand_indices": [], "body_indices": []}
                pred_instances= [{"pred_body_ids": pred_body_ids}]

                pred_instances= our_utils.OverlapEstimationInference(handbody_components, pred_instances, device)


                body_ids = pred_instances[0]["pred_body_ids"]
                # body_ids= body_indices # we 
                hand_body_ids = body_ids[hand_indices]          
                body_body_ids = body_ids[body_indices]   

                num_hands, num_bodies = hand_boxes.shape[0], body_boxes.shape[0]
                body_masks = []
                for body_no in range(num_bodies):
                    box = body_boxes[body_no].view(-1).cpu().numpy()
                    xmin, ymin, xmax, ymax = box
                    body_poly = [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]
                    body_masks.append(body_poly)

                hand_masks = []
                for hand_no in range(num_hands):
                    box = hand_boxes[hand_no].view(-1).cpu().numpy()
                    xmin, ymin, xmax, ymax = box
                    hand_poly = [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]
                    hand_masks.append(hand_poly)


                v = v.modified_draw_instance_predictions(hand_boxes, hand_masks, hand_body_ids, body_boxes, body_masks, body_body_ids)
                out_img = v.get_image()[:, :, ::-1]
                # pred_instances[0].append(pred_body_ids) 
                #pred_instances= our_utils.OverlapEstimationInference(handbody_components, pred_instances, device)
                #print("in main, pred_instances: " ,pred_instances)
                # its visualisation time
                cv2.imwrite('./test_results/epoch_39/' + f'iou_{iou_thresh}_conf_{conf_thresh}' + image_file, out_img)
                #cv2.imwrite('./test_results/epoch_39/' + f'iou_{iou_thresh}_conf_{conf_thresh}' + image_file, results)



