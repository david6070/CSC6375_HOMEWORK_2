"""
CS 6375 Homework 2 Programming
Implement the compute_loss() function in this python script
"""
import os
import torch
import torch.nn as nn


# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou

# TODO: finish the implementation of this loss function for YOLO training
# output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# num_boxes: number of bounding boxes per cell
# num_classes: number of object classes for detection
# grid_size: YOLO grid size, 64 in our case
# image_size: YOLO image size, 448 in our case
def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]

    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                if gt_mask[i, j, k] > 0:
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1
                    for b in range(num_boxes):
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou
                    print('select box %d with iou %.2f' % (select, max_iou))

    weight_coord = 5.0
    weight_noobj = 0.5

    loss_x = 0
    loss_y = 0
    loss_w = 0
    loss_h = 0
    loss_obj = 0
    loss_noobj = 0
    loss_cls = 0

    for b in range(num_boxes):
        base = 5 * b
        mask = box_mask[:, b, :, :]

        # Coordinate losses
        loss_x += weight_coord * torch.sum(mask * torch.pow(gt_box[:, 0] - output[:, base + 0], 2.0))
        loss_y += weight_coord * torch.sum(mask * torch.pow(gt_box[:, 1] - output[:, base + 1], 2.0))

        # Width and height: use square root as in YOLO paper
        loss_w += weight_coord * torch.sum(mask * torch.pow(torch.sqrt(gt_box[:, 2]) - torch.sqrt(torch.clamp(output[:, base + 2], min=1e-6)), 2.0))
        loss_h += weight_coord * torch.sum(mask * torch.pow(torch.sqrt(gt_box[:, 3]) - torch.sqrt(torch.clamp(output[:, base + 3], min=1e-6)), 2.0))

        # Object confidence loss (already implemented partially)
        loss_obj += torch.sum(mask * torch.pow(box_confidence[:, b] - output[:, base + 4], 2.0))

        # No-object confidence loss
        noobj_mask = 1 - mask
        loss_noobj += weight_noobj * torch.sum(noobj_mask * torch.pow(0 - output[:, base + 4], 2.0))

    # Classification loss (only one class: cracker_box = index 0)
    if num_classes > 0:
        pred_cls = output[:, 5*num_boxes:, :, :]  # shape: (batch, num_classes, 7, 7)
        gt_cls = torch.zeros_like(pred_cls)
        gt_cls[:, 0, :, :] = gt_mask  # class 0 (cracker box) only where gt_mask is 1
        loss_cls = torch.sum(torch.pow(gt_cls - pred_cls, 2.0))

    # Total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls

    print('lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' %
          (loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj, loss_cls))

    return loss

