import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    @staticmethod
    def is_center_in_grid_cell(x, y, img_w, img_h, S, cell_i, cell_j):
        cell_w = img_w / S
        cell_h = img_h / S

        # find which cell the center belongs to
        gt_cell_i = int(y / cell_h)
        gt_cell_j = int(x / cell_w)

        return (gt_cell_i == cell_i) and (gt_cell_j == cell_j)
    
    @staticmethod
    def save_sqrt(i):
        return torch.sqrt(torch.clamp(i, min=1e-6))
    
    @staticmethod
    def iou(boxA, boxB):
        xA = torch.max(boxA[0], boxB[0])
        yA = torch.max(boxA[1], boxB[1])
        xB = torch.min(boxA[2], boxB[2])
        yB = torch.min(boxA[3], boxB[3])

        inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = boxA_area + boxB_area - inter_area

        return inter_area / (union + 1e-6)
    
    @staticmethod
    def xywh_to_xyxy(box):
        x, y, w, h = box

        return torch.stack([
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ])

    def forward(self, pred, target, labels, cell_size, img_w, img_h, factor_no_object = 0.5, factor_object = 5):
        #pred: [SxSx(B(x,y,w,h,con)C)]
        #target: [B(x, y, w, h, C)]
        
        for s0 in range(pred.shape[0]):
            for s1 in range(pred.shape[1]):
                boxes = pred[s0, s1, :(-1*labels)]
                pred_label = pred[s0, s1, -1:]

                ground_truth_box = None
                for b in range(target.shape[0] // (4 + labels)):
                    box = target[(b)*(4 + labels):(b+1)*(4 + labels)]
                    if self.is_center_in_grid_cell(box[0]+box[2]/2, box[1]+box[3]/2, img_w, img_h, cell_size, s0, s1):
                        ground_truth_box = box[:4]
                        ground_truth_label = box[4:]
                        break


                coord_loss = torch.tensor(0., device=pred.device)
                size_loss = torch.tensor(0., device=pred.device)
                prob_loss = torch.tensor(0., device=pred.device)
                label_loss = torch.tensor(0., device=pred.device)

                if ground_truth_box is not None:

                    iou_stats = []

                    for b in range(boxes.shape[0] // 5):
                        box = boxes[(b)*5:(b+1)*5]

                        iou_stats.append(self.iou(self.xywh_to_xyxy(box[0:4]), self.xywh_to_xyxy(ground_truth_box[0:4])))
                    iou_tensor = torch.stack(iou_stats)
                    responsible = torch.argmax(iou_tensor)

                    for b in range(boxes.shape[0] // 5):
                        box = boxes[(b)*5:(b+1)*5]

                        is_responsible = (b == responsible).float()

                        coord_loss += factor_object * is_responsible * ((box[0] - ground_truth_box[0])**2 + (box[1] - ground_truth_box[1])**2)
                        size_loss += factor_object * is_responsible * ((self.save_sqrt(box[2]) - self.save_sqrt(ground_truth_box[2]))**2 + 
                                                       (self.save_sqrt(box[3]) - self.save_sqrt(ground_truth_box[3]))**2)
                        
                        prob_loss += is_responsible * (box[4] - 1)**2
                        prob_loss += factor_no_object * (1 - is_responsible) * box[4]**2


                    for i in range(pred_label.shape[0]):
                        label_loss += ((pred_label[i] - ground_truth_label[i])**2) / pred_label.shape[0]
                        

                else:
                    for b in range(boxes.shape[0] // 5):
                            box = boxes[(b)*5:(b+1)*5]

                            prob_loss += factor_no_object * box[4] ** 2

        return (label_loss + prob_loss + coord_loss + size_loss) / pred.shape[0]**2
