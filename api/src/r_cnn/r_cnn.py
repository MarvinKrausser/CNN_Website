import os
import cv2
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn as nn
from util import iou, visualizeImage
import random
import torch.nn.functional as F


class ObjectDetectionCNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, layers):
        super().__init__()

        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
                nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hidden),
                nn.ReLU(inplace=True)
            ))

        for _ in range(layers-1):
            self.model.append(nn.Sequential(
                nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hidden),
                nn.ReLU(inplace=True)
            ))

        self.model.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, c_out),
            nn.Dropout(0.3)
        ))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    


def random_crop(W, H, sizeX=64, sizeY =64):
    x = random.randint(0, max(W - sizeX, 0))
    y = random.randint(0, max(H - sizeY, 0))
    return x, y, x + sizeX, y + sizeY

def is_background(crop, gt_boxes, threshold=0.0):
    for box in gt_boxes:
        if iou(crop, box) > threshold:
            return False
    return True

def get_background_crop(image, gt_boxes, max_trials=100):
    C, H, W = image.shape

    for _ in range(max_trials):
        x1, y1, x2, y2 = random_crop(W, H, sizeX=random.randint(50, 600), sizeY=random.randint(50, 600))

        crop_box = (x1, y1, x2, y2)

        if is_background(crop_box, gt_boxes, threshold=0.1):
            crop = image[:, y1:y2, x1:x2]
            return crop

    return None

def create_background_tensor(amount, dataset, labels, boxes, image):
    for _ in range(amount):
                    background = get_background_crop(image=image, gt_boxes=boxes.to(torch.int64))
                    if background is not None:
                        background = TF.resize(background, [64, 64], antialias=True)
                        dataset.append(background)
                        labels.append(torch.tensor(0))
                    else:
                         print("Background not found")
    return dataset, labels

def create_stack(images, annotations, PADDING):
    croped_images = []
    labels = []
    for i in range(len(images)):
        image = images[i]
        annotation = annotations[i]
        _, h, w = image.shape
        
        for box, label in tuple(zip(annotation["boxes"], annotation["labels"])):
            box = box.to(torch.int64)
            box_copy = []

            box_copy.append(max(box[0]-PADDING, 0))
            box_copy.append(max(box[1]-PADDING, 0))
            box_copy.append(min(box[2]+PADDING, w-1))
            box_copy.append(min(box[3]+PADDING, h-1))
            

            croped_image = image[: , box_copy[1]:box_copy[3], box_copy[0]:box_copy[2]]
            #visualizeImage(croped_image)
            croped_image = TF.resize(croped_image, [64, 64], antialias=True)
            #visualizeImage(croped_image)
            croped_images.append(croped_image)
            labels.append(label)

        #croped_images, labels = create_background_tensor(amount=len(croped_images), dataset=croped_images, labels=labels, boxes=annotation["boxes"], image=image)
    return croped_images, labels

def train(model, loss_module, train_loader, val_loader, optimizer, SAVE_PATH, model_name, saving=True, PADDING=20, device="cpu"):
    best_val = torch.finfo(torch.float32).max

    for epoch in range(200):
        ############
        # Training #
        ############
        model.train()

        true_preds, count, lossCount = 0, 0, 0.
        for images, annotations in tqdm(train_loader, desc=f"Train", leave=False):
            croped_images, labels = create_stack(images, annotations, PADDING)

            croped_images = torch.stack(croped_images).to(device)
            labels = torch.stack(labels).to(device)

            prediction = model(croped_images)

            loss = loss_module(prediction, labels)
            lossCount += loss.sum().item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            true_preds += (prediction.argmax(dim=1) == labels).sum().item()
            count += croped_images.size(0)

        train_acc = true_preds / count
        train_loss = lossCount / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        true_preds, count, lossCount = 0, 0, 0.
        for images, annotations in tqdm(val_loader, desc=f"Test", leave=False):
            with torch.no_grad():
                croped_images, labels = create_stack(images, annotations, PADDING)

                croped_images = torch.stack(croped_images).to(device)
                labels = torch.stack(labels).to(device)

                prediction = model(croped_images)

                loss = loss_module(prediction, labels)
                lossCount += loss.sum().item()

                true_preds += (prediction.argmax(dim=1) == labels).sum().item()
                count += croped_images.size(0)

        val_acc = true_preds / count
        val_loss = lossCount / count

        if(saving and best_val > val_loss):
                best_val = val_loss
                save_dir = os.path.join(SAVE_PATH, model_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | validation accuracy: {int(val_acc * 1000) / 10}% | train loss: {int(train_loss * 1000) / 100} | val loss: {int(val_loss * 1000) / 100}")
        torch.cuda.empty_cache()
    return best_val


def trainNormalDataset(model, loss_module, train_loader, val_loader, optimizer, SAVE_PATH, model_name, saving=True, device="cpu"):
    best_val = torch.finfo(torch.float32).max

    for epoch in range(200):
        ############
        # Training #
        ############
        model.train()

        true_preds, count, lossCount = 0, 0, 0.
        for images, labels in tqdm(train_loader, desc=f"Train", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)

            loss = loss_module(prediction, labels)
            lossCount += loss.sum().item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            true_preds += (prediction.argmax(dim=1) == labels).sum().item()
            count += images.size(0)

        train_acc = true_preds / count
        train_loss = lossCount / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        true_preds, count, lossCount = 0, 0, 0.
        for images, labels in tqdm(val_loader, desc=f"Test", leave=False):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                prediction = model(images)

                loss = loss_module(prediction, labels)
                lossCount += loss.sum().item()

                true_preds += (prediction.argmax(dim=1) == labels).sum().item()
                count += images.size(0)

        val_acc = true_preds / count
        val_loss = lossCount / count

        if(saving and best_val > val_loss):
                best_val = val_loss
                save_dir = os.path.join(SAVE_PATH, model_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | validation accuracy: {int(val_acc * 1000) / 10}% | train loss: {int(train_loss * 1000) / 100} | val loss: {int(val_loss * 1000) / 100}")
        torch.cuda.empty_cache()
    return best_val

def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / new_w, h / new_h

def map_box_to_original(box, scale_x, scale_y):
    x1, y1, x2, y2 = box

    return int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

def eval(model, image, BUILD_PATH, device, PADDING = 0, minSize=5, maxSize=600, minConf=0.8):
    _, H, W = image.shape

    model.load_state_dict(torch.load(BUILD_PATH, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    image_numpy = image.permute(1, 2, 0).cpu().numpy()
    image_numpy = (image_numpy*255).astype("uint8")
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    image_numpy_copy, resizedW, resizedH = resize_keep_aspect(image_numpy, W, H)

    cv2.imshow("", image_numpy_copy)
    cv2.waitKey(0)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_numpy_copy)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    draw = image_numpy.copy()
    predictions = []

    for (x, y, w, h) in tqdm(rects):
        x, y, w, h = map_box_to_original((x, y, w, h), resizedW, resizedH)

        if w < minSize or h < minSize or w > maxSize or h > maxSize:
            continue

        x1 = max(x-PADDING, 0)
        y1 = max(y-PADDING, 0)
        x2 = min(x+w+PADDING, W-1)
        y2 = min(y+h+PADDING, H-1)

        crop = image[:, y1:y2, x1:x2]
        crop = TF.resize(crop, [64, 64], antialias=True)
        crop = crop.unsqueeze(0).to(device)

        pred = model(crop)
        probs = F.softmax(pred, dim=1)
        confidence, cls = torch.max(probs, dim=1)
        if cls.item() == 1 and confidence.item() > minConf:
            predictions.append((confidence.item(), cls.item(), (x, y, w, h)))

    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
    print(len(predictions))
    for conf, cls, (x, y, w, h) in predictions[:]:
        cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow("", draw)
    cv2.waitKey(0)