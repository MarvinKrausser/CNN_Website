import os

import cv2
from tqdm import tqdm
from yolo_dataset import YoloDataset
from yolo_model import train, Yolo_model, sample
from yolo_loss import YoloLoss
from util import TransformedSubset, test_workers_speed, visualizeImage
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F

from yolo_model_production import convert_prediction

def view_data(dataset):
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for images, labels in iter(dataloader):
        for batch in range(images.shape[0]):
            image = images[batch]
            label = labels[batch]
            visualize_boxes(label, image)

def sample_data(dataloader, model, device, SAVE_PATH):
    for images, labels in iter(dataloader):
        predictions = sample(model, images, device, SAVE_PATH)
        for batch in range(predictions.shape[0]):
            image = images[batch]
            prediction = predictions[batch]           
            visualize_boxes(prediction, image)

def visualize_boxes(label, image, threshold=0.95):
    boxes_to_draw, grids_to_draw_obj, grids_to_draw_noobj = convert_prediction(label, image, threshold)
    if len(boxes_to_draw) == 0:
        print("no labels")
        return
    boxes_to_draw = torch.tensor(boxes_to_draw)
    grids_to_draw_noobj = torch.tensor(grids_to_draw_noobj)
    grids_to_draw_obj = torch.tensor(grids_to_draw_obj)
    image = draw_bounding_boxes(image, grids_to_draw_noobj, colors=(0, 255, 0))
    image = draw_bounding_boxes(image, grids_to_draw_obj, colors=(0, 0, 255))
    image = draw_bounding_boxes(image, boxes_to_draw, colors=(255, 0, 0))
    visualizeImage(image)

def use_webcam(grid, img_size):
    # 0 = default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open webcam")

    model = Yolo_model(c_in=3, boxes=1, grid=grid, labels=1)
    state_dict = torch.load(os.path.join("saved_models", "face_detection_yolo", "face_detection_yolo"), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(rgb).permute(2, 0, 1).float()
        C, H, W = image.shape
        scale_w = W / img_size
        scale_h = H / img_size

        image = image / 255.0
        
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=True) #different modes?

        prediction = model(image)

        bboxes, grid_ob, grid_noob = convert_prediction(prediction.squeeze(0), image.squeeze(0), threshold=0.9)

        for bbox in grid_noob:
            xmin = int(bbox[0] * scale_w)
            ymin = int(bbox[1] * scale_h)
            xmax = int(bbox[2] * scale_w)
            ymax = int(bbox[3] * scale_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        for bbox in grid_ob:
            xmin = int(bbox[0] * scale_w)
            ymin = int(bbox[1] * scale_h)
            xmax = int(bbox[2] * scale_w)
            ymax = int(bbox[3] * scale_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

        for bbox in bboxes:
            xmin = int(bbox[0] * scale_w)
            ymin = int(bbox[1] * scale_h)
            xmax = int(bbox[2] * scale_w)
            ymax = int(bbox[3] * scale_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_yolo():
    SAVE_PATH = "./saved_models"
    IMAGE_SIZE = 64
    GRID = 6
    BATCH_SIZE = 64

    dataset = YoloDataset(
        image_dir="data/faces_2/train", 
        annotation_path="data/faces_2/train/_annotations.coco.json",
        img_size=IMAGE_SIZE,
        transform=True,
        grid=GRID
    )

    dataset_valid = YoloDataset(
        image_dir="data/faces_2/test", 
        annotation_path="data/faces_2/test/_annotations.coco.json",
        img_size=IMAGE_SIZE,
        transform=True,
        grid=GRID
    )


    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    model = Yolo_model(c_in=3, boxes=1, grid=GRID, labels=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_module = YoloLoss()


    #view_data(dataset)
    #exit()


    #sample_data(dataloader=val_loader, model=model, device=device, SAVE_PATH=SAVE_PATH)
    #exit()


    #test_workers_speed(dataset, model)
    #exit()


    use_webcam(GRID, IMAGE_SIZE)
    exit()


    train(model=model, loss_module=loss_module, train_loader=train_loader, val_loader=val_loader, 
                optimizer=optimizer, SAVE_PATH=SAVE_PATH, saving=True, model_name="face_detection_yolo")