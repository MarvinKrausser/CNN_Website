import sys

sys.path.insert(1, './src/yolo')
from train_yolo_faces import train_yolo

sys.path.insert(2, './src/bird_cnn')
from bird_cnn import Bird_CNN


def main():
    train_yolo()

if __name__ == "__main__":
    main()