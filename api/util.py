from PIL import Image
import os

folder = "data/train"

widths = []
heights = []

for root, _, files in os.walk(folder):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(root, file)
            img = Image.open(path)
            w, h = img.size
            widths.append(w)
            heights.append(h)

avg_w = sum(widths) / len(widths)
avg_h = sum(heights) / len(heights)

print("Average width:", avg_w)
print("Average height:", avg_h)