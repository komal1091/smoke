from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO('/home/codezeros/Documents/fire&smoke detection/Test/best.pt')

img = Image.open('/home/codezeros/Documents/fire&smoke detection/Test/267.jpg' )

img2 = Image.open('/home/codezeros/Documents/fire&smoke detection/Test/smoke10.jpeg')

results = model.predict(img)


for result in results:
    x = result.boxes.xyxy
    print(len(x))

    if len(x) == 1:
        result.show()






