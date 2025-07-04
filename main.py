import cv2
import pandas as pd
import yaml

from ultralytics import YOLO

model = YOLO('weights/best.pt')
while True:

    img_path = input("Enter the path of your image : ")
    img = cv2.imread(img_path)

    results = model.predict(img, conf = 0.6)

    result = results[0].boxes.cls

    #to list of tensors
    item = result.tolist()
    items = set()

    with open("data/data.yaml", 'r') as classes:
        out = yaml.safe_load(classes)
        for i in item:
            i = int(i)
            items.add(out['names'][i])

    df = pd.read_csv('data/nutrition.csv')
    df.set_index('name', inplace=True)

    try:
        for i in items:
            print(i)
            print(df.loc[i])
            print('\n')
    except:
        print("food not found...")

    results[0].show()