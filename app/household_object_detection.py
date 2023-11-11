import cv2
import numpy as np
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from keras_retinanet import models
from app.helper.visualize import visualize_image
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

model_path = 'app/model/object_detection.h5'
model = models.load_model(model_path, backbone_name='resnet50')

model_labels = {0: 'Bookcase', 1: 'Bathtub', 2: 'Pillow', 3: 'Couch', 4: 'Gas stove', 5: 'Washing machine', 6: 'Bed',
                7: 'Refrigerator', 8: 'Bathroom accessory', 9: 'Kitchen & dining room table', 10: 'Television',
                11: 'Sink', 12: 'Sofa bed', 13: 'Kitchenware', 14: 'Toilet', 15: 'Ceiling fan', 16: 'Microwave oven',
                17: 'Furniture', 18: 'Coffeemaker', 19: 'Cupboard', 20: 'Dishwasher', 21: 'Shower', 22: 'Clock',
                23: 'Countertop', 24: 'Mug', 25: 'Table'}

room_labels = {'kitchen': ['Gas stove', 'Refrigerator', 'Kitchen & dining room table', 'Sink', 'Kitchenware',
                           'Microwave oven', 'Coffeemaker', 'Dishwasher', 'Countertop', 'Mug'],
               'BedRoom': ['Bed', 'Bookcase', 'Pillow', 'Couch', 'Television', 'Sofa bed', 'Ceiling fan', 'Furniture',
                           'Cupboard', 'Clock', 'Table'],
               'Bathroom': ['Bathtub', 'Washing machine', 'Bathroom accessory', 'Toilet', 'Shower']
               }


def categorize_entity(entity):
    for category, items in room_labels.items():
        if entity in items:
            return category
    return "Unknown"


def show(image_path):
    predicted_items = []
    image = read_image_bgr(image_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):

        if score < 0.5:
            break

        visualize_image(draw, box, score, label)
        predicted_items.append({'predicted_label': model_labels[label],
                                'score': score})

    if predicted_items:
        predicted_rooms = []

        for i in predicted_items:
            category = categorize_entity(i['predicted_label'])
            predicted_rooms.append(category)

        counts = Counter(predicted_rooms)

        room = counts.most_common(1)[0][0]

        st.json({'model_predictions': predicted_items,
                 'room': room})

    else:
        st.json({'model_predictions': None})

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


st.title("Household Object Detection")
uploaded_image = st.file_uploader("Choose a image file", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    show(uploaded_image)
