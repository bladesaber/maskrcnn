import cv2
import numpy as np
import os
import datetime
from Utils.sample_utils import color_splash
from Config.CustomConfig import CoCoInferenceConfig
from model.MaskRcnn import MaskRCNN
from Utils.utils import resize_image
import matplotlib.pyplot as plt
from Utils import visualize
import skimage.io

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def video_test(model):
    video_path = 'D:\\coursera\\maskrcnn\\test.mp4'

    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    # file_name = os.path.join(ROOT_DIR, "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
    # vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    images_list = []

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()

        if success:
            # if success:
            #     cv2.imshow('cap video', image)

            # if cv2.waitKey(40) & 0xFF == ord('q'):
            #     break

            images_list.append(image)

            count += 1
            if count >= 0:
                break

    for image in images_list:
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]

        # image, window, scale, padding, crop = resize_image(
        #     image,
        #     min_dim=config.IMAGE_MIN_DIM,
        #     min_scale=config.IMAGE_MIN_SCALE,
        #     max_dim=config.IMAGE_MAX_DIM,
        #     mode=config.IMAGE_RESIZE_MODE)

        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # RGB -> BGR to save image to video
        splash = splash[..., ::-1]
        # Add image to video writer
        # vwriter.write(splash)

        image = image[..., ::-1]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))

        plt.figure()
        plt.imshow(splash)
        plt.show()

    # vwriter.release()
    # print("Saved to ", file_name)

def photo(model):
    image = skimage.io.imread('D:\coursera\maskrcnn\\test.jpg')
    r = model.detect([image], verbose=0)[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))

if __name__ == '__main__':
    # ROOT_DIR = os.path.abspath('.')
    ROOT_DIR = 'D:\\coursera\\maskrcnn'
    coco_weight = os.path.join(ROOT_DIR, 'pretrain/mask_rcnn_coco.h5')

    config = CoCoInferenceConfig()
    model = MaskRCNN(mode='inference', config=config, model_dir=os.path.join(ROOT_DIR, 'checkpoint'))
    model.load_weights(coco_weight, by_name=True)

    # video_test(model)
    photo(model)