import random
from data.DataGenerator import Dataset
import numpy as np
import cv2
import math
from Utils.utils import non_max_suppression
import skimage
import skimage.draw
import skimage.io
import os
import json

class ShapesDataset(Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        # 处理遮挡
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

class BalloonDataset(Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

import scipy.io as scio
from tqdm import tqdm
class CaltechBirdsDataset(Dataset):
    def init_data(self):
        from sklearn.model_selection import train_test_split

        y = np.array(scio.loadmat(os.path.join('D:\\DataSet\\birds', 'imagelabels.mat'))['labels'][0])
        x = np.array(os.listdir(os.path.join('D:\\DataSet\\birds', 'jpg')))

        start_idx = 10
        end_idx = 15
        assert end_idx>start_idx
        index = np.where((y>start_idx)&(y<=end_idx))
        y = y[index]
        x = x[index]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=None)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        print('train num:', len(x_train), ' test num:', len(x_test))
        with open('D:\\DataSet\\birds\\train.txt', 'w') as f:
            for i in range(len(x_train)):
                f.write(x_train[i] + ' ' + str(y_train[i]-start_idx) + '\n')

        with open('D:\\DataSet\\birds\\val.txt', 'w') as f:
            for i in range(len(x_test)):
                f.write(x_test[i] + ' ' + str(y_test[i]-start_idx) + '\n')

    def load_birds(self, data_dir, subset):
        assert subset in ["train", "val"]

        categories = []
        with open(os.path.join(data_dir, 'label.txt'), 'r') as f:
            for line in f.readlines():
                categories.append(line.replace('\n', '').strip())

        img_dir = os.path.join(data_dir, 'jpg')
        gt_dir = os.path.join(data_dir, 'gt')

        class_list = []
        class_idx = 1
        with open(os.path.join(data_dir, subset+'.txt')) as f:
            for line in tqdm(f):
                img_name, label = line.replace('\n','').strip().split(' ')
                label = int(label)
                image_path = os.path.join(img_dir, img_name)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "bird",
                    image_id=image_path,
                    path=image_path,
                    width=width, height=height,
                    label=label,
                    mask_path = os.path.join(gt_dir, img_name.replace('jpg','png'))
                )

                category = categories[label-1]
                if category not in class_list:
                    class_list.append(category)
                    self.add_class("bird", class_idx, category)
                    class_idx += 1

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "bird":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = skimage.io.imread(info['mask_path'])
        mask = (mask/255.).astype(np.bool)
        if len(mask.shape)==2:
            mask = mask[:,:, np.newaxis]

        class_id = np.ones([mask.shape[-1]], dtype=np.int32)
        class_id *= np.array(info['label']).reshape(-1)
        return mask, class_id

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bird":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class HorseDataset(Dataset):
    def init_data(self):

        img_jpg = os.listdir('D:\\DataSet\\horse\\jpg')
        img_gt =  os.listdir('D:\\DataSet\\horse\\gt')

        assert img_jpg==img_gt
        index = np.arange(0, len(img_jpg), 1)
        np.random.shuffle(index)

        train_num = int(len(index)*0.8)
        train_index = index[: train_num]
        val_index = index[train_num:]

        with open('D:\\DataSet\\horse\\train.txt', 'w') as f:
            for i in train_index:
                f.write(img_jpg[i]+'\n')

        with open('D:\\DataSet\\horse\\val.txt', 'w') as f:
            for i in val_index:
                f.write(img_jpg[i]+'\n')

        print('train num:%d val num:%d'%(len(train_index), len(val_index)))

    def load_horse(self, data_dir, subset):
        assert subset in ["train", "val"]

        self.add_class("horse", 1, "horse")

        img_dir = os.path.join(data_dir, 'jpg')
        gt_dir = os.path.join(data_dir, 'gt')

        with open(os.path.join(data_dir, subset+'.txt')) as f:
            for line in tqdm(f):
                img_name = line.replace('\n','').strip()
                image_path = os.path.join(img_dir, img_name)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "horse",
                    image_id=image_path,
                    path=image_path,
                    width=width, height=height,
                    mask_path = os.path.join(gt_dir, img_name)
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "horse":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = skimage.io.imread(info['mask_path'])

        mask[np.where(mask <= mask.mean())] = 0
        mask[np.where(mask > mask.mean())] = 1

        width, height = info['width'], info['height']
        mask = cv2.resize(mask, (width, height))
        mask[np.where(mask <= mask.mean())] = 0
        mask[np.where(mask > mask.mean())] = 1

        if len(mask.shape)==2:
            mask = mask[:,:, np.newaxis]

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "horse":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class NucleusDataset(Dataset):
    def init_data(self):
        data_list = os.listdir('D:\DataSet\data-science-bowl-2018\data')

        index = np.arange(0, len(data_list), 1)
        np.random.shuffle(index)

        train_num = int(len(index) * 0.8)
        train_index = index[: train_num]
        val_index = index[train_num:]

        with open('D:\\DataSet\\data-science-bowl-2018\\train.txt', 'w') as f:
            for i in train_index:
                f.write(data_list[i] + '\n')

        with open('D:\\DataSet\\data-science-bowl-2018\\val.txt', 'w') as f:
            for i in val_index:
                f.write(data_list[i] + '\n')

    def load_nucleus(self, data_dir, subset):
        self.add_class("nucleus", 1, "nucleus")

        assert subset in ["train", "val"]

        cell_data_dir = os.path.join(data_dir, 'data')
        with open(os.path.join(data_dir, subset+'.txt')) as f:
            for line in tqdm(f):
                img_id = line.replace('\n', '').strip()

                img_dir = os.path.join(cell_data_dir, img_id)
                img_path = os.path.join(os.path.join(img_dir, 'images'), img_id)+'.png'

                self.add_image(
                    "nucleus",
                    image_id=img_id,
                    mask_dir = os.path.join(img_dir, 'masks'),
                    path=img_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = info['mask_dir']

        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

if __name__ == '__main__':
    # dataset_train = CaltechBirdsDataset()
    # dataset_train.init_data()
    # dataset_train.load_birds('D:\DataSet\\birds', 'train')
    # dataset_train.prepare()

    # import matplotlib.pyplot as plt
    #
    # dataset_train = HorseDataset()
    # dataset_train.load_horse('D:\DataSet\\horse', 'train')
    # dataset_train.prepare()

    dataset_train = NucleusDataset()
    # dataset_train.init_data()
    dataset_train.load_nucleus('D:\DataSet\data-science-bowl-2018', 'train')
    dataset_train.prepare()

    pass


