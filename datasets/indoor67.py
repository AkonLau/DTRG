import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from datasets.tfs import get_cub_transform

# Dataset
class ImageLoader(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        img_path = root+'/'+'Images'
        class_to_idx = self._find_classes(img_path)

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(img_path, line.rstrip('\n')))
                self.labels.append(class_to_idx[line.split('/')[0]])

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

def get_dtd_transform(conf=None):
    return get_cub_transform(conf)

# Load datasets
def get_dataset(conf):
    datadir = './data/MIT-Scene'
    train_txt = datadir + '/' + 'TrainImages.txt'
    eval_txt = datadir + '/' + 'TestImages.txt'
    conf['num_class'] = 67

    transform_train, transform_test = get_dtd_transform(conf)

    ds_train = ImageLoader(datadir, train_txt, transform=transform_train)
    ds_test = ImageLoader(datadir, eval_txt, transform=transform_test)

    return ds_train,ds_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=200, help="class numbers")

    conf = parser.parse_args()
    ds_train,ds_test = get_dataset(conf)
    print(len(ds_train),len(ds_test))