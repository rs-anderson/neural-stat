from collections import defaultdict
from typing import List, Dict
import torch
import numpy as np
import PIL.ImageOps    
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, Omniglot
from skimage.morphology import binary_dilation, disk
from skimage.transform import resize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### add non-binary dilation blabla
def get_dataset(opts, split='train'):
    """Function to get instance of SyntheticDataset given training options."""
    return OmniglotDataset(opts.num_data_per_dataset, split, opts.test_mnist, opts.num_classes)


class OmniglotDataset():
    """Dataset for omniglot data experiment."""

    def __init__(self, num_data_per_dataset, split, mnist, num_classes):
        """
        :param num_data_per_dataset: int, number of data per dataset
        :param split: str, type of dataset
        :param mnist: boolean, whether to test on mnist
        """
        self.split = split
        self.num_data_per_dataset = num_data_per_dataset # K-shot
        self.mnist = mnist
        self.num_classes = num_classes # K-Way
        self.mode = None

        if self.split == 'test' and self.mnist:
            data = MNIST('mnist', download=True, train=True)
            x_train, y_train = data.data.float()/255, data.targets
            data = MNIST('mnist', download=True, train=False)
            x_test, y_test = data.data.float()/255, data.targets
            self.x = torch.cat([x_train.round(), x_test.round()], dim=0)[:, None]
            self.y = torch.cat([y_train, y_test], dim=0)

        else:
            # self.dataset_background = Omniglot('omniglot', download=True) # train set
            self.dataset_eval = Omniglot('omniglot', download=True, background=False) # val set

    def _get_M_classes(self):
        if self.mnist:
            total_num_classes = 10
        else:
            total_num_classes = 659
        classes_for_experiment = np.random.choice(total_num_classes, size=self.num_classes, replace=False)
        return classes_for_experiment

    def _get_class_inds_in_dataset(self, classes_for_experiment):
        im_inds_for_exp_by_class = defaultdict(list)  # indices in dataset for images from the M classes
        for ind, (_, label) in enumerate(self.dataset_eval):
            if label in classes_for_experiment:
                im_inds_for_exp_by_class[label].append(ind)
        return im_inds_for_exp_by_class
    
    def _get_data_split_inds_for_class(self, inds_list):
        seen_inds = list(np.random.choice(inds_list, size=self.num_data_per_dataset, replace=False))
        test_inds = list(set(inds_list) - set(seen_inds))
        return seen_inds, test_inds
    
    def _get_data_split_inds(self, im_inds_for_exp_by_class: Dict[int, List]):
        test_inds_all = list()  # keep track of indices for testing
        seen_inds_all = list()
        for _, inds_list in im_inds_for_exp_by_class.items():
            train_inds, test_inds = self._get_data_split_inds_for_class(inds_list)
            test_inds_all += test_inds
            seen_inds_all += train_inds
        return seen_inds_all, test_inds_all

    def _dataset_to_np(self, dataset: Dataset):
        images = []
        labels = []
        for im, label in dataset:
            image = PIL.ImageOps.invert(im)
            images += [(np.asarray(image.resize((28, 28)))/255).round()]
            labels += [label]
        return np.stack(images, axis=0), np.array(labels)

    # TODO: make this compatible with dataloader - use changing modes .set_train() etc. Batch sizes are too large otherwise.
    def sample_experiment(self):
        classes_for_experiment = self._get_M_classes()
        im_inds_for_exp_by_class = self._get_class_inds_in_dataset(classes_for_experiment)
        seen_inds_all, test_inds_all = self._get_data_split_inds(im_inds_for_exp_by_class)

        seen_dataset = Subset(self.dataset_eval, seen_inds_all)
        test_dataset = Subset(self.dataset_eval, test_inds_all)
        
        self.seen_images, _ = self._dataset_to_np(seen_dataset)
        self.test_images, self.test_labels = self._dataset_to_np(test_dataset)

        # return torch.FloatTensor(seen_images[:, None]), seen_labels, torch.FloatTensor(test_images[:, None]), test_labels


    def set_train(self):
        self.mode = "seen"
    
    def set_test(self):
        self.mode = "test"

    
    def __len__(self):
        if self.mode is None:
            raise Exception("Set mode before calling function.")
        if self.mode == 'seen':
            return len(self.seen_images)
        elif self.mode == 'test':
            return len(self.test_labels)



    def __getitem__(self, idx):
        if self.mode is None:
            raise Exception("Set mode before calling function.")
        if self.mode == 'seen':
            return {'datasets': torch.FloatTensor(self.seen_images[:, None][idx]), 'targets': -1}
        elif self.mode == 'test':
            return {'datasets': torch.FloatTensor(self.test_images[:, None][idx]), 'targets': self.test_labels[idx]}

    
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = OmniglotDataset(5, "test", False, 10)
    np.random.seed(0)

    test_dataloader = DataLoader(ds, batch_size=16, shuffle=False)

    ds.sample_experiment()
    import ipdb; ipdb.set_trace()
    ds.set_train()
    ds.set_test()