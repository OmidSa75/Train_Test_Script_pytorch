from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image
import torch


class TransForms:
    def __init__(self, img_size):
        self.img_size = img_size
        self.train_tfms = self.return_train_transforms()
        self.test_tfms = self.return_test_transforms()
        self.simple_tfms = self.simple_transform()

    def pad_image(self, image):
        padded_im = functional.pad(image, self.get_padding(image))  # torchvision.transforms.functional.pad
        return padded_im

    @staticmethod
    def get_padding(image: Image.Image):
        """ Add pad to image to square it.
        """
        width, height = image.size
        if height > width:
            pad = (height - width) / 2
            l_pad = pad if pad % 1 == 0 else pad + 0.5
            r_pad = pad if pad % 1 == 0 else pad - 0.5
            return [int(l_pad), 0, int(r_pad), 0]

        elif width > height:
            pad = (width - height) / 2
            t_pad = pad if pad % 1 == 0 else pad + 0.5
            b_pad = pad if pad % 1 == 0 else pad - 0.5
            return [0, int(t_pad), 0, int(b_pad)]

        else:
            return [0, 0, 0, 0]

    def return_train_transforms(self):
        tfms = transforms.Compose([
            # transforms.Lambda(self.pad_image),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomAutocontrast(),
            transforms.RandomRotation([-90, 90]),
            transforms.RandomAffine((-90, 90)),
            transforms.Resize([self.img_size[0], self.img_size[1]], transforms.transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.RandomErasing(0.25),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return tfms

    def return_test_transforms(self):
        tfms = transforms.Compose([
            # transforms.Lambda(self.pad_image),
            transforms.Resize([self.img_size[0], self.img_size[1]], transforms.transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return tfms

    def simple_transform(self):
        tfms = transforms.Compose([
            transforms.Resize([self.img_size[0], self.img_size[1]], transforms.transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        return tfms


class Utils:
    @staticmethod
    def calc_acc(preds: torch.Tensor, labels: torch.Tensor):
        _, pred_max = torch.max(preds, 1)
        acc = torch.sum(pred_max == labels.data, dtype=torch.float64) / len(preds)
        return acc

    @staticmethod
    def make_weights_for_balanced_classes(images, nclasses):
        count = [0] * nclasses
        for item in images:  # count each class population
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

    @staticmethod
    def to_img(x: torch.Tensor, size):
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x
