import math
import torch
import logging
import warnings

from torch import Tensor
from collections.abc import Sequence
from typing import List, Optional, Tuple
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import _setup_size
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

from torchvision import transforms
from .transforms import (GaussianBlur, make_normalize_transform)

logger = logging.getLogger("TCoRe")


class FixedResizedCrop(torch.nn.Module):

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float], center_crop) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        if not center_crop:
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if 0 < w <= width and 0 < h <= height:
                    i = torch.randint(0, height - h + 1, size=(1,)).item()
                    j = torch.randint(0, width - w + 1, size=(1,)).item()
                    return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, pos_tuple=None, center_crop=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        if pos_tuple is None:
            i, j, h, w = self.get_params(img, self.scale, self.ratio, center_crop)
        else:
            i, j, h, w = pos_tuple
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), (i, j, h, w)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string


class DataAugmentationVideo(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = FixedResizedCrop(
            global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image, pos_tuple_list=None, center_crop=False):
        output = {}

        # global crops:
        if pos_tuple_list is None:
            im1_base, global_crop_1_pos_tuple = self.geometric_augmentation_global(image, pos_tuple=None, center_crop=center_crop)
            global_crop_1 = self.global_transfo1(im1_base)
            im2_base, global_crop_2_pos_tuple = self.geometric_augmentation_global(image, pos_tuple=None, center_crop=center_crop)
            global_crop_2 = self.global_transfo2(im2_base)
        else:
            crop_1_pos_tuple = pos_tuple_list[0]
            crop_2_pos_tuple = pos_tuple_list[1]
            im1_base, global_crop_1_pos_tuple = self.geometric_augmentation_global(image, pos_tuple=crop_1_pos_tuple, center_crop=center_crop)
            global_crop_1 = self.global_transfo1(im1_base)
            im2_base, global_crop_2_pos_tuple = self.geometric_augmentation_global(image, pos_tuple=crop_2_pos_tuple, center_crop=center_crop)
            global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        global_crops_pos_tuple_list = [global_crop_1_pos_tuple, global_crop_2_pos_tuple]
        output["global_crops_pos_tuple"] = global_crops_pos_tuple_list

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output, global_crops_pos_tuple_list
