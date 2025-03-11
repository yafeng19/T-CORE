from typing import Any, Tuple
from torchvision.datasets import VisionDataset
from .decoders import TargetDecoder, ImageDataDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError
    
    def get_attr(self):
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if "Kinetics" in self.get_dataset():
            try:
                image_data, frame_ids = self.get_image_data(index)
                image = [ImageDataDecoder(item).decode() for item in image_data]
            except Exception as e:
                raise RuntimeError(f"can not read image for sample {index}") from e
            target = self.get_target(index)
            target = TargetDecoder(target).decode()
            
            if self.transform is not None and self.target_transform is not None:
                past_image, current_image, future_image = image
                past_image, past_global_crops_pos_tuple_list = self.transform(past_image, pos_tuple_list=None, center_crop=True)
                current_image, current_global_crops_pos_tuple_list = self.transform(current_image, pos_tuple_list=None, center_crop=False)
                future_image, future_global_crops_pos_tuple_list = self.transform(future_image, pos_tuple_list=past_global_crops_pos_tuple_list, center_crop=True)
                target = self.target_transform(target)
                image = [past_image, current_image, future_image]
            return image, target
        
        else:
            raise NotImplementedError(f"Not support {self.get_dataset()} dataset.")


    def __len__(self) -> int:
        raise NotImplementedError
