import os
from torchvision import transforms
from torchvision.transforms import functional as F
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

######################################################
# NOTE: For attention map visualization and mask visualization
    def debug_save_crops(self, index, image, frame_ids, past_global_crops_pos_tuple_list, 
                         current_global_crops_pos_tuple_list, future_global_crops_pos_tuple_list):
        past, current, future = image
        size = [224, 224]
        past_crop_1 = F.resized_crop(past, *(past_global_crops_pos_tuple_list[0]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        current_crop_1 = F.resized_crop(current, *(current_global_crops_pos_tuple_list[0]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        future_crop_1 = F.resized_crop(future, *(future_global_crops_pos_tuple_list[0]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        past_crop_2 = F.resized_crop(past, *(past_global_crops_pos_tuple_list[1]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        current_crop_2 = F.resized_crop(current, *(current_global_crops_pos_tuple_list[1]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        future_crop_2 = F.resized_crop(future, *(future_global_crops_pos_tuple_list[1]), size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        entries = self._get_entries()
        video_id = entries[index]["video_id"]
        class_name = entries[index]["class_name"]
        save_dir = os.path.join(self.debug_save_path, 'crops', class_name, video_id)

        past_crop_1_path = os.path.join(save_dir, f'past_frame_{frame_ids[0][:5]}_crop_1.jpg')
        os.makedirs(os.path.dirname(past_crop_1_path), exist_ok=True)
        past_crop_1.save(past_crop_1_path)
        
        current_crop_1_path = os.path.join(save_dir, f'current_frame_{frame_ids[1][:5]}_crop_1.jpg')
        os.makedirs(os.path.dirname(current_crop_1_path), exist_ok=True)
        current_crop_1.save(current_crop_1_path)

        future_crop_1_path = os.path.join(save_dir, f'future_frame_{frame_ids[2][:5]}_crop_1.jpg')
        os.makedirs(os.path.dirname(future_crop_1_path), exist_ok=True)
        future_crop_1.save(future_crop_1_path)

        past_crop_2_path = os.path.join(save_dir, f'past_frame_{frame_ids[0][:5]}_crop_2.jpg')
        os.makedirs(os.path.dirname(past_crop_2_path), exist_ok=True)
        past_crop_2.save(past_crop_2_path)
        
        current_crop_2_path = os.path.join(save_dir, f'current_frame_{frame_ids[1][:5]}_crop_2.jpg')
        os.makedirs(os.path.dirname(current_crop_2_path), exist_ok=True)
        current_crop_2.save(current_crop_2_path)

        future_crop_2_path = os.path.join(save_dir, f'future_frame_{frame_ids[2][:5]}_crop_2.jpg')
        os.makedirs(os.path.dirname(future_crop_2_path), exist_ok=True)
        future_crop_2.save(future_crop_2_path)
######################################################


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

                entries = self._get_entries()
                video_id = entries[index]["video_id"]
                class_name = entries[index]["class_name"]
                frame_name = f"{class_name}#{video_id}#"
                
                past_frame_name = frame_name + f"past_frame_{frame_ids[0][:5]}"
                past_image, past_global_crops_pos_tuple_list = self.transform(past_frame_name, past_image, pos_tuple_list=None, center_crop=True)
                current_frame_name = frame_name + f"current_frame_{frame_ids[1][:5]}"
                current_image, current_global_crops_pos_tuple_list = self.transform(current_frame_name, current_image, pos_tuple_list=None, center_crop=False)
                future_frame_name = frame_name + f"future_frame_{frame_ids[2][:5]}"
                future_image, future_global_crops_pos_tuple_list = self.transform(future_frame_name, future_image, pos_tuple_list=past_global_crops_pos_tuple_list, center_crop=True)

                ######################################################
                # NOTE: For attention map visualization and mask visualization
                if self.debug_save:
                        self.debug_save_crops(index, image, frame_ids, past_global_crops_pos_tuple_list, 
                         current_global_crops_pos_tuple_list, future_global_crops_pos_tuple_list)
                ######################################################
                
                target = self.target_transform(target)
                image = [past_image, current_image, future_image]
            return image, target
        
        else:
            raise NotImplementedError(f"Not support {self.get_dataset()} dataset.")


    def __len__(self) -> int:
        raise NotImplementedError
