import os
import csv
import random
import logging
import numpy as np

from enum import Enum
from torchvision.datasets import ImageFolder
from typing import Callable, List, Optional, Tuple, Union, Any

from .extended import ExtendedVisionDataset


logger = logging.getLogger("TCoRe")
_Target = int


class _Split(Enum):
    TRAIN = "train"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 239_789,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_video_folder_relpath(self, video_id: str, class_name: str) -> str:
        dirname = self.get_dirname()
        return os.path.join(dirname, class_name, video_id)

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self == _Split.TRAIN
        dirname, filename = os.path.split(image_relpath)
        class_name = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        video_id = basename
        return class_name, video_id


class Kinetics(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Kinetics.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        past_offset_range = (0.05, 0.15),
        current_range = (0.3, 0.7),
        future_offset_range = (0.05, 0.15),
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

        self._past_offset_range = past_offset_range
        self._current_range = current_range
        self._future_offset_range = future_offset_range


    @property
    def split(self) -> "Kinetics.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        video_id = entries[index]["video_id"]
        class_name = self.get_class_name(index)
        video_folder_relpath = self.split.get_video_folder_relpath(video_id, class_name)
        video_folder_full_path = os.path.join(self.root, video_folder_relpath)

        past_frame_file, current_frame_file, future_frame_file = self.get_frames(video_folder_full_path, self._past_offset_range, self._current_range, self._future_offset_range)
        past_frame_path = os.path.join(video_folder_full_path, past_frame_file)
        current_frame_path = os.path.join(video_folder_full_path, current_frame_file)
        future_frame_path = os.path.join(video_folder_full_path, future_frame_file)
        with open(past_frame_path, mode="rb") as f:
            past_frame = f.read()
        with open(current_frame_path, mode="rb") as f:
            current_frame = f.read()
        with open(future_frame_path, mode="rb") as f:
            future_frame = f.read()
        image_data = [past_frame, current_frame, future_frame]
        frame_ids = [past_frame_file, current_frame_file, future_frame_file]

        return image_data, frame_ids
        
    def get_dataset(self):
        return "Kinetics"

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return str(class_name)
    
    def calculate_indices(self, frame_range, total_frames):
        start_idx = int(frame_range[0] * total_frames)
        end_idx = int(frame_range[1] * total_frames)
        return start_idx, end_idx
    
    def check_range_diff_past_current_future(self, past_start, past_end, current_idx, future_start, future_end):
        return past_start < past_end <= current_idx <= future_start < future_end

    def check_current(self, current_start, current_end):
        return current_start < current_end
    
    def get_frames(self, full_path, past_offset_range, current_range, future_offset_range):
        frames = sorted(f for f in os.listdir(full_path))
        total_frames = len(frames)
        current_start, current_end = self.calculate_indices(current_range, total_frames)
        valid_current = self.check_current(current_start, current_end)

        assert current_range[0] - past_offset_range[1] >= 0, "The range of past frame is invalid: < 0."
        assert current_range[1] + future_offset_range[1] <= 1, "The range of future frame is invalid: > 1."
        if valid_current:
            current_idx = random.randint(current_start, current_end - 1)
            past_start = current_idx - int(past_offset_range[1] * total_frames) # e.g. 8-0.25*20
            past_end = current_idx - int(past_offset_range[0] * total_frames)   # e.g. 8-0.15*20
            future_start = current_idx + int(future_offset_range[0] * total_frames) # e.g. 8+0.15*20
            future_end = current_idx + int(future_offset_range[1] * total_frames)   # e.g. 8+0.25*20
            valid_ranges = self.check_range_diff_past_current_future(past_start, past_end, current_idx, future_start, future_end)

        if not valid_current or not valid_ranges:
            past_frame = frames[random.randint(0, total_frames - 1)]
            current_frame = past_frame
            future_frame = past_frame
        else:
            past_idx = random.randint(past_start, past_end)
            future_idx = random.randint(future_start, min(future_end, total_frames-1))
            past_frame = frames[past_idx]
            current_frame = frames[current_idx]
            future_frame = frames[future_idx]

        return past_frame, current_frame, future_frame


    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []
        
        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels
    
    def _is_mp4_file(self, filename):
        basename = os.path.splitext(filename)[0]
        frame_dir = os.path.join(self.dataset_root, basename)
        frame_dir = frame_dir.replace("videos", "frames")
        return filename.endswith('.mp4') and os.path.isdir(frame_dir)
    
    def _dump_entries(self) -> None:
        split = self.split
        assert split == Kinetics.Split.TRAIN, 'Only support train split for Kinetics.'

        labels_path = "labels.txt"
        logger.info(f'loading labels from "{labels_path}"')
        labels = self._load_labels(labels_path)

        self.dataset_root = os.path.join(self.root, split.get_dirname())
        dataset = ImageFolder(self.dataset_root, is_valid_file=self._is_mp4_file)
        sample_count = len(dataset)
        max_class_id_length, max_class_name_length = -1, -1
        for sample in dataset.samples:
            _, class_index = sample
            class_id, class_name = labels[class_index]
            class_id = "class_" + str(class_id)
            max_class_id_length = max(len(class_id), max_class_id_length)
            max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("video_id", "<U25"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)
        class_names = {"class_" + class_id: class_name for class_id, class_name in labels}
        class_ids = {class_name: "class_" + class_id  for class_id, class_name in labels}

        assert dataset
        old_percent = -1
        for index in range(sample_count):
            percent = 100 * (index + 1) // sample_count
            if percent > old_percent:
                logger.info(f"creating entries: {percent}%")
                old_percent = percent

            image_full_path, class_index = dataset.samples[index]
            image_relpath = os.path.relpath(image_full_path, self.root)
            class_name, video_id = split.parse_image_relpath(image_relpath)
            class_id = class_ids[class_name]
            entries_array[index] = (video_id, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)


    def _dump_class_ids_and_names(self) -> None:
        split = self.split

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()

