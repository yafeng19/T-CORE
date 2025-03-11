
from base_model.data.datasets import Kinetics

if __name__ == "__main__":
    description = "Dump files for Kinetics-400."
    for split in Kinetics.Split:
        dataset = Kinetics(split=split, root="data/Kinetics-400/videos", extra="data/Kinetics-400/videos")
        dataset.dump_extra()
