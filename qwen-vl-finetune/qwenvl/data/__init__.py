import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}
PixMo_Absolute = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/FalconVLMER_anno_versions/Affordance/PixMo_qwen_absolute.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/PixMo/pixmo-points/images/"
}
PixMo_Relative = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/FalconVLMER_anno_versions/Affordance/PixMo_qwen_relative.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/PixMo/pixmo-points/images/"
}

epic100_relative = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/epic100_train/epic100_train_relative.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}
epic100_absolute = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/epic100_train/epic100_train_absolute.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}

ego4d_relative = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/Ego4D_train/Ego4D_train_relative.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}
ego4d_absolute = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/Ego4D_train/Ego4D_train_absolute.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}

handal_relative = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/HANDAL_train/HANDAL_train_relative.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}
handal_absolute = {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/HANDAL_train/HANDAL_train_absolute.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}
oxe_relative =  {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/affordance_points/oxe_relative_train.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}

oxe_absolute =  {
    "annotation_path": "/lustre1/tier2/projects/falcon-vla/vlm_data/affordance_points/oxe_absolute_train.qwen.json",
    "data_path": "/lustre1/tier2/projects/falcon-vla/oxe_raw_images/"
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    # Aliases for clarity
    "pixmo_pointing_absolute": PixMo_Absolute,
    "pixmo_pointing_relative": PixMo_Relative,
    # Backward-compatible key
    "pixmo_absolute": PixMo_Absolute,
    "epic100_relative": epic100_relative,
    "epic100_absolute": epic100_absolute,
    "ego4d_relative": ego4d_relative,
    "ego4d_absolute": ego4d_absolute,
    "handal_relative": handal_relative,
    "handal_absolute": handal_absolute,
    "oxe_relative": oxe_relative,
    "oxe_absolute": oxe_absolute,

}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
