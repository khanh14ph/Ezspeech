import tarfile
import os
import torch
import glob
from ezspeech.utils.constant import nemo2ez_dict
import shutil

from omegaconf import OmegaConf


def untar(tar_file, folder):
    """Simple one-liner to extract tar to folder"""
    os.makedirs(folder, exist_ok=True)
    tarfile.open(tar_file, "r:*").extractall(folder)
    print(f"Extracted {tar_file} → {folder}")


def convert_nemo_to_ez(nemo_path, ez_checkpoint_folder):
    temp_dir = f"{ez_checkpoint_folder}/nemo_temp"
    os.makedirs(temp_dir, exist_ok=True)
    untar(nemo_path, temp_dir)
    nemo_tokenizer_path = glob.glob(f"{temp_dir}/*.model")[0]
    nemo_weights = torch.load(temp_dir + "/model_weights.ckpt")
    module_set = []
    for i in nemo_weights.keys():
        module_set.append(i.split(".")[0])
    module_set = sorted(list(set(module_set)))

    nemo_config = OmegaConf.load(temp_dir + "/model_config.yaml")
    print("module_set", module_set)
    if "ctc_decoder" in module_set:

        nemo_config["ctc_decoder"] = nemo_config["aux_ctc"]["decoder"]
    print("nemo_config", nemo_config.keys())
    for i in module_set:
        nemo_config[i]["_target_"] = nemo2ez_dict[nemo_config[i]["_target_"]]

    final = dict()
    for i in module_set:
        final[i] = dict()
        for j in nemo_weights.keys():
            name, new_name = j.split(".", 1)
            if name == i:
                final[i][new_name] = nemo_weights[j]
    new_final = dict()
    ez_config = {i: nemo_config[i] for i in module_set}
    for i in ez_config.keys():
        if i in final.keys():
            new_final[i] = final[i]
            print(f"convert module ** {i} ** successfully")
    checkpoint = {
        "state_dict": new_final,
        "hyper_parameters": ez_config,
    }
    shutil.copy(nemo_tokenizer_path, ez_checkpoint_folder + "/tokenizer.model")
    shutil.rmtree(temp_dir)

    torch.save(checkpoint, ez_checkpoint_folder + "/model_weights.ckpt")
    OmegaConf.save(ez_config, ez_checkpoint_folder + "/model_configs.yaml")


if __name__ == "__main__":
    convert_nemo_to_ez(
        "/home4/khanhnd/cache/hub/models--nvidia--stt_en_fastconformer_hybrid_large_streaming_multi/snapshots/ae98143333690bd7ced4bc8ec16769bcb8918374/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
        "/home4/khanhnd/exported_checkpoint/checkpoint_streaming",
    )
