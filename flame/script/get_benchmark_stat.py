import argparse
import os
import re

import numpy as np


def print_array(array):
    string = array[0]
    for col in array[1:]:
        string += f",{col}"

    print(string)


def print_dict(array, dict):
    string = f"{dict[array[0]]}"
    for col in array[1:]:
        string += f",{dict[col]}"

    print(string)


def get_stat(log_file, cols):
    res = {}
    for col in cols:
        res[col] = 0
    res["memory"] = []
    res["tgs"] = []
    pattern = r"memory:\s*(\d+\.?\d*)GiB.*?tgs:\s*(\d+(?:,\d+)*)"
    mode_pattern = r"'mode':\s*'(\w+)'"
    selective_pattern = r"'selective_ac_option':\s*'(\d+)'"

    with open(log_file, "r") as f:
        for line in f.readlines():
            for col in ["model_source", "model_type", "tie_word_embeddings"]:
                if col in line:
                    res[col] = line.split()[-1].strip(",").strip("'").strip('"')
            for col in ["parameters"]:
                if col in line and res[col] == 0:
                    res[col] = round(float(line.split()[-1].replace(",", "")) / 1e9, 4)
            for col in [
                "tgs",
            ]:
                if col in line:
                    match = re.search(pattern, line)
                    if match:
                        res["memory"].append(float(match.group(1)))
                        res["tgs"].append(float(match.group(2).replace(",", "")))
            for col in [
                "batch_size",
                "context_len",
                "'compile'",
                "tensor_parallel_degree",
            ]:
                if col in line:
                    res[col] = line.split()[-1].strip(",")
            if "mode" in line:
                match = re.search(mode_pattern, line)
                if match:
                    res["mode"] = match.group(1)
            if "selective_ac_option" in line:
                match = re.search(selective_pattern, line)
                if match:
                    res["selective_ac_option"] = match.group(1)

    res["memory"] = round(np.mean(res["memory"][-10:]), 2)
    res["tgs"] = round(np.mean(res["tgs"][-10:]), 2)
    res["activation_checkpoint"] = f"{res['mode']}_{res['selective_ac_option']}"

    return res


def main(args):
    cols = [
        "model_source",
        "model_type",
        "parameters",
        "memory",
        "tgs",
        "batch_size",
        "context_len",
        "'compile'",
        "activation_checkpoint",
        "tensor_parallel_degree",
        "tie_word_embeddings",
    ]
    print_array(cols)

    log_file = args.log_file
    log_dir = args.log_dir
    if log_dir == "":
        stat = get_stat(log_file, cols)
        print_dict(cols, stat)
    else:
        for log_file in os.listdir(log_dir):
            stat = get_stat(os.path.join(log_dir, log_file), cols)
            print_dict(cols, stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    args = parser.parse_args()
    main(args)
