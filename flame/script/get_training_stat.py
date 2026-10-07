import argparse
import csv
import sys
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def print_array(array):
    csv.writer(sys.stdout).writerow(array)


def print_dict(array, values):
    csv.writer(sys.stdout).writerow([values[col] for col in array])


def nan_stat(cols):
    res = {}
    for col in cols:
        res[col] = "nan"
    return res


def get_stat(log_file, cols, target_step=None):
    res = {}
    for col in cols:
        res[col] = 0
    res["memory"] = []
    res["tgs"] = []
    loss = []
    max_step = 0
    memory_tgs_pattern = r"memory:\s*(\d+\.?\d*)GiB.*?tgs:\s*(\d+(?:,\d+)*)"
    token_loss_pattern = r"token:\s*([\d,]+(?:\.\d+)?)\s+loss:\s*(\d+\.\d+)"
    step_pattern = r"step:\s*([\d,]+)"
    mode_pattern = r"'mode':\s*'(\w+)'"
    selective_pattern = r"'selective_ac_option':\s*'(\d+)'"
    finish = False

    earliest_time = None
    latest_time = None

    with open(log_file, "r") as f:
        for line in f.readlines():
            if "Training completed" in line:
                finish = True

            current_step = None
            if "step:" in line:
                step_match = re.search(step_pattern, line)
                if step_match:
                    current_step = int(step_match.group(1).replace(",", ""))
                    max_step = max(max_step, current_step)

            if "INFO" in line:
                time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
                if time_match:
                    try:
                        current_time = datetime.strptime(
                            time_match.group(1), "%Y-%m-%d %H:%M:%S"
                        )

                        if earliest_time is None or current_time < earliest_time:
                            earliest_time = current_time
                        if latest_time is None or current_time > latest_time:
                            latest_time = current_time
                    except ValueError:
                        continue

            if "PisaConfig(" in line:
                res["model_type"] = "pisa"
            for col in ["model_source", "model_type"]:
                if col in line:
                    res[col] = line.split()[-1].strip(",").strip("'").strip('"')
            for col in ["parameters"]:
                if col in line and res[col] == 0:
                    res[col] = round(float(line.split()[-1].replace(",", "")) / 1e9, 4)
            for col in [
                "tgs",
            ]:
                if col in line and (
                    target_step is None
                    or current_step is None
                    or current_step <= target_step
                ):
                    match = re.search(memory_tgs_pattern, line)
                    if match:
                        res["memory"].append(float(match.group(1)))
                        res["tgs"].append(float(match.group(2).replace(",", "")))
            if "loss" in line and (
                target_step is None
                or current_step is None
                or current_step <= target_step
            ):
                match = re.search(token_loss_pattern, line)
                if match:
                    res["token"] = round(
                        float(match.group(1).replace(",", "")) / 1e9, 2
                    )
                    res["loss"] = round(float(match.group(2)), 4)
                    loss.append(res["loss"])
            for col in [
                "batch_size",
                "seq_len",
                "context_len",
                "'compile'",
                "tensor_parallel_degree",
                "'eps'",
                "varlen",
                "warmup_steps",
            ]:
                if col in line:
                    res[col] = line.split()[-1].strip(",")

            if '"steps"' in line:
                res["steps"] = line.split()[-1].strip(",")

    if target_step is not None and max_step < target_step:
        return nan_stat(cols), finish

    res["memory"] = round(np.mean(res["memory"][-10:]), 2) if res["memory"] else float("nan")
    res["tgs"] = round(np.mean(res["tgs"][-10:]), 2) if res["tgs"] else float("nan")
    res["loss_avg_5"] = round(np.mean(loss[-5:]), 4) if loss else float("nan")

    # get time
    try:
        duration = latest_time - earliest_time
        time = duration.total_seconds() / 3600
        res["time"] = round(time, 2)
    except:
        res["time"] = 0

    return res, finish


def main(args):
    cols = [
        "name",
        "model_source",
        "model_type",
        "parameters",
        "loss",
        "loss_avg_5",
        "token",
        "memory",
        "tgs",
        "batch_size",
        "context_len",
        "seq_len",
        "'eps'",
        "steps",
        "warmup_steps",
        "varlen",
        "time",
    ]
    print_array(cols)

    paths = sorted(Path(args.log_dir).glob("*.log")) if args.log_dir else [Path(args.log_file)]
    for log_file in paths:
        if not log_file.is_file():
            raise FileNotFoundError(f"Training log does not exist: {log_file}")
        # Original dated log names support the same threshold/deadhold options.
        # Explicit workflow names such as pt-pisa.log carry no filename date.
        dated = re.match(r"^\d{8}-", log_file.name)
        if dated and (log_file.name < args.threshold or (args.deadhold and log_file.name > args.deadhold)):
            continue
        stat, finish = get_stat(log_file, cols, target_step=args.target_step)
        stat["name"] = re.sub(r"^\d{8}-\d{2}:\d{2}:\d{2}-", "", log_file.stem)
        if finish or args.allow_unfinished:
            print_dict(cols, stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--threshold", type=str, default="20250101")
    parser.add_argument("--deadhold", type=str, default="")
    parser.add_argument("--allow_unfinished", action="store_true")
    parser.add_argument("--target_step", type=int, default=None)
    args = parser.parse_args()
    if not args.log_file and not args.log_dir:
        parser.error("provide --log_file or --log_dir")
    if args.log_dir and not Path(args.log_dir).is_dir():
        parser.error(f"log directory does not exist: {args.log_dir}")
    main(args)
