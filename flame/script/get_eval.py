import os
import re
from pathlib import Path


def extract_metrics_from_log(log_content):
    """Extract metrics from log content"""
    metrics = {}
    current_task = None
    flag = False

    # Find the results table in log content
    lines = log_content.split("\n")
    for line in lines:
        if "hf (pretrained" in line:
            flag = True

        if not flag:
            continue

        if "|" not in line and "Tasks" not in line:
            continue

        parts = [part.strip() for part in line.split("|")]

        task = parts[1].strip()
        metric = parts[5].strip()
        value = parts[7].strip()

        if task:
            current_task = task
        # assert False
        if metric and value and value != "N/A":
            if current_task not in metrics:
                metrics[current_task] = {}
            try:
                metrics[current_task][metric] = float(value) if value else None
            except:
                pass

    return metrics


def create_metrics_row(metrics):
    """Create a row of metrics based on predefined columns"""
    columns = {
        "wikitext": "word_perplexity",
        "lambada_openai": "perplexity",
        "boolq": "acc",
        "piqa": "acc",
        "hellaswag": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "openbookqa": "acc_norm",
        "social_iqa": "acc",
        "swde": "contains",
        "squad_completion": "contains",
        "fda": "contains",
    }

    row = {}
    for task, metric in columns.items():
        if task in metrics and metric in metrics[task]:
            row[task] = metrics[task][metric]
        else:
            row[task] = None

    return row


def process_log_file(file_path):
    """Process a single log file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return

    metrics = extract_metrics_from_log(content)
    row = create_metrics_row(metrics)

    # Add model name (from file name)
    model_name = Path(file_path).stem[18:]
    row = {"model": model_name, **row}

    return row


def process_logs(args):
    """Process log file(s) and create CSV"""
    input_path = args.input_path
    threshold = args.threshold
    rows = []

    if os.path.isfile(input_path):
        # Process single file
        row = process_log_file(input_path)
        if row:
            rows.append(row)
    elif os.path.isdir(input_path):
        # Process all .log files in directory
        for file_name in os.listdir(input_path):
            if file_name.endswith(".log"):
                if file_name >= threshold:
                    file_path = os.path.join(input_path, file_name)
                    row = process_log_file(file_path)
                    if row:
                        rows.append(row)

    if rows:
        # Define columns order
        columns = [
            "model",
            "wikitext",
            "lambada_openai",
            "boolq",
            "piqa",
            "hellaswag",
            "winogrande",
            "arc_easy",
            "arc_challenge",
            "openbookqa",
            "social_iqa",
            "swde",
            "squad_completion",
            "fda",
        ]

        # Define metric types
        metrics_types = [
            "task",
            "word_perplexity",
            "perplexity",
            "acc",
            "acc",
            "acc_norm",
            "acc",
            "acc",
            "acc_norm",
            "acc_norm",
            "acc",
            "contains",
            "contains",
            "contains",
        ]

        # Print headers and metric types
        print(",".join(columns))
        print(",".join(metrics_types))

        # Print data rows
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, "")
                values.append(str(val) if val is not None else "")

            if args.full == "True":
                print(",".join(values))
            else:
                if values[-1] != "":
                    print(",".join(values))

        return rows
    else:
        print("No valid log files found or unable to process log content")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to log file or directory containing log files",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="20250101",
    )
    parser.add_argument(
        "--full",
        type=str,
        default="False",
    )
    args = parser.parse_args()

    process_logs(args)
