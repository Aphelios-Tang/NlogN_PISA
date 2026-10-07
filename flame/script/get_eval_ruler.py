import os
from pathlib import Path
from typing import Dict, List, Optional


def extract_metrics_from_log(log_content: str) -> Dict[str, Dict[str, float]]:
    """
    Extract metrics from lm_eval's markdown-like results table.

    This parser is tailored for RULER logs, where the final aggregated table
    appears after a line containing "hf (pretrained".
    """
    metrics: Dict[str, Dict[str, float]] = {}
    current_task: Optional[str] = None
    in_results_table = False

    for line in log_content.split("\n"):
        if "hf (pretrained" in line or "hf ({'pretrained'" in line:
            in_results_table = True

        if not in_results_table or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        # Expected columns:
        # ['', task, version, filter, n-shot, metric, '', value, '', stderr, ...]
        if len(parts) < 8:
            continue

        task = parts[1]
        metric = parts[5]
        value = parts[7]

        # Skip header/separator lines.
        if task.lower() == "tasks" or task.startswith("-"):
            continue
        if metric.startswith("-") or value.startswith("-"):
            continue

        if task:
            current_task = task

        if not current_task:
            continue

        if metric and value and value != "N/A":
            metrics.setdefault(current_task, {})
            try:
                metrics[current_task][metric] = float(value)
            except ValueError:
                pass

    return metrics


def _mean(values: List[Optional[float]]) -> Optional[float]:
    """Compute mean of non-None values; return None if all values are None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _len_label(length: str) -> str:
    """Map raw length string to a compact label used in column names."""
    mapping = {
        "1024": "1k",
        "2048": "2k",
        "4096": "4k",
        "8192": "8k",
        "16384": "16k",
    }
    return mapping.get(length, length)


def _format_value(val: Optional[float]) -> str:
    """Format value as percentage with 2 decimals; empty for None."""
    if val is None:
        return ""
    return f"{val * 100:.2f}"


def create_metrics_row(
    metrics: Dict[str, Dict[str, float]],
    include_cwe_fwe_vt: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Create a flat row for CSV printing.

    This file targets logs like:
    20260112-21:42:54_llama_half_rope_...log
    which contain these RULER tasks and lengths.
    """
    tasks = [
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multiquery",
        "niah_multivalue",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
    ]
    if include_cwe_fwe_vt:
        tasks.extend(["ruler_vt", "ruler_cwe", "ruler_fwe"])
    lengths = ["1024", "2048", "4096", "8192", "16384"]

    row: Dict[str, Optional[float]] = {}
    # Aggregate columns (placed "before" raw columns in the final CSV).
    for length in lengths:
        lab = _len_label(length)

        # Averages across task variants.
        row[f"niah_multikey-{lab}"] = _mean(
            [
                metrics.get("niah_multikey_1", {}).get(length),
                metrics.get("niah_multikey_2", {}).get(length),
                metrics.get("niah_multikey_3", {}).get(length),
            ]
        )
        row[f"niah_single-{lab}"] = _mean(
            [
                metrics.get("niah_single_1", {}).get(length),
                metrics.get("niah_single_2", {}).get(length),
                metrics.get("niah_single_3", {}).get(length),
            ]
        )

        # Single tasks (kept for consistency; mean equals the raw value).
        row[f"niah_multiquery-{lab}"] = metrics.get("niah_multiquery", {}).get(length)
        row[f"niah_multivalue-{lab}"] = metrics.get("niah_multivalue", {}).get(length)
        if include_cwe_fwe_vt:
            row[f"ruler_vt-{lab}"] = metrics.get("ruler_vt", {}).get(length)
            row[f"ruler_cwe-{lab}"] = metrics.get("ruler_cwe", {}).get(length)
            row[f"ruler_fwe-{lab}"] = metrics.get("ruler_fwe", {}).get(length)

    for task in tasks:
        for length in lengths:
            col = f"{task}_{length}"
            row[col] = metrics.get(task, {}).get(length)

    return row


def process_log_file(
    file_path: str,
    include_cwe_fwe_vt: bool = False,
) -> Optional[Dict[str, Optional[float]]]:
    """Process a single log file and return a row dict."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    metrics = extract_metrics_from_log(content)
    row = create_metrics_row(metrics, include_cwe_fwe_vt=include_cwe_fwe_vt)

    # Add model name (from file name). Prefix "YYYYMMDD-HH:MM:SS_" has length 18.
    model_name = Path(file_path).stem[18:]
    return {"model": model_name, **row}


def process_logs(args):
    """Process log file(s) and print CSV to stdout."""
    input_path = args.input_path
    threshold = args.threshold
    # Backward-compatible parsing:
    # - New flag: include_cwe_fwe_vt
    # - Old flag: include_cwe_fwe (deprecated)
    include_cwe_fwe_vt = (
        getattr(args, "include_cwe_fwe_vt", "False") == "True"
        or getattr(args, "include_cwe_fwe", "False") == "True"
    )
    rows: List[Dict[str, Optional[float]]] = []

    if os.path.isfile(input_path):
        row = process_log_file(input_path, include_cwe_fwe_vt=include_cwe_fwe_vt)
        if row:
            rows.append(row)
    elif os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            if not file_name.endswith(".log"):
                continue
            if file_name < threshold:
                continue
            file_path = os.path.join(input_path, file_name)
            row = process_log_file(file_path, include_cwe_fwe_vt=include_cwe_fwe_vt)
            if row:
                rows.append(row)

    if not rows:
        print("No valid log files found or unable to process log content")
        return None

    # Fixed columns order.
    tasks = [
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multiquery",
        "niah_multivalue",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
    ]
    if include_cwe_fwe_vt:
        tasks.extend(["ruler_vt", "ruler_cwe", "ruler_fwe"])
    lengths = ["1024", "2048", "4096", "8192", "16384"]

    columns: List[str] = ["model"]
    # Aggregate columns first.
    # Order requirement: for each type, iterate all lengths (e.g. multikey-1k,2k,4k,8k,...).
    agg_types = [
        "niah_multikey",
        "niah_single",
        "niah_multiquery",
        "niah_multivalue",
    ]
    if include_cwe_fwe_vt:
        agg_types.extend(["ruler_vt", "ruler_cwe", "ruler_fwe"])
    for agg_type in agg_types:
        for length in lengths:
            columns.append(f"{agg_type}-{_len_label(length)}")
    for task in tasks:
        for length in lengths:
            columns.append(f"{task}_{length}")

    # Metric types row (mirrors get_eval.py's output convention).
    metric_types = ["task"] + ["acc"] * (len(columns) - 1)

    print(",".join(columns))
    print(",".join(metric_types))

    for row in rows:
        values: List[str] = []
        for col in columns:
            val = row.get(col)
            if col == "model":
                values.append(str(val) if val is not None else "")
            else:
                values.append(
                    _format_value(val if isinstance(val, (int, float)) else None)
                )

        if args.full == "True":
            print(",".join(values))
        else:
            if values[-1] != "":
                print(",".join(values))

    return rows


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
    parser.add_argument(
        "--include_cwe_fwe_vt",
        type=str,
        default="False",
        help='Whether to include ruler_vt / ruler_cwe / ruler_fwe in the output columns. Use "True" to include.',
    )
    # Deprecated alias (kept for backward compatibility).
    parser.add_argument(
        "--include_cwe_fwe",
        type=str,
        default="False",
        help="[DEPRECATED] Use --include_cwe_fwe_vt. Kept for backward compatibility.",
    )
    args = parser.parse_args()

    process_logs(args)
