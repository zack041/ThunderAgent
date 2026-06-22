#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

STOP = False
PROM_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([^\s]+)$")
PROM_NAMES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
]


def stop(*_args):
    global STOP
    STOP = True


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def scrape(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_prometheus(text):
    values = {name: 0.0 for name in PROM_NAMES}
    present = set()
    for line in text.splitlines():
        match = PROM_RE.match(line)
        if not match or match.group(1) not in values:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        values[match.group(1)] += value
        present.add(match.group(1))
    return {name: values[name] if name in present else "" for name in PROM_NAMES}


def read_meminfo():
    data = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            data[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        pass
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "system_metrics.csv"
    raw_path = output / "vllm_metrics.prom.jsonl.gz"
    fields = [
        "timestamp",
        "gpu_util_pct",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_power_w",
        "gpu_temperature_c",
        "load1",
        "load5",
        "load15",
        "host_memory_used_bytes",
        "host_memory_total_bytes",
        "disk_used_bytes",
        "disk_total_bytes",
        "docker_running_containers",
        *PROM_NAMES,
    ]

    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        while not STOP:
            started = time.time()
            row = {field: "" for field in fields}
            row["timestamp"] = started

            gpu = command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            if gpu:
                samples = []
                for line in gpu.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 5:
                        continue
                    try:
                        samples.append([float(part) for part in parts])
                    except ValueError:
                        continue
                if samples:
                    # Utilization is averaged across GPUs. Memory and power are
                    # summed so TP runs report aggregate device usage. The
                    # hottest GPU is retained for temperature.
                    row["gpu_util_pct"] = sum(item[0] for item in samples) / len(samples)
                    row["gpu_memory_used_mib"] = sum(item[1] for item in samples)
                    row["gpu_memory_total_mib"] = sum(item[2] for item in samples)
                    row["gpu_power_w"] = sum(item[3] for item in samples)
                    row["gpu_temperature_c"] = max(item[4] for item in samples)

            try:
                row["load1"], row["load5"], row["load15"] = os.getloadavg()
            except Exception:
                pass
            mem = read_meminfo()
            if mem:
                row["host_memory_total_bytes"] = mem.get("MemTotal", 0)
                row["host_memory_used_bytes"] = mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)
            disk = shutil.disk_usage("/")
            row["disk_used_bytes"] = disk.used
            row["disk_total_bytes"] = disk.total
            docker_count = command(["docker", "ps", "-q"])
            row["docker_running_containers"] = len(docker_count.splitlines()) if docker_count else 0

            raw = scrape("http://127.0.0.1:8100/metrics")
            if raw:
                row.update(parse_prometheus(raw))
                with gzip.open(raw_path, "at") as raw_file:
                    raw_file.write(json.dumps({"timestamp": started, "text": raw}) + "\n")

            writer.writerow(row)
            csv_file.flush()
            time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
