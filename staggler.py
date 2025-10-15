# import re
# import numpy as np
#
# # Read log file
# with open("./logs/log_2025-09-27_14-29-20.log", "r") as f:
#     log = f.read()
#
# # Find all CPU worker blocks: ([Worker-XX / CPU] ... Duration: NNN.NNs)
# pattern = re.compile(
#     r"(\[Worker-\d+\s*/\s*CPU\][\s\S]*?Duration:\s*([0-9]+\.[0-9]+)s)",
#     re.MULTILINE
# )
#
# cpu_durations = [float(m.group(2)) for m in pattern.finditer(log)]
# print(f"Found {len(cpu_durations)} CPU durations: {cpu_durations}")
#
# if cpu_durations:
#     median = np.median(cpu_durations)
#     threshold = 2.0  # Straggler = >2x median
#     stragglers = [d for d in cpu_durations if d > threshold * median]
#     print(f"Median CPU duration: {median:.2f} s")
#     print(f"Straggler threshold: {threshold * median:.2f} s")
#     print(f"CPU Straggler durations: {stragglers}")
# else:
#     print("No CPU durations found in the log.")

# test_numa_affinity.py
import psutil
import os
import time


def check_worker_affinities():
    """Check CPU affinity of all Python workers"""
    python_processes = []

    for proc in psutil.process_iter(["pid", "name", "cpu_affinity"]):
        if "python" in proc.info["name"].lower():
            try:
                affinity = proc.cpu_affinity()
                python_processes.append((proc.info["pid"], affinity))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    return python_processes


# Monitor for 30 seconds
print("Monitoring worker CPU affinities for 30 seconds...")
print("If you see frequent changes, that's the problem!")

for i in range(30):
    affinities = check_worker_affinities()
    print(f"Time {i}s: {affinities}")
    time.sleep(1)
