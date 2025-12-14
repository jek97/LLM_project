import re
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

log_file_path = "./gpt_outputs_0/ollama_outputs.txt"  # <- change this if needed

# Read the log file
with open(log_file_path, "r") as f:
    log_text = f.read()

# Regular expression to match each input block
input_pattern = re.compile(r"(input \d+.*?succ (?:True|False))", re.DOTALL)

# Pattern to extract structured data
entry_pattern = re.compile(
    r"input (\d+), model ([^,]+), multimodal (\w+), temp ([^,]+), question_time ([\d.]+),"
    r"(.*?)succ (True|False)", re.DOTALL
)
retry_pattern = re.compile(r"answer ([^,]+), ret ([\d.]+)")

results = []

for block in input_pattern.findall(log_text):
    match = entry_pattern.search(block)
    if not match:
        continue

    input_num = int(match.group(1))
    model = match.group(2)
    multimodal = match.group(3) == "True"
    temp = float(match.group(4))
    question_time = float(match.group(5))
    retry_text = match.group(6)
    succ = match.group(7) == "True"

    retries = retry_pattern.findall(retry_text)
    retry_paths = [r[0] for r in retries]
    retry_times = [float(r[1]) for r in retries]

    result = {
        "input_num": input_num,
        "model": model,
        "multimodal": multimodal,
        "temp": temp,
        "question_time": question_time,
        "retry_count": len(retries),
        "retry_total_time": sum(retry_times),
        "retry_avg_time": sum(retry_times) / len(retry_times) if retries else 0,
        "retry_min_time": min(retry_times) if retries else 0,
        "retry_max_time": max(retry_times) if retries else 0,
        "succ": succ,
        "retry_paths": retry_paths,
    }

    results.append(result)

# ?? Aggregate average question_time by model and temp
agg_data = defaultdict(lambda: defaultdict(list))

for r in results:
    model = r["model"]
    temp = r["temp"]
    qtime = r["question_time"]
    agg_data[model][temp].append(qtime)

# Compute average question time
avg_data = {}
for model, temp_dict in agg_data.items():
    avg_data[model] = {
        temp: sum(times) / len(times)
        for temp, times in sorted(temp_dict.items())
    }

# ?? Plotting
plt.figure(figsize=(10, 6))

for model, temp_qtime_dict in avg_data.items():
    temps = list(temp_qtime_dict.keys())
    avg_qtimes = list(temp_qtime_dict.values())
    plt.plot(temps, avg_qtimes, marker='o', label=model)

plt.xlabel("Temperature")
plt.ylabel("Average Question Time (s)")
plt.title("Model Question Time vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("question_time_vs_temp.png", dpi=300)

# # ?? Aggregate average retry time by model and temp
# retry_agg_data = defaultdict(lambda: defaultdict(list))

# for r in results:
#     model = r["model"]
#     temp = r["temp"]
#     retry_avg = r["retry_avg_time"]
#     if retry_avg > 0:  # Skip if there were no retries
#         retry_agg_data[model][temp].append(retry_avg)

# # Compute average retry time
# retry_avg_data = {}
# for model, temp_dict in retry_agg_data.items():
#     retry_avg_data[model] = {
#         temp: sum(times) / len(times)
#         for temp, times in sorted(temp_dict.items())
#     }

# # ?? Plotting
# plt.figure(figsize=(10, 6))

# for model, temp_retry_dict in retry_avg_data.items():
#     temps = list(temp_retry_dict.keys())
#     avg_retry_times = list(temp_retry_dict.values())
#     plt.plot(temps, avg_retry_times, marker='o', label=model)

# plt.xlabel("Temperature")
# plt.ylabel("Average Retry Time (s)")
# plt.title("Model Retry Time vs Temperature")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("question_ret_time_vs_temp.png", dpi=300)
