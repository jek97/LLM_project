import re
from collections import defaultdict

log_file_path = "./gpt_outputs_0/ollama_outputs.txt"  # <- change this if needed

# Read the log file
with open(log_file_path, "r") as f:
    log_text = f.read()

# Regular expression to match each log entry line
log_pattern = re.compile(
    r"input (\d+), model ([^,]+), multimodal (\w+), temp ([^,]+), question_time ([^,]+),(.*?)succ (\w+)",
    re.DOTALL
)

# Pattern to extract retry information
retry_pattern = re.compile(r"answer ([^,]+), ret ([^,]+)")

# Data structures to store retry times by model, temperature, and mode
multimodal_data = defaultdict(lambda: defaultdict(list))
non_multimodal_data = defaultdict(lambda: defaultdict(list))

# Process each log entry
for match in log_pattern.finditer(log_text):
    if not match:
        continue
    
    model = match.group(2)
    multimodal = match.group(3) == "True"
    temp = float(match.group(4))
    retry_text = match.group(6)
    
    retries = retry_pattern.findall(retry_text)
    retry_times = [float(r[1]) for r in retries]
    
    if not retry_times:
        continue
    
    # Add each retry time to the appropriate collection based on multimodal flag
    for retry_time in retry_times:
        if multimodal:
            multimodal_data[model][temp].append(retry_time)
        else:
            non_multimodal_data[model][temp].append(retry_time)

# Print header
print("\n" + "="*80)
print("AVERAGE RETRY TIME ANALYSIS BY MODEL, TEMPERATURE AND MODE")
print("="*80)

# Print multimodal results
print("\nMULTIMODAL MODE RESULTS:")
print("="*60)

# Print multimodal results by model and temperature
for model in sorted(multimodal_data.keys()):
    print(f"\nModel: {model}")
    print("-" * 50)
    print("{:<15} {:<15} {:<12} {:<15}".format("Temperature", "Avg Time (s)", "Count", "Total Time (s)"))
    print("-" * 50)
    
    # Sort temperatures for consistent output
    model_avg_time = 0
    model_total_retries = 0
    model_total_time = 0
    
    for temp in sorted(multimodal_data[model].keys()):
        times = multimodal_data[model][temp]
        count = len(times)
        avg_time = sum(times) / count
        total_time = sum(times)
        
        print("{:<15.2f} {:<15.4f} {:<12} {:<15.4f}".format(temp, avg_time, count, total_time))
        
        model_total_retries += count
        model_total_time += total_time
    
    # Calculate model-wide average
    model_avg_time = model_total_time / model_total_retries if model_total_retries > 0 else 0
    print("-" * 50)
    print("{:<15} {:<15.4f} {:<12} {:<15.4f}".format(
        "Model Average", model_avg_time, model_total_retries, model_total_time
    ))

# Print non-multimodal results
print("\n\nNON-MULTIMODAL MODE RESULTS:")
print("="*60)

# Print non-multimodal results by model and temperature
for model in sorted(non_multimodal_data.keys()):
    print(f"\nModel: {model}")
    print("-" * 50)
    print("{:<15} {:<15} {:<12} {:<15}".format("Temperature", "Avg Time (s)", "Count", "Total Time (s)"))
    print("-" * 50)
    
    # Sort temperatures for consistent output
    model_avg_time = 0
    model_total_retries = 0
    model_total_time = 0
    
    for temp in sorted(non_multimodal_data[model].keys()):
        times = non_multimodal_data[model][temp]
        count = len(times)
        avg_time = sum(times) / count
        total_time = sum(times)
        
        print("{:<15.2f} {:<15.4f} {:<12} {:<15.4f}".format(temp, avg_time, count, total_time))
        
        model_total_retries += count
        model_total_time += total_time
    
    # Calculate model-wide average
    model_avg_time = model_total_time / model_total_retries if model_total_retries > 0 else 0
    print("-" * 50)
    print("{:<15} {:<15.4f} {:<12} {:<15.4f}".format(
        "Model Average", model_avg_time, model_total_retries, model_total_time
    ))

# Comparative analysis: temperature impact across models
print("\n\nCOMPARATIVE ANALYSIS: TEMPERATURE IMPACT")
print("="*70)

# Collect all temperatures
all_temps = set()
for model_data in [multimodal_data, non_multimodal_data]:
    for model in model_data:
        for temp in model_data[model]:
            all_temps.add(temp)

all_temps = sorted(all_temps)

# Print temperature impact for multimodal mode
print("\nMultimodal Mode - Average Retry Time by Temperature:")
print("-" * 70)
print("{:<15}".format("Temperature"), end="")
for model in sorted(multimodal_data.keys()):
    print("{:<15}".format(model), end="")
print()
print("-" * 70)

for temp in all_temps:
    print("{:<15.2f}".format(temp), end="")
    for model in sorted(multimodal_data.keys()):
        times = multimodal_data[model].get(temp, [])
        avg_time = sum(times) / len(times) if times else float('nan')
        avg_time_str = "{:.4f}".format(avg_time) if times else "N/A"
        print("{:<15}".format(avg_time_str), end="")
    print()

# Print temperature impact for non-multimodal mode
print("\nNon-Multimodal Mode - Average Retry Time by Temperature:")
print("-" * 70)
print("{:<15}".format("Temperature"), end="")
for model in sorted(non_multimodal_data.keys()):
    print("{:<15}".format(model), end="")
print()
print("-" * 70)

for temp in all_temps:
    print("{:<15.2f}".format(temp), end="")
    for model in sorted(non_multimodal_data.keys()):
        times = non_multimodal_data[model].get(temp, [])
        avg_time = sum(times) / len(times) if times else float('nan')
        avg_time_str = "{:.4f}".format(avg_time) if times else "N/A"
        print("{:<15}".format(avg_time_str), end="")
    print()

# Calculate overall statistics
print("\n\nOVERALL SUMMARY:")
print("="*50)

# Get all unique models
all_models = sorted(set(list(multimodal_data.keys()) + list(non_multimodal_data.keys())))

# Print model comparison table
print("\nAverage Retry Time Comparison by Model:")
print("-"*70)
print("{:<20} {:<15} {:<15} {:<15}".format(
    "Model", "Multimodal (s)", "Non-Multimodal (s)", "Difference (s)"
))
print("-"*70)

for model in all_models:
    # Calculate multimodal average
    mm_times = []
    for temp_times in multimodal_data[model].values():
        mm_times.extend(temp_times)
    mm_avg = sum(mm_times) / len(mm_times) if mm_times else float('nan')
    
    # Calculate non-multimodal average
    non_mm_times = []
    for temp_times in non_multimodal_data[model].values():
        non_mm_times.extend(temp_times)
    non_mm_avg = sum(non_mm_times) / len(non_mm_times) if non_mm_times else float('nan')
    
    # Calculate difference
    if mm_times and non_mm_times:
        diff = mm_avg - non_mm_avg
        diff_str = "{:.4f}".format(diff)
    else:
        diff_str = "N/A"
    
    mm_avg_str = "{:.4f}".format(mm_avg) if mm_times else "N/A"
    non_mm_avg_str = "{:.4f}".format(non_mm_avg) if non_mm_times else "N/A"
    
    print("{:<20} {:<15} {:<15} {:<15}".format(model, mm_avg_str, non_mm_avg_str, diff_str))

# Grand total statistics
all_mm_times = []
all_non_mm_times = []

for model in multimodal_data:
    for temp in multimodal_data[model]:
        all_mm_times.extend(multimodal_data[model][temp])

for model in non_multimodal_data:
    for temp in non_multimodal_data[model]:
        all_non_mm_times.extend(non_multimodal_data[model][temp])

overall_mm_avg = sum(all_mm_times) / len(all_mm_times) if all_mm_times else 0
overall_non_mm_avg = sum(all_non_mm_times) / len(all_non_mm_times) if all_non_mm_times else 0

print("\nGrand Totals:")
print("-"*50)
print(f"Overall Multimodal Average Retry Time: {overall_mm_avg:.4f} seconds")
print(f"Overall Non-Multimodal Average Retry Time: {overall_non_mm_avg:.4f} seconds")

if all_mm_times and all_non_mm_times:
    print(f"Overall Difference: {overall_mm_avg - overall_non_mm_avg:.4f} seconds")

print("\n" + "="*80)
print("End of Analysis")
print("="*80)