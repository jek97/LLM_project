import re
from collections import defaultdict

log_file_path = "./gpt_outputs_0/ollama_outputs.txt"  # <- change this if needed

# Read the log file
with open(log_file_path, "r") as f:
    log_text = f.read()

# Regular expression to match each log entry line
# Updated pattern based on the actual file format
log_pattern = re.compile(
    r"input (\d+), model ([^,]+), multimodal (\w+), temp ([^,]+), question_time ([^,]+),(.*?)succ (\w+)",
    re.DOTALL
)

# Data structures to store question times (first answer times)
# Format: model -> temperature -> list of question times
multimodal_question_times = defaultdict(lambda: defaultdict(list))
non_multimodal_question_times = defaultdict(lambda: defaultdict(list))

# Process each log entry
entries_count = 0
for match in log_pattern.finditer(log_text):
    if not match:
        continue
    
    entries_count += 1
    model = match.group(2)
    multimodal = match.group(3) == "True"
    temp = float(match.group(4))
    question_time = float(match.group(5))  # This is the first answer time
    
    # Store the question time in the appropriate collection
    if multimodal:
        multimodal_question_times[model][temp].append(question_time)
    else:
        non_multimodal_question_times[model][temp].append(question_time)

# Print header
print("\n" + "="*80)
print("AVERAGE FIRST ANSWER TIME ANALYSIS (QUESTION TIME)")
print("="*80)
print(f"Total log entries processed: {entries_count}")

# Print multimodal results
print("\nMULTIMODAL MODE RESULTS:")
print("="*60)

# Print multimodal results by model and temperature
for model in sorted(multimodal_question_times.keys()):
    print(f"\nModel: {model}")
    print("-" * 50)
    print("{:<15} {:<15} {:<12} {:<15}".format("Temperature", "Avg Time (s)", "Count", "Total Time (s)"))
    print("-" * 50)
    
    # Sort temperatures for consistent output
    model_times = []
    
    for temp in sorted(multimodal_question_times[model].keys()):
        times = multimodal_question_times[model][temp]
        count = len(times)
        avg_time = sum(times) / count if count > 0 else 0
        total_time = sum(times)
        
        print("{:<15.2f} {:<15.4f} {:<12} {:<15.4f}".format(temp, avg_time, count, total_time))
        model_times.extend(times)
    
    # Calculate model-wide average
    model_count = len(model_times)
    model_avg_time = sum(model_times) / model_count if model_count > 0 else 0
    model_total_time = sum(model_times)
    
    print("-" * 50)
    print("{:<15} {:<15.4f} {:<12} {:<15.4f}".format(
        "Model Average", model_avg_time, model_count, model_total_time
    ))

# Print non-multimodal results
print("\n\nNON-MULTIMODAL MODE RESULTS:")
print("="*60)

# Print non-multimodal results by model and temperature
for model in sorted(non_multimodal_question_times.keys()):
    print(f"\nModel: {model}")
    print("-" * 50)
    print("{:<15} {:<15} {:<12} {:<15}".format("Temperature", "Avg Time (s)", "Count", "Total Time (s)"))
    print("-" * 50)
    
    # Sort temperatures for consistent output
    model_times = []
    
    for temp in sorted(non_multimodal_question_times[model].keys()):
        times = non_multimodal_question_times[model][temp]
        count = len(times)
        avg_time = sum(times) / count if count > 0 else 0
        total_time = sum(times)
        
        print("{:<15.2f} {:<15.4f} {:<12} {:<15.4f}".format(temp, avg_time, count, total_time))
        model_times.extend(times)
    
    # Calculate model-wide average
    model_count = len(model_times)
    model_avg_time = sum(model_times) / model_count if model_count > 0 else 0
    model_total_time = sum(model_times)
    
    print("-" * 50)
    print("{:<15} {:<15.4f} {:<12} {:<15.4f}".format(
        "Model Average", model_avg_time, model_count, model_total_time
    ))

# Comparative analysis: temperature impact across models
print("\n\nCOMPARATIVE ANALYSIS: QUESTION TIME BY TEMPERATURE")
print("="*70)

# Collect all temperatures
all_temps = set()
for model_data in [multimodal_question_times, non_multimodal_question_times]:
    for model in model_data:
        for temp in model_data[model]:
            all_temps.add(temp)

all_temps = sorted(all_temps)

# Print temperature impact for multimodal mode
print("\nMultimodal Mode - Average Question Time by Temperature:")
print("-" * 70)
print("{:<15}".format("Temperature"), end="")
for model in sorted(multimodal_question_times.keys()):
    print("{:<15}".format(model), end="")
print()
print("-" * 70)

for temp in all_temps:
    print("{:<15.2f}".format(temp), end="")
    for model in sorted(multimodal_question_times.keys()):
        times = multimodal_question_times[model].get(temp, [])
        avg_time = sum(times) / len(times) if times else float('nan')
        avg_time_str = "{:.4f}".format(avg_time) if times else "N/A"
        print("{:<15}".format(avg_time_str), end="")
    print()

# Print temperature impact for non-multimodal mode
print("\nNon-Multimodal Mode - Average Question Time by Temperature:")
print("-" * 70)
print("{:<15}".format("Temperature"), end="")
for model in sorted(non_multimodal_question_times.keys()):
    print("{:<15}".format(model), end="")
print()
print("-" * 70)

for temp in all_temps:
    print("{:<15.2f}".format(temp), end="")
    for model in sorted(non_multimodal_question_times.keys()):
        times = non_multimodal_question_times[model].get(temp, [])
        avg_time = sum(times) / len(times) if times else float('nan')
        avg_time_str = "{:.4f}".format(avg_time) if times else "N/A"
        print("{:<15}".format(avg_time_str), end="")
    print()

# Calculate overall statistics
print("\n\nOVERALL SUMMARY:")
print("="*50)

# Get all unique models
all_models = sorted(set(list(multimodal_question_times.keys()) + list(non_multimodal_question_times.keys())))

# Print model comparison table
print("\nAverage Question Time Comparison by Model:")
print("-"*70)
print("{:<20} {:<15} {:<15} {:<15}".format(
    "Model", "Multimodal (s)", "Non-Multimodal (s)", "Difference (s)"
))
print("-"*70)

for model in all_models:
    # Calculate multimodal average
    mm_times = []
    for temp_times in multimodal_question_times[model].values():
        mm_times.extend(temp_times)
    mm_avg = sum(mm_times) / len(mm_times) if mm_times else float('nan')
    
    # Calculate non-multimodal average
    non_mm_times = []
    for temp_times in non_multimodal_question_times[model].values():
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

for model in multimodal_question_times:
    for temp in multimodal_question_times[model]:
        all_mm_times.extend(multimodal_question_times[model][temp])

for model in non_multimodal_question_times:
    for temp in non_multimodal_question_times[model]:
        all_non_mm_times.extend(non_multimodal_question_times[model][temp])

overall_mm_avg = sum(all_mm_times) / len(all_mm_times) if all_mm_times else 0
overall_non_mm_avg = sum(all_non_mm_times) / len(all_non_mm_times) if all_non_mm_times else 0

print("\nGrand Totals:")
print("-"*50)
print(f"Overall Multimodal Average Question Time: {overall_mm_avg:.4f} seconds")
print(f"Overall Non-Multimodal Average Question Time: {overall_non_mm_avg:.4f} seconds")

if all_mm_times and all_non_mm_times:
    print(f"Overall Difference: {overall_mm_avg - overall_non_mm_avg:.4f} seconds")

print("\n" + "="*80)
print("End of Analysis")
print("="*80)