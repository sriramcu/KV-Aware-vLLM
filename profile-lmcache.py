import csv
import os
from prometheus_client import CollectorRegistry
from prometheus_client import multiprocess as multiproc 

def save_stats_to_csv(filename="lmcache-ray1q-_results.csv"):
    # 1. Define and Set the path
    multiproc_dir = '/mnt/shared/gpfs/home/seliny2/.cache/vllm'
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = multiproc_dir
    
    if not os.path.exists(multiproc_dir):
        print(f"Error: {multiproc_dir} not found.")
        return

    # 2. Initialize Registry and Collector
    registry = CollectorRegistry()
    # This specifically aggregates all those PIDs (637177, etc.) into one set of metrics
    multiproc.MultiProcessCollector(registry)
    
    # 3. Extract and Clean LMCache metrics
    metrics_data = {}
    for metric in registry.collect():
        if "lmcache" in metric.name:
            clean_name = metric.name.replace("lmcache:", "")
            if metric.samples:
                # Summing samples if multiple PIDs reported the same metric
                metrics_data[clean_name] = sum(s.value for s in metric.samples)
    
    if not metrics_data:
        print("No LMCache metrics found. Ensure the engine actually ran prompts.")
        return

    # 4. Sort and Write
    sorted_keys = sorted(metrics_data.keys())
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted_keys)
        if not file_exists:
            w.writeheader()
        w.writerow(metrics_data)
        
    print(f"✅ Success! Results saved to {filename}")

if __name__ == "__main__":
    save_stats_to_csv()