import os
import re
import csv

LOG_DIR = "benchmarks/logs"
if not os.path.exists(LOG_DIR) and os.path.exists("logs"):
    LOG_DIR = "logs"

def parse_log(filename):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        content = f.read()

    metrics = []
    # Extract common metrics pattern
    # [Mask 0x01] Cycles: 69905, Cost: 2130939.00, Eff: 0.0000, Hits: 2, Miss: 0
    # Run 0: Cost: 46011513.00, Efficiency: 0.0000, Residency Hits: 0
    # Density 0.00: Skip Ratio 0.0%, Eff: 0.0000, Cost: 1468007.00

    # Generic extractor
    res = re.findall(r"Cost: ([\d.]+)", content)
    if res:
        metrics.append(("Cost", float(res[-1])))

    res = re.findall(r"Eff(?:iciency)?: ([\d.]+)", content)
    if res:
        metrics.append(("Efficiency", float(res[-1])))

    res = re.findall(r"Cycles: (\d+)", content)
    if res:
        metrics.append(("Cycles", int(res[-1])))

    res = re.findall(r"Hits: (\d+)", content)
    if res:
        metrics.append(("Hits", int(res[-1])))

    res = re.findall(r"Residency Hits: (\d+)", content)
    if res:
        metrics.append(("Hits", int(res[-1])))

    res = re.findall(r"Skip Ratio ([\d.]+)%", content)
    if res:
        metrics.append(("SkipRatio", float(res[-1])))

    return dict(metrics)

def main():
    results = []
    for filename in os.listdir(LOG_DIR):
        if not filename.endswith(".log"):
            continue

        data = parse_log(filename)
        if data:
            data['filename'] = filename
            results.append(data)

    # Write to summary CSV
    out_path = "benchmarks/summary.csv"
    if not os.path.exists("benchmarks"):
        out_path = "summary.csv"
    with open(out_path, "w", newline="") as f:
        if not results:
            return
        all_keys = set()
        for r in results: all_keys.update(r.keys())
        writer = csv.DictWriter(f, fieldnames=sorted(list(all_keys)))
        writer.writeheader()
        writer.writerows(results)

    print(f"Parsed {len(results)} logs into benchmarks/summary.csv")

if __name__ == "__main__":
    main()
