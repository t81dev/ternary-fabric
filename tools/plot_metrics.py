import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(csv_path, output_dir="plots"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Efficiency vs Cost
    plt.figure(figsize=(10, 6))
    plt.scatter(df['fabric_cost'], df['semantic_efficiency'], color='blue', alpha=0.6)
    plt.title('Semantic Efficiency vs Fabric Cost')
    plt.xlabel('Fabric Cost')
    plt.ylabel('Semantic Efficiency (Useful Ops / Cost)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'efficiency_vs_cost.png'))

    # 2. Residency Hit Ratio
    if 'residency_hits' in df.columns and 'residency_misses' in df.columns:
        plt.figure(figsize=(8, 8))
        total = df['residency_hits'].sum() + df['residency_misses'].sum()
        if total > 0:
            labels = ['Hits', 'Misses']
            sizes = [df['residency_hits'].sum(), df['residency_misses'].sum()]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['green', 'red'])
            plt.title('Overall Residency Hit/Miss Ratio')
            plt.savefig(os.path.join(output_dir, 'residency_ratio.png'))

    # 3. Cost Breakdown
    plt.figure(figsize=(12, 6))
    metrics = ['mem_reads', 'mem_writes', 'broadcasts', 'active_ops']
    values = [df[m].sum() for m in metrics]
    plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen', 'gold'])
    plt.title('Fabric Activity Breakdown')
    plt.ylabel('Total Count')
    plt.savefig(os.path.join(output_dir, 'activity_breakdown.png'))

    # 4. Learning Trends (Phase 20)
    if 'weight_cost' in df.columns and 'step' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['weight_cost'], label='Weight Cost')
        plt.plot(df['step'], df['residency_miss_cost'], label='Residency Miss Cost')
        plt.title('Adaptive Cost Coefficients Evolution')
        plt.xlabel('Step')
        plt.ylabel('Coefficient Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_coeffs.png'))

    if 'batch_size' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['batch_size'], color='purple', marker='o', linestyle='-', alpha=0.5)
        plt.title('Dynamic Batch Size Evolution')
        plt.xlabel('Step')
        plt.ylabel('Batch Size')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_batch.png'))

    print(f"Plots generated in {output_dir}/")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "run_metrics.csv"
    plot_metrics(csv_file)
