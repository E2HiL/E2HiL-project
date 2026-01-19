#!/usr/bin/env python3
"""
Analyze all cov_step_*.json files under cov_logs and summarize cov_per_sample by tag.
Helps compare how tag 0 vs tag 1 samples contribute to entropy reduction.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from collections import defaultdict


def load_cov_data(cov_logs_dir: str) -> Dict:
    """
    Load all cov_step_*.json files.
    
    Args:
        cov_logs_dir: cov_logs directory path
        
    Returns:
        Dictionary keyed by step number with loaded data
    """
    all_data = {}
    
    # Collect all JSON files
    json_files = glob.glob(os.path.join(cov_logs_dir, "cov_step_*.json"))
    json_files.sort(key=lambda x: int(
        os.path.basename(x).replace("cov_step_", "").replace(".json", "")
    ))
    
    print(f"Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        try:
            basename = os.path.basename(file_path)
            step_num = int(basename.replace("cov_step_", "").replace(".json", ""))
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data[step_num] = data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    return all_data

def analyze_by_tag(all_data: Dict) -> Dict:
    """
    Group and analyze cov_per_sample values by tag.
    
    Args:
        all_data: Data from all steps
        
    Returns:
        Aggregated analysis results
    """
    # Collect data for each tag
    tag_data = defaultdict(list)
    step_stats = []
    
    for step, data in all_data.items():
        samples = data.get('samples', [])
        step_tag_0 = []
        step_tag_1 = []
        
        for sample in samples:
            tag = sample.get('tag')
            cov_value = sample.get('cov_per_sample')
            
            if tag is not None and cov_value is not None:
                tag_data[tag].append(cov_value)
                
                if tag == 0:
                    step_tag_0.append(cov_value)
                elif tag == 1:
                    step_tag_1.append(cov_value)
        
        # Record per-step statistics
        step_info = {
            'step': step,
            'tag_0_count': len(step_tag_0),
            'tag_1_count': len(step_tag_1),
            'tag_0_mean': np.mean(step_tag_0) if step_tag_0 else np.nan,
            'tag_1_mean': np.mean(step_tag_1) if step_tag_1 else np.nan,
            'tag_0_std': np.std(step_tag_0) if step_tag_0 else np.nan,
            'tag_1_std': np.std(step_tag_1) if step_tag_1 else np.nan,
        }
        step_stats.append(step_info)
    
    # Compute overall statistics
    results = {}
    for tag in [0, 1]:
        if tag in tag_data:
            values = np.array(tag_data[tag])
            results[f'tag_{tag}'] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'values': values.tolist()
            }
    
    results['step_stats'] = step_stats
    return results

def print_summary(results: Dict):
    """Print a concise summary of the analysis."""
    print("\n" + "="*60)
    print("CoV Per Sample Summary")
    print("="*60)
    
    for tag in [0, 1]:
        tag_key = f'tag_{tag}'
        if tag_key in results:
            data = results[tag_key]
            print(f"\nTag {tag} sample stats:")
            print(f"  Sample count: {data['count']:,}")
            print(f"  Mean: {data['mean']:.6f}")
            print(f"  Std: {data['std']:.6f}")
            print(f"  Min: {data['min']:.6f}")
            print(f"  Max: {data['max']:.6f}")
            print(f"  Median: {data['median']:.6f}")
            print(f"  25th percentile: {data['q25']:.6f}")
            print(f"  75th percentile: {data['q75']:.6f}")
    
    # Compare the two tags
    if 'tag_0' in results and 'tag_1' in results:
        print(f"\nComparison:")
        tag0_mean = results['tag_0']['mean']
        tag1_mean = results['tag_1']['mean']
        print(f"  Tag 1 vs Tag 0 mean difference: {tag1_mean - tag0_mean:.6f}")
        print(f"  Is Tag 1 mean higher?: {'Yes' if tag1_mean > tag0_mean else 'No'}")
        
        # Effect size (Cohen's d)
        tag0_std = results['tag_0']['std']
        tag1_std = results['tag_1']['std']
        pooled_std = np.sqrt((tag0_std**2 + tag1_std**2) / 2)
        cohens_d = (tag1_mean - tag0_mean) / pooled_std
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")

def create_visualizations(results: Dict, output_dir: str):
    """Create visualization figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Font setup
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribution comparison box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    if 'tag_0' in results and 'tag_1' in results:
        tag0_values = results['tag_0']['values']
        tag1_values = results['tag_1']['values']
        
        box_data = [tag0_values, tag1_values]
        labels = ['Tag 0', 'Tag 1']
        
        ax1.boxplot(box_data, labels=labels)
        ax1.set_title('Distribution comparison of CoV Per Sample (Box plot)')
        ax1.set_ylabel('CoV Per Sample')
        ax1.grid(True, alpha=0.3)
        
        # Histogram comparison
        ax2.hist(tag0_values, bins=50, alpha=0.7, label='Tag 0', density=True)
        ax2.hist(tag1_values, bins=50, alpha=0.7, label='Tag 1', density=True)
        ax2.set_title('Distribution comparison of CoV Per Sample (Histogram)')
        ax2.set_xlabel('CoV Per Sample')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cov_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Trends over training steps
    step_stats = results.get('step_stats', [])
    if step_stats:
        df_steps = pd.DataFrame(step_stats)
        df_steps = df_steps.sort_values('step')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean trend
        ax1.plot(df_steps['step'], df_steps['tag_0_mean'], label='Tag 0', marker='o', markersize=2)
        ax1.plot(df_steps['step'], df_steps['tag_1_mean'], label='Tag 1', marker='o', markersize=2)
        ax1.set_title('CoV Per Sample Mean over Training Steps')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean CoV Per Sample')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Std trend
        ax2.plot(df_steps['step'], df_steps['tag_0_std'], label='Tag 0', marker='o', markersize=2)
        ax2.plot(df_steps['step'], df_steps['tag_1_std'], label='Tag 1', marker='o', markersize=2)
        ax2.set_title('CoV Per Sample Std over Training Steps')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('CoV Per Sample Std')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample count trend
        ax3.plot(df_steps['step'], df_steps['tag_0_count'], label='Tag 0', marker='o', markersize=2)
        ax3.plot(df_steps['step'], df_steps['tag_1_count'], label='Tag 1', marker='o', markersize=2)
        ax3.set_title('Sample Count over Training Steps')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Sample Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mean difference trend
        mean_diff = df_steps['tag_1_mean'] - df_steps['tag_0_mean']
        ax4.plot(df_steps['step'], mean_diff, marker='o', markersize=2, color='purple')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('The average difference between Tag 1 and Tag 0 varies with the training steps')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Average Difference (Tag 1 - Tag 0)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cov_trends_over_steps.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_detailed_results(results: Dict, output_file: str):
    """Save detailed results to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CoV Per Sample Analysis\n")
        f.write("="*60 + "\n\n")
        
        for tag in [0, 1]:
            tag_key = f'tag_{tag}'
            if tag_key in results:
                data = results[tag_key]
                f.write(f"  Tag {tag} Statistical results:\n")
                f.write(f"  Number: {data['count']:,}\n")
                f.write(f"  Mean: {data['mean']:.8f}\n")
                f.write(f"  Std: {data['std']:.8f}\n")
                f.write(f"  Min: {data['min']:.8f}\n")
                f.write(f"  Max: {data['max']:.8f}\n")
                f.write(f"  Median: {data['median']:.8f}\n")
                f.write(f"  25% Quantile: {data['q25']:.8f}\n")
                f.write(f"  75% Quantile: {data['q75']:.8f}\n\n")
        
        # Per-step statistics
        f.write("\n Statistical results by step:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Step':<8} {'Tag0 Number':<10} {'Tag1 Number':<10} {'Tag0 Mean':<12} {'Tag1 Mean':<12} {'Difference':<12}\n")
        f.write("-" * 80 + "\n")
        
        step_stats = results.get('step_stats', [])
        for step_info in sorted(step_stats, key=lambda x: x['step']):
            step = step_info['step']
            tag0_count = step_info['tag_0_count']
            tag1_count = step_info['tag_1_count']
            tag0_mean = step_info['tag_0_mean']
            tag1_mean = step_info['tag_1_mean']
            diff = tag1_mean - tag0_mean if not (np.isnan(tag0_mean) or np.isnan(tag1_mean)) else np.nan
            
            f.write(f"{step:<8} {tag0_count:<10} {tag1_count:<10} {tag0_mean:<12.6f} {tag1_mean:<12.6f} {diff:<12.6f}\n")

def main():
    """Entrypoint for the analysis script."""
    # Paths
    cov_logs_dir = "/home/luka/lerobot_srl/outputs/HIL-SERL-so101/cov_logs"
    output_dir = "/home/luka/lerobot_srl/outputs/HIL-SERL-so101/analysis_results"
    
    print("Start analyzing CoV log data...")
    print(f"Data directory: {cov_logs_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    all_data = load_cov_data(cov_logs_dir)
    
    if not all_data:
        print("No valid data files found!")
        return
    
    print(f"Loaded data for {len(all_data)} steps")
    
    # Analyze data
    results = analyze_by_tag(all_data)
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    print("\nCreating visualization figures...")
    create_visualizations(results, output_dir)
    
    # Save detailed results
    output_file = os.path.join(output_dir, "detailed_analysis_results.txt")
    save_detailed_results(results, output_file)
    
    print(f"\nAnalysis completed!")
    print(f"Visualization files saved to: {output_dir}")
    print(f"Detailed results saved to: {output_file}")
    
    # Key findings
    print("\nKey findings:")
    if 'tag_0' in results and 'tag_1' in results:
        tag0_mean = results['tag_0']['mean']
        tag1_mean = results['tag_1']['mean']
        tag0_count = results['tag_0']['count']
        tag1_count = results['tag_1']['count']
        
        print(f"- Tag 0 samples: {tag0_count:,}, mean CoV: {tag0_mean:.6f}")
        print(f"- Tag 1 samples: {tag1_count:,}, mean CoV: {tag1_mean:.6f}")
        print(f"- CoV difference (Tag 1 - Tag 0): {tag1_mean - tag0_mean:.6f}")
        
        if tag1_mean > tag0_mean:
            print("- Tag 1 samples contribute more to entropy reduction (positive contribution)")
        else:
            print("- Tag 0 samples contribute more to entropy reduction")

if __name__ == "__main__":
    main()
