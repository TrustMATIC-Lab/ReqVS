"""
Analyze performance of all models + consensus methods + ensemble
Adapted for complete_pipeline.py output, reads parameters from pipeline_config.yaml
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import yaml
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams['axes.unicode_minus'] = False


def load_config(config_path='pipeline_config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_ef_data(file_path):
    """Load EF data, exclude average row"""
    df = pd.read_csv(file_path)
    df = df[df['target'] != 'average EF:'].copy()
    return df


def build_methods_dict(config):
    """
    Build methods dictionary from configuration file
    Returns mapping from method names to file paths
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_config = config['output']
    output_suffix = output_config['suffix']
    output_folder_name = output_config['folder_name']
    file_names = output_config.get('file_names', {})
    
    folder_path = output_folder_name + output_suffix
    folder_path = os.path.join(base_dir, folder_path)
    
    methods = {}
    
    # Get model list
    cls_models = config['models']['classifier_models']
    reg_models = config['models']['regressor_models']
    
    # Get EF file prefix
    ef_prefix = file_names.get('ef_prefix', 'enrichment_factor')
    consensus_score_name = file_names.get('consensus_score', 'consensus_score')
    
    # Single models (classifier group) - use display_name from config
    for model_info in cls_models:
        model_name = model_info['name']
        display_name = model_info.get('display_name', model_name.capitalize())
        file_path = os.path.join(folder_path, f'{ef_prefix}_{model_name}_all{output_suffix}.csv')
        methods[display_name] = file_path
    
    # Single models (regressor group) - use display_name from config
    for model_info in reg_models:
        model_name = model_info['name']
        display_name = model_info.get('display_name', model_name.capitalize())
        file_path = os.path.join(folder_path, f'{ef_prefix}_{model_name}_all{output_suffix}.csv')
        methods[display_name] = file_path
    
    # Classifier group consensus
    methods['DTI SE Consensus'] = os.path.join(folder_path, f'{ef_prefix}_cls_SEconsensus{output_suffix}.csv')
    
    # Regressor group consensus
    methods['DTA SE Consensus'] = os.path.join(folder_path, f'{ef_prefix}_reg_SEconsensus{output_suffix}.csv')
    
    # Overall consensus score
    methods['Overall SE Consensus'] = os.path.join(folder_path, f'{consensus_score_name}_SE_ef{output_suffix}.csv')
    
    # Ensemble (classifier group, regressor group, overall)
    methods['DTI Ensemble'] = os.path.join(folder_path, f'{ef_prefix}_ensemble_cls{output_suffix}.csv')
    methods['DTA Ensemble'] = os.path.join(folder_path, f'{ef_prefix}_ensemble_reg{output_suffix}.csv')
    methods['Overall Ensemble'] = os.path.join(folder_path, f'{ef_prefix}_ensemble{output_suffix}.csv')
    
    return methods, folder_path


def analyze_performance(config):
    """
    Analyze performance of all methods
    """
    print("="*80)
    print("Analyzing Performance")
    print("="*80)
    
    output_config = config['output']
    output_suffix = output_config['suffix']
    file_names = output_config.get('file_names', {})
    
    # Build methods dictionary
    methods, output_folder = build_methods_dict(config)
    
    print(f"Output folder: {output_folder}")
    print(f"Number of methods: {len(methods)}")
    
    # Load all data
    all_data = {}
    for method_name, file_path in methods.items():
        try:
            if os.path.exists(file_path):
                df = load_ef_data(file_path)
                all_data[method_name] = df
                print(f"Loaded {method_name}: {len(df)} targets")
            else:
                print(f"Warning: File not found {file_path}, skipping {method_name}")
        except Exception as e:
            print(f"Error loading {method_name} from {file_path}: {e}")
            continue
    
    # Define all metrics
    metrics = ['EF0.1%', 'EF0.5%', 'EF1%', 'EF5%', 'AUROC', 'AUPRC', 'BEDROC']
    
    # Calculate performance statistics
    results = []
    
    for method_name, df in all_data.items():
        # Overall data
        n_total = len(df)
        
        # Initialize result dictionary
        result_dict = {
            'Method': method_name,
            'Total_N': n_total
        }
        
        # Calculate mean for each metric
        for metric in metrics:
            if metric in df.columns:
                # Overall
                if n_total > 0:
                    total_mean = df[metric].mean()
                    total_median = df[metric].median()
                else:
                    total_mean = total_median = 0.0
                
                result_dict[f'Total_{metric}_mean'] = total_mean
                result_dict[f'Total_{metric}_median'] = total_median
            else:
                # If metric doesn't exist, set to 0
                result_dict[f'Total_{metric}_mean'] = 0.0
                result_dict[f'Total_{metric}_median'] = 0.0
        
        results.append(result_dict)
        
        # Print summary information
        print(f"\n{method_name}:")
        print(f"  Total targets: {n_total}")
        for metric in metrics[:4]:  # Only print first 4 EF metrics
            if f'Total_{metric}_mean' in result_dict:
                print(f"  {metric}={result_dict[f'Total_{metric}_mean']:.3f} ", end="")
        print(f"AUROC={result_dict.get('Total_AUROC_mean', 0):.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    performance_summary_name = file_names.get('performance_summary', 'performance_summary')
    results_file = os.path.join(output_folder, f'{performance_summary_name}{output_suffix}.csv')
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {performance_summary_name}{output_suffix}.csv")
    
    # Plot comparison
    plot_performance_boxplot(results_df, all_data, output_folder, output_suffix, file_names, config)
    
    return results_df, all_data


def get_sorted_methods_for_metric(results_df, metric, config):
    """
    Sort methods according to specified metric
    Sorting rules:
    1. Overall consensus at the top
    2. Regressor group: consensus and ensemble first, then single models sorted by performance
    3. Classifier group: consensus and ensemble first, then single models sorted by performance
    
    Args:
        results_df: DataFrame containing results for all methods
        metric: Metric name (e.g., 'EF0.1%')
        config: Configuration dictionary
    
    Returns:
        Sorted list of methods
    """
    # Define method categories
    overall_consensus = ['Overall SE Consensus']
    ensemble_average = ['Overall Ensemble']
    
    reg_consensus_ensemble = ['DTA SE Consensus', 'DTA Ensemble']
    reg_single = []
    
    cls_consensus_ensemble = ['DTI SE Consensus', 'DTI Ensemble']
    cls_single = []
    
    # Build single model list from config - use display_name from config
    cls_models = config['models']['classifier_models']
    reg_models = config['models']['regressor_models']
    
    for model_info in cls_models:
        display_name = model_info.get('display_name', model_info['name'].capitalize())
        cls_single.append(display_name)
    
    for model_info in reg_models:
        display_name = model_info.get('display_name', model_info['name'].capitalize())
        reg_single.append(display_name)
    
    # Get sort column name
    sort_col = f'Total_{metric}_mean'
    
    # 1. Overall consensus (maintain fixed order)
    sorted_methods = []
    for method in overall_consensus:
        if method in results_df['Method'].values:
            sorted_methods.append(method)
    
    # 2. Overall Ensemble
    for method in ensemble_average:
        if method in results_df['Method'].values:
            sorted_methods.append(method)
    
    # 3. Classifier group: consensus and ensemble first
    for method in cls_consensus_ensemble:
        if method in results_df['Method'].values:
            sorted_methods.append(method)
    
    # 4. Classifier single models: sort by performance
    cls_single_data = []
    for method in cls_single:
        if method in results_df['Method'].values:
            method_row = results_df[results_df['Method'] == method]
            if len(method_row) > 0 and sort_col in method_row.columns:
                score = method_row[sort_col].values[0]
                cls_single_data.append((method, score))
    
    # Sort by score from high to low
    cls_single_data.sort(key=lambda x: x[1], reverse=True)
    sorted_methods.extend([m[0] for m in cls_single_data])
    
    # 5. Regressor group: consensus and ensemble first
    for method in reg_consensus_ensemble:
        if method in results_df['Method'].values:
            sorted_methods.append(method)
    
    # 6. Regressor single models: sort by performance
    reg_single_data = []
    for method in reg_single:
        if method in results_df['Method'].values:
            method_row = results_df[results_df['Method'] == method]
            if len(method_row) > 0 and sort_col in method_row.columns:
                score = method_row[sort_col].values[0]
                reg_single_data.append((method, score))
    
    # Sort by score from high to low
    reg_single_data.sort(key=lambda x: x[1], reverse=True)
    sorted_methods.extend([m[0] for m in reg_single_data])
    
    return sorted_methods


def get_method_name_mapping(config=None):
    """
    Define method name mapping dictionary
    Can customize display name for each method here
    Use LaTeX format to display superscripts, e.g., r'CS$^2$' displays as CS²
    If method name is not in mapping, will use original name or display_name from config
    """
    mapping = {
        # Classifier group consensus
        'DTI SE Consensus': 'ReqVS-EPI',
        # Regressor group consensus
        'DTA SE Consensus': 'ReqVS-EPA',
        # Overall consensus score
        'Overall SE Consensus': 'ReqVS',
        # Ensemble
        'DTI Ensemble': 'ReqVS-ESI',
        'DTA Ensemble': 'ReqVS-ESA',
        'Overall Ensemble': 'ReqVS-ES',
    }
    
    # If config is provided, use display_name from config
    if config:
        cls_models = config['models']['classifier_models']
        reg_models = config['models']['regressor_models']
        
        for model_info in cls_models:
            display_name = model_info.get('display_name', model_info['name'].capitalize())
            mapping[display_name] = display_name  # Default to display_name, can be customized later
        
        for model_info in reg_models:
            display_name = model_info.get('display_name', model_info['name'].capitalize())
            mapping[display_name] = display_name  # Default to display_name, can be customized later
    
    return mapping


def get_method_label_styles():
    """
    Define style for each method label (color, bold, etc.)
    """
    return {
        # Overall consensus score - red bold
        'Overall SE Consensus': {
            'color': 'red',
            'weight': 'bold'
        },
        # Classifier group consensus - black bold
        'DTI SE Consensus': {
            'color': 'black',
            'weight': 'bold'
        },
        # Regressor group consensus - black bold
        'DTA SE Consensus': {
            'color': 'black',
            'weight': 'bold'
        },
        # Ensemble methods
        'Overall Ensemble': {
            'color': 'black',
            'weight': 'bold'
        },
        'DTI Ensemble': {
            'color': 'black',
            'weight': 'bold'
        },
        'DTA Ensemble': {
            'color': 'black',
            'weight': 'bold'
        },
        # Single models - default style
        'Drugban': {
            'color': 'black',
            'weight': 'normal'
        },
        'MCANet': {
            'color': 'black',
            'weight': 'normal'
        },
        'Hitscreen': {
            'color': None,
            'weight': 'normal'
        },
        'ColdstartCPI': {
            'color': None,
            'weight': 'normal'
        },
        'GraphDTA': {
            'color': None,
            'weight': 'normal'
        },
        'MGraphDTA': {
            'color': None,
            'weight': 'normal'
        },
        'PSICHIC': {
            'color': None,
            'weight': 'normal'
        },
        'DeepDTAGEN': {
            'color': None,
            'weight': 'normal'
        },
    }


def get_plot_config(config):
    """
    Get plotting parameters from configuration file
    """
    plotting_config = config.get('plotting', {})
    
    # Metric configuration
    metrics_to_plot = plotting_config.get('metrics_to_plot', ['EF0.1%', 'EF0.5%', 'EF1%', 'BEDROC'])
    metric_labels = plotting_config.get('metric_labels', {
        'EF0.1%': 'EF 0.1%',
        'EF0.5%': 'EF 0.5%',
        'EF1%': 'EF 1%',
        'BEDROC': 'BEDROC'
    })
    
    # Layout configuration
    n_rows = plotting_config.get('n_rows', 2)
    n_cols = plotting_config.get('n_cols', 2)
    
    # Figure size configuration
    fig_width = plotting_config.get('fig_width', 11)
    fig_height_base = plotting_config.get('fig_height_base', 14)
    dpi = plotting_config.get('dpi', 600)
    
    # Font size configuration
    fontsize_method_label = plotting_config.get('fontsize_method_label', 12)
    fontsize_xlabel = plotting_config.get('fontsize_xlabel', 10)
    fontsize_ylabel = plotting_config.get('fontsize_ylabel', 10)
    
    # Color configuration
    color_vmin = 0.0
    color_vmax = 1.0
    color_palette = plotting_config.get('color_palette', ['#d43325','#e45238','#ef764f','#fbb475','#fff2ad','#f4fad4','#4476b3','#7dacd1','#b8d7e9'])
    consensus_label_color = plotting_config.get('consensus_label_color', 'red')
    consensus_label_weight = plotting_config.get('consensus_label_weight', 'bold')
    
    return {
        'metrics_to_plot': metrics_to_plot,
        'metric_labels': metric_labels,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'fig_width': fig_width,
        'fig_height_base': fig_height_base,
        'dpi': dpi,
        'fontsize_method_label': fontsize_method_label,
        'fontsize_xlabel': fontsize_xlabel,
        'fontsize_ylabel': fontsize_ylabel,
        'color_vmin': color_vmin,
        'color_vmax': color_vmax,
        'color_palette': color_palette,
        'consensus_label_color': consensus_label_color,
        'consensus_label_weight': consensus_label_weight
    }


def plot_performance_boxplot(results_df, all_data, output_folder, output_suffix, file_names, config):
    """
    Plot boxplot of distribution for each method across all targets
    Use configuration function to set metrics, layout, and other parameters
    """
    print("\nGenerating Boxplot...")
    
    # Get plotting configuration from config file
    plot_config = get_plot_config(config)
    method_name_mapping = get_method_name_mapping(config)
    method_label_styles = get_method_label_styles()
    
    metrics = plot_config['metrics_to_plot']
    metric_labels = plot_config['metric_labels']
    
    # Validate layout configuration
    total_subplots = plot_config['n_rows'] * plot_config['n_cols']
    if total_subplots < len(metrics):
        print(f"Warning: Layout {plot_config['n_rows']}×{plot_config['n_cols']} = {total_subplots} subplots, but need to display {len(metrics)} metrics")
        print(f"Will only display first {total_subplots} metrics")
        metrics = metrics[:total_subplots]
    
    # Determine maximum number of methods across all metrics
    max_methods = 0
    for metric in metrics:
        methods = get_sorted_methods_for_metric(results_df, metric, config)
        n_valid_methods = len([m for m in methods if m in all_data])
        max_methods = max(max_methods, n_valid_methods)
    
    # Create figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['mathtext.default'] = 'regular'
    
    fig_height = max(plot_config['fig_height_base'], max_methods * 0.5)
    fig, axes = plt.subplots(plot_config['n_rows'], plot_config['n_cols'], 
                            figsize=(plot_config['fig_width'], fig_height), 
                            dpi=plot_config['dpi'])
    axes = axes.flatten()
    
    # Prepare data and plot for each metric
    for idx, metric in enumerate(metrics):
        # Get sorted method list for current metric
        methods = get_sorted_methods_for_metric(results_df, metric, config)
        
        total_data_list = []
        valid_labels = []
        display_to_original = {}
        
        for method_name in methods:
            if method_name in all_data:
                df = all_data[method_name]
                
                if metric in df.columns:
                    total_values = df[metric].dropna().values
                    
                    if len(total_values) > 0:
                        total_data_list.append(total_values)
                        # Use mapped display name
                        display_name = method_name_mapping.get(method_name, method_name)
                        if display_name not in valid_labels:
                            valid_labels.append(display_name)
                            display_to_original[display_name] = method_name
        
        # Get current subplot
        ax = axes[idx]

        if len(total_data_list) > 0:
            # Generate gradient colors
            n_boxes = len(total_data_list)
            vmin = plot_config['color_vmin']
            vmax = plot_config['color_vmax']
            my_colors = plot_config['color_palette']
            
            cmap = LinearSegmentedColormap.from_list('my_cmap', my_colors)
            colors = [cmap(vmax - (vmax - vmin) * i / max(n_boxes - 1, 1)) for i in range(n_boxes,0,-1)]
            spacing = 0.7
            positions = np.arange(n_boxes) * spacing
            
            bp = ax.boxplot(total_data_list,
                           vert=False,
                           patch_artist=True,
                           widths=0.4,
                           positions=positions,
                           showmeans=True,
                           medianprops=dict(color='black', linewidth=1),
                           meanprops=dict(marker='D', markersize=4, markerfacecolor='pink', 
                                         markeredgecolor='pink', markeredgewidth=1.5),
                           boxprops=dict(linewidth=1, edgecolor='black'),
                           whiskerprops=dict(linewidth=1.5, color='black'),
                           capprops=dict(linewidth=1, color='black'),
                           flierprops=dict(marker='o', markersize=6, alpha=0.5, 
                                          markeredgecolor='black', markerfacecolor='gray'))
            ax.invert_yaxis()
            
            # Set gradient colors for each box
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(1.0)
        
        # Set axis labels
        ax.set_xlabel(metric_labels[metric], fontsize=plot_config['fontsize_xlabel'], fontweight='bold')
        if idx % plot_config['n_cols'] == 0:
            ax.set_ylabel('Method', fontsize=plot_config['fontsize_ylabel'], fontweight='normal', labelpad=5)
        else:
            ax.set_ylabel('', fontsize=plot_config['fontsize_ylabel'], fontweight='normal', labelpad=5)

        if len(valid_labels) > 0:
            # Set y-axis labels and styles
            ax.set_yticks(positions)
            ax.set_yticklabels(valid_labels, fontsize=plot_config['fontsize_method_label'], fontweight='normal')
            for label in ax.get_yticklabels():
                text = label.get_text()
                original_method_name = display_to_original.get(text)
                
                if original_method_name and original_method_name in method_label_styles:
                    style = method_label_styles[original_method_name]
                    if style['color'] is not None:
                        label.set_color(style['color'])
                    if style['weight'] is not None:
                        label.set_fontweight(style['weight'])
                elif "Consensus" in text:
                    for orig_name, display_name in method_name_mapping.items():
                        if display_name == text and "Consensus" in orig_name:
                            label.set_color(plot_config['consensus_label_color'])
                            label.set_fontweight(plot_config['consensus_label_weight'])
                            break

        ax.grid(True, alpha=0.8, linestyle='--', axis='x', linewidth=1)
        ax.set_axisbelow(True)
    
    # Hide extra subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    performance_plot_name = file_names.get('performance_plot', 'performance_boxplot')
    boxplot_path = os.path.join(output_folder, f'{performance_plot_name}{output_suffix}.png')
    plt.savefig(boxplot_path, dpi=plot_config['dpi'], bbox_inches='tight')
    print(f"Boxplot saved to: {performance_plot_name}{output_suffix}.png")
    print(f"  Layout: {plot_config['n_rows']}×{plot_config['n_cols']}, Metrics: {', '.join(metrics)}")
    plt.close()


def main():
    """Main function"""
    print("="*80)
    print("Performance Analysis Script")
    print("="*80)
    
    # Load configuration
    config_path = 'pipeline_config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file not found {config_path}")
        return
    
    config = load_config(config_path)
    print(f"\nConfig file: {config_path}")
    output_config = config['output']
    print(f"Output suffix: {output_config['suffix']}")
    print(f"Output folder: {output_config['folder_name']}")
    
    # Analyze performance
    results_df, all_data = analyze_performance(config)
    
    print("\n" + "="*80)
    print("Analysis Completed!")
    print("="*80)
    print("\nOutput files:")
    file_names = output_config.get('file_names', {})
    performance_summary_name = file_names.get('performance_summary', 'performance_summary')
    performance_plot_name = file_names.get('performance_plot', 'performance_boxplot')
    print(f"  1. {performance_summary_name}{output_config['suffix']}.csv")
    print(f"  2. {performance_plot_name}{output_config['suffix']}.png")


if __name__ == '__main__':
    main()
