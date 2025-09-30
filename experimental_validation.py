"""
Enhanced Experimental Validation Framework

This module provides comprehensive experimental validation including:
- Rigorous statistical testing
- Error analysis and failure modes
- Performance analysis across different conditions
- Interpretability and visualization tools
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExperimentalResult:
    """Container for experimental results."""
    method_name: str
    dataset_name: str
    metrics: Dict[str, float]
    per_query_metrics: Dict[str, List[float]]
    runtime_ms: float
    memory_mb: float
    convergence_info: Dict[str, Any]
    failure_cases: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class StatisticalTest:
    """Statistical test results."""
    test_name: str
    p_value: float
    statistic: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class RigorousStatisticalTesting:
    """Comprehensive statistical testing framework."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def paired_t_test_with_effect_size(self, baseline: List[float], 
                                     treatment: List[float]) -> StatisticalTest:
        """Paired t-test with Cohen's d effect size."""
        baseline_arr = np.array(baseline)
        treatment_arr = np.array(treatment)
        
        if len(baseline_arr) != len(treatment_arr):
            raise ValueError("Arrays must have same length for paired test")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(treatment_arr, baseline_arr)
        
        # Cohen's d for paired samples
        diff = treatment_arr - baseline_arr
        effect_size = np.mean(diff) / np.std(diff, ddof=1)
        
        # Confidence interval for the difference
        diff_mean = np.mean(diff)
        diff_sem = stats.sem(diff)
        ci = stats.t.interval(1 - self.alpha, len(diff) - 1, diff_mean, diff_sem)
        
        # Interpretation
        if p_value < self.alpha:
            if effect_size > 0.8:
                interpretation = "Large significant improvement"
            elif effect_size > 0.5:
                interpretation = "Medium significant improvement"
            elif effect_size > 0.2:
                interpretation = "Small significant improvement"
            else:
                interpretation = "Significant but negligible improvement"
        else:
            interpretation = "No significant difference"
            
        return StatisticalTest(
            test_name="Paired t-test",
            p_value=p_value,
            statistic=t_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )
    
    def wilcoxon_signed_rank_test(self, baseline: List[float], 
                                treatment: List[float]) -> StatisticalTest:
        """Non-parametric Wilcoxon signed-rank test."""
        baseline_arr = np.array(baseline)
        treatment_arr = np.array(treatment)
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(treatment_arr, baseline_arr, 
                                          alternative='two-sided')
        
        # Effect size (r = Z / sqrt(N))
        n = len(baseline_arr)
        z_score = stats.norm.ppf(1 - p_value/2)
        effect_size = z_score / np.sqrt(n)
        
        interpretation = (
            "Significant improvement" if p_value < self.alpha 
            else "No significant difference"
        )
        
        return StatisticalTest(
            test_name="Wilcoxon signed-rank",
            p_value=p_value,
            statistic=statistic,
            effect_size=effect_size,
            confidence_interval=(np.nan, np.nan),
            interpretation=interpretation
        )
    
    def bootstrap_confidence_interval(self, data: List[float], 
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean."""
        data_arr = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)
        
        return (lower, upper)
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> List[float]:
        """Multiple comparison correction."""
        p_array = np.array(p_values)
        
        if method == 'bonferroni':
            return (p_array * len(p_values)).clip(0, 1).tolist()
        elif method == 'fdr_bh':  # Benjamini-Hochberg
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(p_array, method='fdr_bh')
            return corrected_p.tolist()
        else:
            raise ValueError(f"Unknown correction method: {method}")


class FailureModeAnalysis:
    """Comprehensive failure mode and error analysis."""
    
    def __init__(self):
        self.failure_threshold = 0.1  # Performance drop threshold
        
    def identify_failure_cases(self, baseline_scores: Dict[str, float],
                             method_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify queries where the method significantly underperforms."""
        failure_cases = []
        
        for query_id in baseline_scores:
            if query_id in method_scores:
                baseline_score = baseline_scores[query_id]
                method_score = method_scores[query_id]
                
                # Identify significant performance drops
                relative_drop = (baseline_score - method_score) / max(baseline_score, 1e-8)
                
                if relative_drop > self.failure_threshold:
                    failure_cases.append({
                        'query_id': query_id,
                        'baseline_score': baseline_score,
                        'method_score': method_score,
                        'relative_drop': relative_drop,
                        'absolute_drop': baseline_score - method_score,
                        'failure_type': self._classify_failure_type(relative_drop)
                    })
        
        return failure_cases
    
    def _classify_failure_type(self, relative_drop: float) -> str:
        """Classify the type of failure based on performance drop."""
        if relative_drop > 0.5:
            return "catastrophic"
        elif relative_drop > 0.3:
            return "severe"
        elif relative_drop > 0.1:
            return "moderate"
        else:
            return "minor"
    
    def analyze_failure_patterns(self, failure_cases: List[Dict[str, Any]],
                               query_metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """Analyze patterns in failure cases."""
        if not failure_cases:
            return {"message": "No failure cases detected"}
        
        analysis = {
            'total_failures': len(failure_cases),
            'failure_types': {},
            'severity_distribution': {},
            'average_drops': {}
        }
        
        # Count failure types
        for case in failure_cases:
            failure_type = case['failure_type']
            analysis['failure_types'][failure_type] = analysis['failure_types'].get(failure_type, 0) + 1
            
        # Severity analysis
        relative_drops = [case['relative_drop'] for case in failure_cases]
        analysis['severity_distribution'] = {
            'mean_relative_drop': np.mean(relative_drops),
            'median_relative_drop': np.median(relative_drops),
            'std_relative_drop': np.std(relative_drops),
            'max_relative_drop': np.max(relative_drops)
        }
        
        # If query metadata is available, analyze failure patterns
        if query_metadata:
            analysis['pattern_analysis'] = self._analyze_query_patterns(failure_cases, query_metadata)
        
        return analysis
    
    def _analyze_query_patterns(self, failure_cases: List[Dict[str, Any]],
                              query_metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze failure patterns based on query characteristics."""
        patterns = {
            'query_length_distribution': [],
            'query_complexity_distribution': [],
            'topic_distribution': {}
        }
        
        for case in failure_cases:
            query_id = case['query_id']
            if query_id in query_metadata:
                metadata = query_metadata[query_id]
                
                # Query length analysis
                if 'length' in metadata:
                    patterns['query_length_distribution'].append(metadata['length'])
                
                # Query complexity analysis
                if 'complexity' in metadata:
                    patterns['query_complexity_distribution'].append(metadata['complexity'])
                
                # Topic distribution
                if 'topic' in metadata:
                    topic = metadata['topic']
                    patterns['topic_distribution'][topic] = patterns['topic_distribution'].get(topic, 0) + 1
        
        return patterns


class PerformanceAnalyzer:
    """Comprehensive performance analysis across different conditions."""
    
    def __init__(self):
        pass
        
    def analyze_performance_vs_sequence_length(self, method_func, sequence_lengths: List[int],
                                             batch_size: int = 8, num_trials: int = 5) -> Dict[str, List[float]]:
        """Analyze performance scaling with sequence length."""
        results = {
            'sequence_lengths': sequence_lengths,
            'runtime_ms': [],
            'memory_mb': [],
            'accuracy': []
        }
        
        for seq_len in sequence_lengths:
            trial_runtimes = []
            trial_memories = []
            trial_accuracies = []
            
            for _ in range(num_trials):
                # Create synthetic data
                hidden_size = 768
                input_data = torch.randn(batch_size, seq_len, hidden_size)
                
                # Measure performance
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()
                
                import time
                start_time = time.time()
                
                # Run method
                with torch.no_grad():
                    output = method_func(input_data)
                
                end_time = time.time()
                runtime = (end_time - start_time) * 1000  # ms
                
                if torch.cuda.is_available():
                    memory_used = (torch.cuda.max_memory_allocated() - start_memory) / 1024**2  # MB
                else:
                    memory_used = 0
                
                # Synthetic accuracy (in real use, replace with actual metric)
                accuracy = 0.8 + 0.1 * np.random.randn()
                
                trial_runtimes.append(runtime)
                trial_memories.append(memory_used)
                trial_accuracies.append(accuracy)
            
            results['runtime_ms'].append(np.mean(trial_runtimes))
            results['memory_mb'].append(np.mean(trial_memories))
            results['accuracy'].append(np.mean(trial_accuracies))
        
        return results
    
    def analyze_robustness_to_noise(self, method_func, noise_levels: List[float],
                                   base_input: torch.Tensor, num_trials: int = 10) -> Dict[str, List[float]]:
        """Analyze method robustness to input noise."""
        results = {
            'noise_levels': noise_levels,
            'accuracy_drop': [],
            'output_variance': []
        }
        
        # Get baseline performance
        with torch.no_grad():
            baseline_output = method_func(base_input)
            baseline_accuracy = 0.85  # In real use, compute actual metric
        
        for noise_level in noise_levels:
            trial_accuracies = []
            trial_outputs = []
            
            for _ in range(num_trials):
                # Add noise to input
                noise = torch.randn_like(base_input) * noise_level
                noisy_input = base_input + noise
                
                with torch.no_grad():
                    output = method_func(noisy_input)
                
                # Compute metrics
                accuracy = baseline_accuracy * (1 - noise_level * 0.5 + 0.1 * np.random.randn())
                trial_accuracies.append(accuracy)
                trial_outputs.append(output)
            
            accuracy_drop = baseline_accuracy - np.mean(trial_accuracies)
            output_variance = np.var([torch.norm(out).item() for out in trial_outputs])
            
            results['accuracy_drop'].append(accuracy_drop)
            results['output_variance'].append(output_variance)
        
        return results


class ExperimentVisualizer:
    """Visualization tools for experimental results."""
    
    def __init__(self, output_dir: str = "experiment_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_method_comparison(self, results: Dict[str, ExperimentalResult], 
                             metric: str = "nDCG@10", save_name: str = "method_comparison.png"):
        """Plot comparison of different methods."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(results.keys())
        metric_values = [results[method].metrics.get(metric, 0) for method in methods]
        
        # Bar plot of metrics
        bars = ax1.bar(methods, metric_values)
        ax1.set_title(f'{metric} Comparison Across Methods')
        ax1.set_ylabel(metric)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add confidence intervals if available
        for i, method in enumerate(methods):
            if metric in results[method].confidence_intervals:
                ci = results[method].confidence_intervals[metric]
                ax1.errorbar(i, metric_values[i], yerr=[[metric_values[i] - ci[0]], [ci[1] - metric_values[i]]], 
                           fmt='none', capsize=5, color='black')
        
        # Runtime comparison
        runtimes = [results[method].runtime_ms for method in methods]
        ax2.bar(methods, runtimes, color='orange')
        ax2.set_title('Runtime Comparison (ms)')
        ax2.set_ylabel('Runtime (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_scaling(self, scaling_results: Dict[str, List[float]], 
                               save_name: str = "performance_scaling.png"):
        """Plot performance scaling with sequence length."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        seq_lengths = scaling_results['sequence_lengths']
        
        # Runtime scaling
        axes[0, 0].plot(seq_lengths, scaling_results['runtime_ms'], 'b-o')
        axes[0, 0].set_title('Runtime Scaling')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Runtime (ms)')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        
        # Memory scaling
        axes[0, 1].plot(seq_lengths, scaling_results['memory_mb'], 'r-s')
        axes[0, 1].set_title('Memory Scaling')
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        
        # Accuracy vs sequence length
        axes[1, 0].plot(seq_lengths, scaling_results['accuracy'], 'g-^')
        axes[1, 0].set_title('Accuracy vs Sequence Length')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Theoretical vs empirical complexity
        theoretical_complexity = [s**2 for s in seq_lengths]  # O(L²)
        theoretical_complexity = np.array(theoretical_complexity) / theoretical_complexity[0] * scaling_results['runtime_ms'][0]
        
        axes[1, 1].plot(seq_lengths, scaling_results['runtime_ms'], 'b-o', label='Empirical')
        axes[1, 1].plot(seq_lengths, theoretical_complexity, 'r--', label='Theoretical O(L²)')
        axes[1, 1].set_title('Complexity Analysis')
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('Runtime (ms)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_failure_analysis(self, failure_cases: List[Dict[str, Any]], 
                            save_name: str = "failure_analysis.png"):
        """Plot failure mode analysis."""
        if not failure_cases:
            print("No failure cases to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Failure type distribution
        failure_types = [case['failure_type'] for case in failure_cases]
        type_counts = pd.Series(failure_types).value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Failure Type Distribution')
        
        # Relative drop distribution
        relative_drops = [case['relative_drop'] for case in failure_cases]
        axes[0, 1].hist(relative_drops, bins=20, alpha=0.7, color='red')
        axes[0, 1].set_title('Relative Performance Drop Distribution')
        axes[0, 1].set_xlabel('Relative Drop')
        axes[0, 1].set_ylabel('Frequency')
        
        # Scatter plot: baseline vs method scores
        baseline_scores = [case['baseline_score'] for case in failure_cases]
        method_scores = [case['method_score'] for case in failure_cases]
        axes[1, 0].scatter(baseline_scores, method_scores, alpha=0.6)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
        axes[1, 0].set_title('Baseline vs Method Scores (Failure Cases)')
        axes[1, 0].set_xlabel('Baseline Score')
        axes[1, 0].set_ylabel('Method Score')
        axes[1, 0].legend()
        
        # Absolute drop vs baseline score
        absolute_drops = [case['absolute_drop'] for case in failure_cases]
        axes[1, 1].scatter(baseline_scores, absolute_drops, alpha=0.6, color='orange')
        axes[1, 1].set_title('Absolute Drop vs Baseline Score')
        axes[1, 1].set_xlabel('Baseline Score')
        axes[1, 1].set_ylabel('Absolute Drop')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_tests(self, test_results: List[StatisticalTest], 
                             save_name: str = "statistical_tests.png"):
        """Plot statistical test results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        test_names = [test.test_name for test in test_results]
        p_values = [test.p_value for test in test_results]
        effect_sizes = [test.effect_size for test in test_results]
        
        # P-values
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        bars1 = ax1.bar(test_names, p_values, color=colors)
        ax1.axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
        ax1.set_title('Statistical Test P-values')
        ax1.set_ylabel('P-value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # Effect sizes
        bars2 = ax2.bar(test_names, effect_sizes)
        ax2.set_title('Effect Sizes')
        ax2.set_ylabel('Effect Size (Cohen\'s d)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add horizontal lines for effect size interpretation
        ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Small effect')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
        ax2.axhline(y=0.8, color='gray', linestyle='-', alpha=0.7, label='Large effect')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()


class ExperimentalValidationFramework:
    """Main framework orchestrating all experimental validation."""
    
    def __init__(self, output_dir: str = "experimental_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.statistical_tester = RigorousStatisticalTesting()
        self.failure_analyzer = FailureModeAnalysis()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = ExperimentVisualizer(str(self.output_dir / "plots"))
        
    def run_comprehensive_validation(self, methods: Dict[str, callable],
                                   datasets: Dict[str, Any],
                                   num_trials: int = 5) -> Dict[str, Any]:
        """Run comprehensive experimental validation."""
        
        print("Starting Comprehensive Experimental Validation...")
        
        all_results = {}
        statistical_tests = []
        failure_analyses = {}
        
        # 1. Run experiments for each method and dataset
        for dataset_name, dataset in datasets.items():
            print(f"\nTesting on dataset: {dataset_name}")
            
            dataset_results = {}
            for method_name, method_func in methods.items():
                print(f"  Running method: {method_name}")
                
                trial_results = []
                for trial in range(num_trials):
                    # Run single trial
                    result = self._run_single_trial(method_func, dataset, trial)
                    trial_results.append(result)
                
                # Aggregate trial results
                aggregated = self._aggregate_trial_results(trial_results, method_name, dataset_name)
                dataset_results[method_name] = aggregated
            
            all_results[dataset_name] = dataset_results
            
            # 2. Statistical testing
            if len(dataset_results) >= 2:
                baseline_name = list(dataset_results.keys())[0]  # Use first as baseline
                baseline_scores = dataset_results[baseline_name].per_query_metrics.get('primary_metric', [])
                
                for method_name, result in dataset_results.items():
                    if method_name != baseline_name:
                        method_scores = result.per_query_metrics.get('primary_metric', [])
                        
                        if len(baseline_scores) == len(method_scores):
                            # Paired t-test
                            t_test = self.statistical_tester.paired_t_test_with_effect_size(
                                baseline_scores, method_scores
                            )
                            t_test.test_name = f"{method_name} vs {baseline_name} (t-test)"
                            statistical_tests.append(t_test)
                            
                            # Wilcoxon test
                            w_test = self.statistical_tester.wilcoxon_signed_rank_test(
                                baseline_scores, method_scores
                            )
                            w_test.test_name = f"{method_name} vs {baseline_name} (Wilcoxon)"
                            statistical_tests.append(w_test)
            
            # 3. Failure analysis
            if len(dataset_results) >= 2:
                baseline_name = list(dataset_results.keys())[0]
                baseline_per_query = dict(zip(
                    range(len(dataset_results[baseline_name].per_query_metrics.get('primary_metric', []))),
                    dataset_results[baseline_name].per_query_metrics.get('primary_metric', [])
                ))
                
                for method_name, result in dataset_results.items():
                    if method_name != baseline_name:
                        method_per_query = dict(zip(
                            range(len(result.per_query_metrics.get('primary_metric', []))),
                            result.per_query_metrics.get('primary_metric', [])
                        ))
                        
                        failure_cases = self.failure_analyzer.identify_failure_cases(
                            baseline_per_query, method_per_query
                        )
                        failure_analysis = self.failure_analyzer.analyze_failure_patterns(failure_cases)
                        failure_analyses[f"{dataset_name}_{method_name}"] = {
                            'failure_cases': failure_cases,
                            'analysis': failure_analysis
                        }
        
        # 4. Generate visualizations
        print("\nGenerating visualizations...")
        
        # Plot method comparisons for each dataset
        for dataset_name, dataset_results in all_results.items():
            self.visualizer.plot_method_comparison(
                dataset_results, save_name=f"method_comparison_{dataset_name}.png"
            )
        
        # Plot statistical test results
        if statistical_tests:
            self.visualizer.plot_statistical_tests(
                statistical_tests, save_name="statistical_tests.png"
            )
        
        # Plot failure analyses
        for analysis_name, analysis_data in failure_analyses.items():
            if analysis_data['failure_cases']:
                self.visualizer.plot_failure_analysis(
                    analysis_data['failure_cases'], 
                    save_name=f"failure_analysis_{analysis_name}.png"
                )
        
        # 5. Save comprehensive report
        report = {
            'experimental_results': self._serialize_results(all_results),
            'statistical_tests': [self._serialize_statistical_test(test) for test in statistical_tests],
            'failure_analyses': failure_analyses,
            'summary': self._generate_summary(all_results, statistical_tests, failure_analyses)
        }
        
        with open(self.output_dir / "comprehensive_validation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nComprehensive validation completed. Results saved to {self.output_dir}")
        
        return report
    
    def _run_single_trial(self, method_func: callable, dataset: Any, trial: int) -> Dict[str, Any]:
        """Run a single experimental trial."""
        # This is a simplified version - in practice, integrate with your evaluation pipeline
        
        # Synthetic trial result
        primary_metric_score = 0.75 + 0.1 * np.random.randn() + trial * 0.01
        secondary_scores = [0.7 + 0.05 * np.random.randn() for _ in range(10)]
        
        return {
            'primary_metric': primary_metric_score,
            'secondary_scores': secondary_scores,
            'runtime_ms': 100 + 20 * np.random.randn(),
            'memory_mb': 500 + 50 * np.random.randn()
        }
    
    def _aggregate_trial_results(self, trial_results: List[Dict], method_name: str, 
                               dataset_name: str) -> ExperimentalResult:
        """Aggregate results from multiple trials."""
        
        primary_scores = [trial['primary_metric'] for trial in trial_results]
        all_secondary_scores = []
        for trial in trial_results:
            all_secondary_scores.extend(trial['secondary_scores'])
        
        runtimes = [trial['runtime_ms'] for trial in trial_results]
        memories = [trial['memory_mb'] for trial in trial_results]
        
        # Compute confidence intervals
        primary_ci = self.statistical_tester.bootstrap_confidence_interval(primary_scores)
        
        return ExperimentalResult(
            method_name=method_name,
            dataset_name=dataset_name,
            metrics={
                'primary_metric': np.mean(primary_scores),
                'primary_metric_std': np.std(primary_scores)
            },
            per_query_metrics={
                'primary_metric': primary_scores
            },
            runtime_ms=np.mean(runtimes),
            memory_mb=np.mean(memories),
            convergence_info={},
            failure_cases=[],
            confidence_intervals={
                'primary_metric': primary_ci
            }
        )
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Serialize results for JSON output."""
        serialized = {}
        for dataset_name, dataset_results in results.items():
            serialized[dataset_name] = {}
            for method_name, result in dataset_results.items():
                serialized[dataset_name][method_name] = {
                    'metrics': result.metrics,
                    'runtime_ms': result.runtime_ms,
                    'memory_mb': result.memory_mb,
                    'confidence_intervals': result.confidence_intervals
                }
        return serialized
    
    def _serialize_statistical_test(self, test: StatisticalTest) -> Dict:
        """Serialize statistical test for JSON output."""
        return {
            'test_name': test.test_name,
            'p_value': test.p_value,
            'statistic': test.statistic,
            'effect_size': test.effect_size,
            'confidence_interval': test.confidence_interval,
            'interpretation': test.interpretation
        }
    
    def _generate_summary(self, results: Dict, statistical_tests: List[StatisticalTest],
                        failure_analyses: Dict) -> Dict[str, Any]:
        """Generate summary of experimental validation."""
        
        # Count significant improvements
        significant_tests = [test for test in statistical_tests if test.p_value < 0.05]
        
        # Count total failure cases
        total_failures = sum(
            len(analysis['failure_cases']) 
            for analysis in failure_analyses.values()
        )
        
        return {
            'total_experiments': len(results),
            'total_statistical_tests': len(statistical_tests),
            'significant_improvements': len(significant_tests),
            'total_failure_cases': total_failures,
            'recommendations': self._generate_recommendations(results, statistical_tests, failure_analyses)
        }
    
    def _generate_recommendations(self, results: Dict, statistical_tests: List[StatisticalTest],
                                failure_analyses: Dict) -> List[str]:
        """Generate recommendations based on experimental results."""
        recommendations = []
        
        # Check for consistent improvements
        significant_tests = [test for test in statistical_tests if test.p_value < 0.05]
        if len(significant_tests) >= len(statistical_tests) * 0.5:
            recommendations.append("Method shows consistent significant improvements across datasets")
        else:
            recommendations.append("Method improvements are inconsistent across datasets")
        
        # Check for failure modes
        total_failures = sum(len(analysis['failure_cases']) for analysis in failure_analyses.values())
        if total_failures > 0:
            recommendations.append(f"Found {total_failures} failure cases requiring investigation")
        
        # Effect size recommendations
        large_effects = [test for test in significant_tests if test.effect_size > 0.8]
        if large_effects:
            recommendations.append("Method shows large effect sizes, indicating practical significance")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the framework
    print("Setting up Experimental Validation Framework...")
    
    framework = ExperimentalValidationFramework()
    
    # Example methods (in practice, replace with your actual methods)
    def standard_attention(x):
        return torch.randn_like(x)
    
    def maw_attention(x):
        return torch.randn_like(x) + 0.1  # Slight improvement
    
    methods = {
        'standard': standard_attention,
        'maw': maw_attention
    }
    
    # Example datasets (in practice, replace with your actual datasets)
    datasets = {
        'dataset1': {'size': 1000},
        'dataset2': {'size': 500}
    }
    
    # Run validation
    validation_report = framework.run_comprehensive_validation(methods, datasets, num_trials=3)
    
    print("Experimental validation completed!")
    print(f"Summary: {validation_report['summary']}")