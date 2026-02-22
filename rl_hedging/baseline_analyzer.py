#!/usr/bin/env python3
"""
RL Hedging: Baseline Performance Analyzer

Analyzes classical hedger results and provides:
1. Statistical summary of hedging errors
2. Distribution analysis (percentiles, quartiles)
3. Visualization-ready metrics
4. Comparison framework for RL agent
"""

import numpy as np
import pandas as pd


class BaselineAnalyzer:
    """
    Analyze classical hedger performance across all paths.
    """
    
    def __init__(self, final_errors_df):
        """
        Initialize analyzer with results from classical hedger.
        
        Parameters
        ----------
        final_errors_df : pd.DataFrame
            Output from ClassicalDeltaHedger.calculate_final_hedging_error()
        """
        self.df = final_errors_df
        self.summary = {}
    
    def analyze(self):
        """Run all analyses and return summary."""
        self.summary = {
            'payoff_stats': self._payoff_stats(),
            'hedging_error_stats': self._hedging_error_stats(),
            'transaction_cost_stats': self._transaction_cost_stats(),
            'pnl_stats': self._pnl_stats(),
            'efficiency_metrics': self._efficiency_metrics(),
            'distribution': self._distribution_analysis()
        }
        return self.summary
    
    def _payoff_stats(self):
        """Option payoff statistics."""
        return {
            'min': self.df['option_payoff'].min(),
            'max': self.df['option_payoff'].max(),
            'mean': self.df['option_payoff'].mean(),
            'std': self.df['option_payoff'].std(),
            'median': self.df['option_payoff'].median()
        }
    
    def _hedging_error_stats(self):
        """
        Hedging error statistics using MULTIPLE metrics to avoid OTM distortion.
        
        Metrics:
        1. abs_hedging_error: Dollar error (works for all paths)
        2. abs_hedging_error_pct: Absolute error as % of initial option value (all paths)
        3. rel_hedging_error_pct: Relative error (ITM only, meaningful)
        4. return_on_notional: Error as % of initial option value (all paths, RECOMMENDED)
        """
        abs_errors = self.df['abs_hedging_error']
        abs_errors_pct = self.df['abs_hedging_error_pct']
        
        # Relative error (ITM paths only - where it's meaningful)
        itm_mask = self.df['is_itm']
        rel_errors_itm = self.df[itm_mask]['rel_hedging_error_pct']
        
        # Return on notional (all paths - always meaningful)
        return_on_notional = self.df['return_on_notional']
        
        return {
            # Absolute error in dollars
            'abs_mean': abs_errors.mean(),
            'abs_std': abs_errors.std(),
            'abs_min': abs_errors.min(),
            'abs_max': abs_errors.max(),
            'abs_median': abs_errors.median(),
            
            # Absolute error as percentage
            'abs_pct_mean': abs_errors_pct.mean(),
            'abs_pct_std': abs_errors_pct.std(),
            'abs_pct_min': abs_errors_pct.min(),
            'abs_pct_max': abs_errors_pct.max(),
            'abs_pct_median': abs_errors_pct.median(),
            
            # Relative error (ITM only)
            'rel_mean_pct_itm': rel_errors_itm.mean() if len(rel_errors_itm) > 0 else None,
            'rel_std_pct_itm': rel_errors_itm.std() if len(rel_errors_itm) > 0 else None,
            'rel_count_itm': len(rel_errors_itm),
            'itm_paths_pct': (itm_mask.sum() / len(self.df) * 100),
            
            # Return on notional (all paths - RECOMMENDED for comparison)
            'ron_mean': return_on_notional.mean(),
            'ron_std': return_on_notional.std(),
            'ron_min': return_on_notional.min(),
            'ron_max': return_on_notional.max(),
            'ron_median': return_on_notional.median()
        }
    
    def _transaction_cost_stats(self):
        """Transaction cost statistics."""
        costs = self.df['total_transaction_costs']
        payoffs = self.df['option_payoff']
        cost_pct = (costs / payoffs * 100)
        
        return {
            'total': costs.sum(),
            'mean': costs.mean(),
            'std': costs.std(),
            'min': costs.min(),
            'max': costs.max(),
            'as_pct_payoff_mean': cost_pct.mean(),
            'as_pct_payoff_total': (costs.sum() / payoffs.sum() * 100)
        }
    
    def _pnl_stats(self):
        """Hedger P&L statistics."""
        pnl = self.df['net_pnl']
        
        return {
            'total': pnl.sum(),
            'mean': pnl.mean(),
            'std': pnl.std(),
            'min': pnl.min(),
            'max': pnl.max(),
            'positive_count': (pnl > 0).sum(),
            'negative_count': (pnl < 0).sum(),
            'win_rate_pct': (pnl > 0).sum() / len(pnl) * 100
        }
    
    def _efficiency_metrics(self):
        """Efficiency metrics for comparison."""
        payoff = self.df['option_payoff'].sum()
        total_cost = (self.df['abs_hedging_error'] + self.df['total_transaction_costs']).sum()
        
        return {
            'total_payoff': payoff,
            'total_cost': total_cost,
            'cost_as_pct_payoff': total_cost / payoff * 100,
            'avg_cost_per_option': total_cost / len(self.df),
            'hedging_error_pct': (self.df['abs_hedging_error'].sum() / payoff * 100),
            'transaction_cost_pct': (self.df['total_transaction_costs'].sum() / payoff * 100)
        }
    
    def _distribution_analysis(self):
        """Percentile analysis for distribution understanding."""
        abs_errors = self.df['abs_hedging_error']
        abs_errors_pct = self.df['abs_hedging_error_pct']
        rel_errors = self.df['rel_hedging_error_pct']
        ron = self.df['return_on_notional']
        
        return {
            'abs_error_p10': abs_errors.quantile(0.1),
            'abs_error_p25': abs_errors.quantile(0.25),
            'abs_error_p50': abs_errors.quantile(0.50),
            'abs_error_p75': abs_errors.quantile(0.75),
            'abs_error_p90': abs_errors.quantile(0.90),
            'abs_error_pct_p10': abs_errors_pct.quantile(0.1),
            'abs_error_pct_p25': abs_errors_pct.quantile(0.25),
            'abs_error_pct_p50': abs_errors_pct.quantile(0.50),
            'abs_error_pct_p75': abs_errors_pct.quantile(0.75),
            'abs_error_pct_p90': abs_errors_pct.quantile(0.90),
            'rel_error_p10': rel_errors.quantile(0.1),
            'rel_error_p25': rel_errors.quantile(0.25),
            'rel_error_p50': rel_errors.quantile(0.50),
            'rel_error_p75': rel_errors.quantile(0.75),
            'rel_error_p90': rel_errors.quantile(0.90),
            'ron_p10': ron.quantile(0.1),
            'ron_p25': ron.quantile(0.25),
            'ron_p50': ron.quantile(0.50),
            'ron_p75': ron.quantile(0.75),
            'ron_p90': ron.quantile(0.90)
        }
    
    def print_report(self):
        """Print formatted analysis report."""
        if not self.summary:
            self.analyze()
        
        s = self.summary
        
        print("\n" + "="*70)
        print("BASELINE PERFORMANCE ANALYSIS")
        print("="*70)
        
        print("\n1. OPTION PAYOFF STATISTICS")
        print("-" * 70)
        ps = s['payoff_stats']
        print(f"  Min:      ${ps['min']:.2f}")
        print(f"  Max:      ${ps['max']:.2f}")
        print(f"  Mean:     ${ps['mean']:.2f}")
        print(f"  Std Dev:  ${ps['std']:.2f}")
        print(f"  Median:   ${ps['median']:.2f}")
        
        print("\n2. HEDGING ERROR STATISTICS")
        print("-" * 70)
        hs = s['hedging_error_stats']
        print(f"  Absolute Error (dollars):")
        print(f"    Mean:    ${hs['abs_mean']:.4f}")
        print(f"    Std Dev: ${hs['abs_std']:.4f}")
        print(f"    Median:  ${hs['abs_median']:.4f}")
        print(f"    Range:   ${hs['abs_min']:.4f} - ${hs['abs_max']:.4f}")
        
        print(f"\n  Absolute Error (% of initial option value):")
        print(f"    Mean:    {hs['abs_pct_mean']:.2f}%")
        print(f"    Std Dev: {hs['abs_pct_std']:.2f}%")
        print(f"    Median:  {hs['abs_pct_median']:.2f}%")
        print(f"    Range:   {hs['abs_pct_min']:.2f}% - {hs['abs_pct_max']:.2f}%")
        
        print(f"\n  Return on Notional (error % of initial option value - RECOMMENDED):")
        print(f"    Mean:    {hs['ron_mean']:.2f}%")
        print(f"    Std Dev: {hs['ron_std']:.2f}%")
        print(f"    Median:  {hs['ron_median']:.2f}%")
        print(f"    Range:   {hs['ron_min']:.2f}% - {hs['ron_max']:.2f}%")
        
        print(f"\n  Relative Error (% of payoff - ITM paths only, {hs['itm_paths_pct']:.0f}% of paths):")
        if hs['rel_mean_pct_itm'] is not None:
            print(f"    Mean:    {hs['rel_mean_pct_itm']:.2f}%")
            print(f"    Std Dev: {hs['rel_std_pct_itm']:.2f}%")
            print(f"    Count:   {hs['rel_count_itm']}/{len(self.df)} paths")
        else:
            print(f"    (No ITM paths in sample)")
        
        print(f"\n  Note: Return on Notional is recommended for comparison because it's")
        print(f"        unaffected by OTM paths where payoff=0 (dividing by notional value).")
        
        print("\n3. TRANSACTION COST STATISTICS")
        print("-" * 70)
        ts = s['transaction_cost_stats']
        print(f"  Total Costs:        ${ts['total']:.2f}")
        print(f"  Mean per Path:      ${ts['mean']:.4f}")
        print(f"  Std Dev:            ${ts['std']:.4f}")
        print(f"  Range:              ${ts['min']:.4f} - ${ts['max']:.4f}")
        print(f"  As % of Total Payoff: {ts['as_pct_payoff_total']:.2f}%")
        print(f"  As % of Payoff (mean): {ts['as_pct_payoff_mean']:.2f}%")
        
        print("\n4. HEDGER P&L STATISTICS")
        print("-" * 70)
        ps = s['pnl_stats']
        print(f"  Total P&L:    ${ps['total']:.2f}")
        print(f"  Mean P&L:     ${ps['mean']:.4f}")
        print(f"  Std Dev:      ${ps['std']:.4f}")
        print(f"  Min P&L:      ${ps['min']:.4f}")
        print(f"  Max P&L:      ${ps['max']:.4f}")
        print(f"  Winning paths: {ps['positive_count']} / {int(ps['positive_count'] + ps['negative_count'])}")
        print(f"  Win Rate:     {ps['win_rate_pct']:.1f}%")
        
        print("\n5. EFFICIENCY METRICS (Key for RL comparison)")
        print("-" * 70)
        es = s['efficiency_metrics']
        print(f"  Total Payoff:           ${es['total_payoff']:.2f}")
        print(f"  Total Cost (Error+TC):  ${es['total_cost']:.2f}")
        print(f"  Cost as % Payoff:       {es['cost_as_pct_payoff']:.2f}%")
        print(f"  Avg Cost per Option:    ${es['avg_cost_per_option']:.4f}")
        print(f"    - Hedging Error:      {es['hedging_error_pct']:.2f}%")
        print(f"    - Transaction Costs:  {es['transaction_cost_pct']:.2f}%")
        
        print("\n6. DISTRIBUTION ANALYSIS (Percentiles)")
        print("-" * 70)
        ds = s['distribution']
        print(f"  Absolute Hedging Error (dollars):")
        print(f"    P10: ${ds['abs_error_p10']:.4f}")
        print(f"    P25: ${ds['abs_error_p25']:.4f}")
        print(f"    P50: ${ds['abs_error_p50']:.4f}")
        print(f"    P75: ${ds['abs_error_p75']:.4f}")
        print(f"    P90: ${ds['abs_error_p90']:.4f}")
        
        print(f"\n  Absolute Hedging Error (% of initial option value):")
        print(f"    P10: {ds['abs_error_pct_p10']:.2f}%")
        print(f"    P25: {ds['abs_error_pct_p25']:.2f}%")
        print(f"    P50: {ds['abs_error_pct_p50']:.2f}%")
        print(f"    P75: {ds['abs_error_pct_p75']:.2f}%")
        print(f"    P90: {ds['abs_error_pct_p90']:.2f}%")
        
        print(f"\n  Return on Notional (% of initial option value):")
        print(f"    P10: {ds['ron_p10']:.2f}%")
        print(f"    P25: {ds['ron_p25']:.2f}%")
        print(f"    P50: {ds['ron_p50']:.2f}%")
        print(f"    P75: {ds['ron_p75']:.2f}%")
        print(f"    P90: {ds['ron_p90']:.2f}%")
        
        print("\n" + "="*70)
        print("BASELINE INTERPRETATION")
        print("="*70)
        print(f"""
The classical delta-hedging baseline:
  • Mean absolute error: ${hs['abs_mean']:.2f} per path
  • Mean return on notional: {hs['ron_mean']:.2f}% (UNAFFECTED BY OTM PATHS)
  • Incurs transaction costs: {ts['as_pct_payoff_total']:.2f}% of total payoff
  • Win rate (paths with positive P&L): {ps['win_rate_pct']:.1f}%
  • ITM paths: {hs['itm_paths_pct']:.0f}% (OTM paths don't distort RON metric)

For the RL agent to be worthwhile, it should:
  ✓ Reduce return on notional by at least 20-30%
  ✓ Improve absolute error by 10-20%
  ✓ Improve win rate by 5-15 percentage points
  ✓ Maintain consistency (lower std dev)

WHY Return on Notional?
  • Fair comparison unaffected by whether paths finish ITM or OTM
  • Measures: Cost / Initial Option Value (what was paid upfront)
  • Directly comparable across different strikes and expiries

This is the target to beat!
""")
        print("="*70 + "\n")


if __name__ == "__main__":
    from classical_hedger import ClassicalDeltaHedger
    from data_generator import generate_training_paths
    
    print("\n" + "="*70)
    print("PHASE 2B: BASELINE ANALYSIS")
    print("="*70)
    
    # Generate data
    print("\nGenerating training data...")
    df = generate_training_paths(S0=100, K=105, T=0.25, r=0.05, sigma=0.20,
                                 num_paths=100, num_steps=20)
    
    # Run hedger
    print("Running classical delta hedger...")
    hedger = ClassicalDeltaHedger(K=105, r=0.05, transaction_cost_bps=5.0)
    results = hedger.hedge_paths(df)
    final_errors = hedger.calculate_final_hedging_error(results)
    
    # Analyze
    print("Analyzing results...")
    analyzer = BaselineAnalyzer(final_errors)
    analyzer.print_report()
