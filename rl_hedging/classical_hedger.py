#!/usr/bin/env python3
"""
RL Hedging: Classical Delta Hedger (Baseline Strategy)

Implements textbook Black-Scholes delta hedging:
- At each time step, rebalance to the BS-optimal delta
- Track hedging error: (Option payoff) - (Hedge portfolio value)
- Deduct transaction costs for each rebalance

This is our benchmark to beat with the RL agent.
"""

import numpy as np
import pandas as pd
from bs.pricing import calculate_black_scholes_price


class ClassicalDeltaHedger:
    """
    Delta hedging using Black-Scholes deltas.
    
    Strategy:
    1. At each step, calculate BS delta for current spot and time remaining
    2. Rebalance hedge position to match delta
    3. Track P&L from hedge adjustments
    4. Deduct transaction costs
    5. At maturity, calculate final hedging error
    """
    
    def __init__(self, K, r, transaction_cost_bps=5.0):
        """
        Initialize hedger.
        
        Parameters
        ----------
        K : float
            Strike price (from data)
        r : float
            Risk-free rate
        transaction_cost_bps : float
            Transaction cost in basis points (default: 5 = 0.05%)
            Applied as a percentage of the notional value traded
        """
        self.K = K
        self.r = r
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
    
    def hedge_paths(self, training_df):
        """
        Apply delta hedging to all paths in training data.
        
        Parameters
        ----------
        training_df : pd.DataFrame
            Output from data_generator.py with columns:
            [path_id, step, spot_price, time_remaining, bs_delta, option_value, 
             payoff_at_maturity, next_spot_price]
        
        Returns
        -------
        pd.DataFrame
            Results for each (path, step) with columns:
            [path_id, step, spot_price, bs_delta, hedge_position, 
             cash_account, rebalance_amount, transaction_cost, 
             hedge_pnl, cumulative_pnl]
        """
        results = []
        dt = 0.25 / 20  # Time step (assuming T=0.25, num_steps=20)
        
        # Process each path separately
        for path_id in training_df['path_id'].unique():
            path_data = training_df[training_df['path_id'] == path_id].sort_values('step')
            
            prev_hedge = 0  # Start with no hedge
            cash = 0  # Start with zero cash
            
            for _, row in path_data.iterrows():
                current_spot = row['spot_price']
                next_spot = row['next_spot_price']
                time_remaining = row['time_remaining']
                bs_delta = row['bs_delta']
                
                # Target hedge position is BS delta
                target_hedge = bs_delta
                
                # Rebalance amount
                rebalance_amount = target_hedge - prev_hedge
                
                # If rebalancing:
                # - Positive rebalance_amount: buy shares (borrow cash)
                # - Negative rebalance_amount: sell shares (lend out cash)
                # Cost: |rebalance_amount| * current_spot * transaction_cost_bps
                transaction_cost = abs(rebalance_amount) * current_spot * self.transaction_cost_bps
                
                # Cash flow from rebalancing:
                # - Buying shares costs: rebalance_amount * current_spot
                # - We need to fund this from cash (or borrow it)
                # - We also pay transaction costs
                cash_flow = -rebalance_amount * current_spot - transaction_cost
                cash += cash_flow
                
                # Interest accrual on cash (daily compounding approximation)
                # If cash < 0, we're borrowing; if cash > 0, we're lending
                interest = cash * self.r * dt
                cash += interest
                
                # P&L from stock position: if I'm holding `prev_hedge` shares and spot moves
                stock_pnl = prev_hedge * (next_spot - current_spot)
                
                results.append({
                    'path_id': path_id,
                    'step': row['step'],
                    'spot_price': current_spot,
                    'next_spot_price': next_spot,
                    'bs_delta': bs_delta,
                    'hedge_position': target_hedge,
                    'prev_hedge_position': prev_hedge,
                    'rebalance_amount': rebalance_amount,
                    'transaction_cost': transaction_cost,
                    'stock_pnl': stock_pnl,
                    'cash_account': cash,
                    'payoff_at_maturity': row['payoff_at_maturity']
                })
                
                prev_hedge = target_hedge
        
        return pd.DataFrame(results)
    
    def calculate_final_hedging_error(self, results_df, training_df=None):
        """
        Calculate final hedging error for each path.
        
        Hedging error = Option payoff - (Hedge portfolio value)
        
        Hedge portfolio value at maturity includes:
        - Stock holdings: final_delta * final_spot_price
        - Cash account: accumulated cash (with interest)
        
        Net result: How much profit/loss from being hedged vs option payoff
        
        NOTE: We use multiple error metrics to avoid OTM path distortion:
        - abs_hedging_error: Dollar error (works for all paths)
        - rel_hedging_error_pct: Relative error (ITM only, meaningful percentage)
        - return_on_notional: Error as % of BS option value (fair comparison)
        - is_itm: Flag for filtering in analysis
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Output from hedge_paths()
        training_df : pd.DataFrame, optional
            Input to hedge_paths() (contains initial option values)
            If not provided, will estimate sigma as 0.2
        
        Returns
        -------
        pd.DataFrame
            Summary stats per path with columns:
            [path_id, final_hedge_position, final_spot_price, option_payoff, 
             stock_value, cash_account, hedge_portfolio_value, hedging_error, 
             abs_hedging_error, rel_hedging_error_pct, return_on_notional, 
             is_itm, total_transaction_costs, net_pnl]
        """
        final_errors = []
        
        for path_id in results_df['path_id'].unique():
            path_results = results_df[results_df['path_id'] == path_id].sort_values('step')
            
            # Final values
            final_row = path_results.iloc[-1]
            final_hedge_position = final_row['hedge_position']
            final_spot = final_row['next_spot_price']  # Use next_spot since we're at end
            option_payoff = final_row['payoff_at_maturity']
            cash_account = final_row['cash_account']
            
            # Get initial option value
            if training_df is not None:
                # Use training data if provided
                training_initial = training_df[training_df['path_id'] == path_id].iloc[0]
                initial_option_value = training_initial['option_value']
            else:
                # Calculate from initial spot price using BS (assume sigma=0.2)
                initial_row = path_results.iloc[0]
                initial_spot = initial_row['spot_price']
                # Reconstruct initial time using the step size in results
                # because time_remaining is only guaranteed in training data.
                num_steps = len(path_results)
                dt = 0.25 / max(num_steps, 1)
                initial_time = dt * num_steps
                initial_option_value = calculate_black_scholes_price(
                    initial_spot, self.K, initial_time, self.r, 0.2, 'call'
                )
            
            # Stock value at maturity
            stock_value = final_hedge_position * final_spot
            
            # Total hedge portfolio value
            hedge_portfolio_value = stock_value + cash_account
            
            # Hedging error: option payoff vs portfolio value
            # If portfolio value > payoff, we made money
            # If portfolio value < payoff, we lost money
            hedging_error = option_payoff - hedge_portfolio_value
            
            # Total transaction costs (cumulative)
            total_transaction_costs = path_results['transaction_cost'].sum()
            
            # Multiple error metrics (different perspectives)
            # 1. Absolute error as % of initial option value
            abs_error_pct = (abs(hedging_error) / initial_option_value * 100) if initial_option_value > 0 else 0
            
            # 2. Relative error (ITM only) - meaningful only when option has value
            is_itm = option_payoff > 0
            rel_error = (abs(hedging_error) / option_payoff * 100) if option_payoff > 0 else None
            
            # 3. Return on Notional - error as % of initial option value (always meaningful)
            return_on_notional = abs_error_pct
            
            final_errors.append({
                'path_id': path_id,
                'final_hedge_position': final_hedge_position,
                'final_spot_price': final_spot,
                'option_payoff': option_payoff,
                'stock_value': stock_value,
                'cash_account': cash_account,
                'hedge_portfolio_value': hedge_portfolio_value,
                'hedging_error': hedging_error,
                'abs_hedging_error': abs(hedging_error),
                'abs_hedging_error_pct': abs_error_pct,
                'rel_hedging_error_pct': rel_error,
                'return_on_notional': return_on_notional,
                'is_itm': is_itm,
                'total_transaction_costs': total_transaction_costs,
                'net_pnl': -hedging_error  # P&L is opposite of error
            })
        
        return pd.DataFrame(final_errors)


if __name__ == "__main__":
    from rl_hedging.data_generator import generate_training_paths
    
    print("\n" + "="*70)
    print("PHASE 2A: CLASSICAL DELTA HEDGER")
    print("="*70)
    
    # Generate training data
    print("\nGenerating training data...")
    df = generate_training_paths(S0=100, K=105, T=0.25, r=0.05, sigma=0.20,
                                 num_paths=100, num_steps=20)
    print(f"✓ Generated {len(df)} rows")
    
    # Initialize hedger
    print("\nInitializing classical delta hedger...")
    hedger = ClassicalDeltaHedger(K=105, r=0.05, transaction_cost_bps=5.0)
    print(f"✓ Transaction cost: 5.0 bps (0.05%)")
    
    # Run hedging strategy
    print("\nApplying delta hedging to all paths...")
    results = hedger.hedge_paths(df)
    print(f"✓ Generated {len(results)} step results")
    
    # Calculate final errors
    print("\nCalculating hedging errors...")
    final_errors = hedger.calculate_final_hedging_error(results)
    print(f"✓ Computed errors for {len(final_errors)} paths")
    
    print("\n" + "="*70)
    print("RESULTS SAMPLE (First 5 paths)")
    print("="*70)
    print(final_errors.head().to_string())
    
    print("\n" + "="*70)
    print("HEDGING ERROR ANALYSIS")
    print("="*70)
    
    print(f"\nOption Payoff Statistics:")
    print(f"  Min payoff:    ${final_errors['option_payoff'].min():.2f}")
    print(f"  Max payoff:    ${final_errors['option_payoff'].max():.2f}")
    print(f"  Mean payoff:   ${final_errors['option_payoff'].mean():.2f}")
    
    print(f"\nAbsolute Hedging Error:")
    print(f"  Min error:     ${final_errors['abs_hedging_error'].min():.4f}")
    print(f"  Max error:     ${final_errors['abs_hedging_error'].max():.4f}")
    print(f"  Mean error:    ${final_errors['abs_hedging_error'].mean():.4f}")
    print(f"  Std dev:       ${final_errors['abs_hedging_error'].std():.4f}")
    
    print(f"\nRelative Hedging Error (% of payoff):")
    print(f"  Min:           {final_errors['rel_hedging_error_pct'].min():.2f}%")
    print(f"  Max:           {final_errors['rel_hedging_error_pct'].max():.2f}%")
    print(f"  Mean:          {final_errors['rel_hedging_error_pct'].mean():.2f}%")
    print(f"  Std dev:       {final_errors['rel_hedging_error_pct'].std():.2f}%")
    
    print(f"\nTransaction Costs:")
    print(f"  Total (all paths):  ${final_errors['total_transaction_costs'].sum():.2f}")
    print(f"  Mean per path:      ${final_errors['total_transaction_costs'].mean():.4f}")
    print(f"  As % of payoff:     {(final_errors['total_transaction_costs'].sum() / final_errors['option_payoff'].sum() * 100):.2f}%")
    
    print(f"\nNet P&L (Hedger Profit/Loss):")
    print(f"  Total:     ${final_errors['net_pnl'].sum():.2f}")
    print(f"  Mean:      ${final_errors['net_pnl'].mean():.4f}")
    print(f"  Positive:  {(final_errors['net_pnl'] > 0).sum()} out of {len(final_errors)} paths")
    print(f"  Win rate:  {(final_errors['net_pnl'] > 0).sum() / len(final_errors) * 100:.1f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHT FOR RL AGENT")
    print("="*70)
    print(f"""
The baseline delta-hedging strategy achieves:
  • Mean hedging error: ${final_errors['abs_hedging_error'].mean():.4f}
  • As % of option value: {final_errors['rel_hedging_error_pct'].mean():.2f}%
  • Transaction costs: ${final_errors['total_transaction_costs'].mean():.4f} per path

The RL agent should try to:
  1. Reduce hedging error (smaller deviations from payoff)
  2. Reduce transaction costs (fewer rebalances or smarter timing)
  3. Improve win rate (more paths with positive P&L)

Target improvement: 10-30% reduction in total cost (error + transaction costs)
""")
    
    print("="*70 + "\n")
