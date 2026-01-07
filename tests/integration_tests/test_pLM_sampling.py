import pathlib as pl
import numpy as np
import bagel as bg


def test_pLM_sampling_does_not_raise_exceptions_with_nominal_inputs(
    plm_only_state: bg.State, test_output_path: pl.Path
) -> None:
    test_system = bg.System(states=[plm_only_state], name='test_pLM')

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.Canonical(),
        temperature=1.0,
        n_steps=3,
        callbacks=[bg.callbacks.DefaultLogger(log_interval=1)],
        log_path=test_output_path,
    )

    final_system = minimizer.minimize_system(system=test_system)


def test_pLM_sampling_early_stopping_stops_on_plateau(
    plm_only_state: bg.State, test_output_path: pl.Path
) -> None:
    """Test that early stopping correctly stops optimization when energy plateaus."""
    np.random.seed(42)
    
    test_system = bg.System(states=[plm_only_state], name='test_pLM')
    
    # Track actual steps executed
    step_count = []
    
    class StepCounter(bg.callbacks.Callback):
        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            step_count.append(context.step)
    
    # Create early stopping callback with moderate patience
    early_stop = bg.callbacks.EarlyStopping(
        monitor='system_energy',
        patience=50,
        mode='min'
    )
    
    step_counter = StepCounter()
    
    # Create minimizer with very low temperature (causes rapid plateaus)
    # and many steps to allow early stopping to demonstrate
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.Canonical(),
        temperature=1e-6,  # Very low temperature - mutations rarely accepted
        n_steps=1000,  # Many steps to allow early stopping
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=10),
            early_stop,
            step_counter,
        ],
        log_path=test_output_path,
    )
    
    final_system = minimizer.minimize_system(system=test_system)
    
    # Verify that optimization stopped early (actual steps < n_steps)
    actual_steps = len(step_count)
    assert actual_steps < 1000, f'Expected early stopping, but ran all {actual_steps} steps'
    
    # Verify that early stopping was triggered (ran for more than patience steps)
    # This ensures it ran long enough to detect a plateau
    assert actual_steps > 50, f'Expected to run for more than patience (50) steps, but only ran {actual_steps} steps'
    
    # Verify that early stopping flag was set
    assert early_stop._should_stop is True, 'Early stopping should have been triggered'
    
    # Verify final system was returned
    assert final_system is not None
