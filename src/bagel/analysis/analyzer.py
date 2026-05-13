import logging
import pathlib as pl
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import AtomArray

logger = logging.getLogger(__name__)


@dataclass
class State:
    name: str
    sequences: dict[int, str]
    # structures: dict[int, AtomArray] not implemented yet


class Analyzer(ABC):
    def __init__(self, path: str):
        self.path = pl.Path(path)


class MonteCarloAnalyzer(Analyzer):
    def __init__(self, path: str):
        super().__init__(path)
        self._load_data()

    def _load_data(self):
        # Load optimization and energy data
        self.optimization_df = pd.read_csv(self.path / 'optimization.log')
        self.current_energies_df = pd.read_csv(self.path / 'current' / 'energies.csv')
        self.best_energies_df = pd.read_csv(self.path / 'best' / 'energies.csv')

        # Initialize dictionaries for FASTA data
        self.current_sequences: dict[str, dict[int, str]] = {}
        self.best_sequences: dict[str, dict[int, str]] = {}
        self.config_df = self._load_config()
        self.energy_weights = self._build_energy_weight_lookup()

        # Load FASTA files from 'current' and 'best' directories
        for directory in ['current', 'best']:
            fasta_dir = self.path / directory
            if not fasta_dir.exists():
                logger.warning(f'Directory {fasta_dir} does not exist. Skipping FASTA loading.')
                continue

            for fasta_file in fasta_dir.glob('*.fasta'):
                state_name = fasta_file.stem  # e.g., 'state_A'
                state_sequences: dict[int, str] = {}

                fasta = FastaFile.read(fasta_file)
                for header in fasta.keys():
                    try:
                        step = int(header)
                    except ValueError:
                        logger.warning(f'Skipping FASTA entry {header} in {fasta_file}: header is not an int')
                        continue
                    state_sequences[step] = fasta[header]

                if directory == 'current':
                    self.current_sequences[state_name] = state_sequences
                else:
                    self.best_sequences[state_name] = state_sequences

        self._attach_sequences_to_df(self.current_energies_df, self.current_sequences)
        self._attach_sequences_to_df(self.best_energies_df, self.best_sequences)

    def _load_config(self) -> pd.DataFrame:
        config_path = self.path / 'config.csv'
        if not config_path.exists():
            logger.warning(f'No config.csv found under {self.path} - weights will default to 1.0.')
            return pd.DataFrame(columns=['state', 'energy', 'weight'])
        return pd.read_csv(config_path)

    def _build_energy_weight_lookup(self) -> dict[str, float]:
        if self.config_df.empty:
            return {}
        return {f'{row["state"]}/{row["energy"]}': float(row['weight']) for _, row in self.config_df.iterrows()}

    def _attach_sequences_to_df(self, energies_df: pd.DataFrame, sequences: dict[str, dict[int, str]]) -> None:
        if 'step' not in energies_df.columns:
            logger.warning('Energies dataframe missing step column; cannot attach sequences.')
            return

        step_series = energies_df['step'].astype(int)
        for state_name, seq_by_step in sequences.items():
            column = f'{state_name}/sequence'
            energies_df[column] = step_series.map(seq_by_step.get)

    def plot_energies(self, weighted: bool = True, use_best: bool = False, ax: plt.Axes = None):
        """
        Plot the evolution of system energy together with individual energy terms.

        Parameters
        ----------
        weighted : bool, optional
            If True, multiply individual energy terms by their configured weights before plotting.
        use_best : bool, optional
            If True, plot the best energies; otherwise plot the current energies.
        """
        energies_df = self.best_energies_df if use_best else self.current_energies_df
        if 'step' not in energies_df.columns:
            raise ValueError('Energies dataframe must contain a "step" column.')

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        weight_lookup = self.energy_weights if weighted else {}
        columns_to_plot = [
            col for col in energies_df.columns if col not in {'step', 'system_energy'} and not col.endswith('/sequence')
        ]

        for column in columns_to_plot:
            series = energies_df[column]
            if weighted and column in weight_lookup:
                series = series * weight_lookup[column]
            ax.plot(energies_df['step'], series, label=column)

        # Add system energy as a bold line for emphasis
        ax.plot(energies_df['step'], energies_df['system_energy'], label='system_energy', linewidth=2.5, color='black')

        title_prefix = 'Best' if use_best else 'Current'
        weighting_label = 'Weighted' if weighted and weight_lookup else 'Unweighted'
        ax.set_title(f'{title_prefix} Energies ({weighting_label})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.legend(loc='best', fontsize='small', ncol=2)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return ax


class SimulatedTemperingAnalyzer(MonteCarloAnalyzer):
    def __init__(self, path: str):
        super().__init__(path)
        self.acceptance_rates = {}

    def plot_acceptance_rate(self):
        """
        Plots the cumulative evolution of acceptance rates by temperature.
        """
        # Calculate cumulative acceptance rates for each temperature
        for temp in self.optimization_df['temperature'].unique():
            temp_data = self.optimization_df[self.optimization_df['temperature'] == temp].sort_values('step')
            cum_accepts = temp_data['accept'].astype(int).cumsum()
            # Calculate rate as cumulative accepts divided by total steps at each point
            self.acceptance_rates[temp] = []
            for i in range(len(temp_data)):
                self.acceptance_rates[temp].append(cum_accepts.tolist()[i] / (i + 1))

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot cumulative evolution
        for temperature in sorted(self.acceptance_rates.keys()):
            temp_data = self.optimization_df[self.optimization_df['temperature'] == temperature].sort_values('step')
            ax.plot(temp_data['step'], self.acceptance_rates[temperature], label=f'Temperature {temperature}')
            logger.info(
                f'Temperature {temperature} final acceptance rate: {self.acceptance_rates[temperature][-1]:.3f}'
            )

        ax.set_xlabel('Step')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rate Evolution')
        ax.legend()

        plt.tight_layout()
        return fig, ax
