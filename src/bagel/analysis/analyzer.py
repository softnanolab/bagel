import pandas as pd
import pathlib as pl
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging
from biotite.sequence.io.fasta import FastaFile
from dataclasses import dataclass
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

    @abstractmethod
    def analyze(self):
        pass


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
        self.current_sequences = {}
        self.best_sequences = {}

        # Load FASTA files from 'current' and 'best' directories
        for directory in ['current', 'best']:
            fasta_dir = self.path / directory
            if not fasta_dir.exists():
                print(f'Warning: Directory {fasta_dir} does not exist.')
                continue

            for fasta_file in fasta_dir.glob('*.fasta'):
                state_name = fasta_file.stem  # e.g., 'state_A'
                state_sequences = []

                fasta = FastaFile.read(fasta_file)
                num_sequences = len(fasta.lines) // 2  # each sequence is 2 lines, i.e. header and sequence

                for step in range(num_sequences):
                    state_sequences.append(fasta[str(step)])

                if directory == 'current':
                    self.current_sequences[state_name] = state_sequences
                else:
                    self.best_sequences[state_name] = state_sequences

    def analyze(self):
        pass


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
