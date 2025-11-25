import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest  # type: ignore
from pathlib import Path

from bagel.analysis.analyzer import MonteCarloAnalyzer


FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'analyzer'


@pytest.fixture(scope='module')
def analyzer():
    return MonteCarloAnalyzer(str(FIXTURE_DIR))


def test_sequences_attached_to_energy_frames(analyzer):
    for df in (analyzer.current_energies_df, analyzer.best_energies_df):
        assert 'state_A:sequence' in df.columns
        row = df[df['step'] == 2]
        assert row['state_A:sequence'].iloc[0] in {'AAC:BBC', 'AAA:BBB'}


def test_energy_weights_loaded(analyzer):
    assert analyzer.energy_weights['state_A:hydrophobicity'] == 2.0
    assert analyzer.energy_weights['state_A:packing'] == 0.5


def _assert_weighted_line(ax, label, expected):
    line = next(l for l in ax.get_lines() if l.get_label() == label)
    np.testing.assert_allclose(line.get_ydata(), expected)


def test_plot_energies_current_weighted(analyzer):
    ax = analyzer.plot_energies(weighted=True, use_best=False)
    fig = ax.figure
    try:
        expected_hydro = analyzer.current_energies_df['state_A:hydrophobicity'] * 2.0
        expected_packing = analyzer.current_energies_df['state_A:packing'] * 0.5
        _assert_weighted_line(ax, 'state_A:hydrophobicity', expected_hydro)
        _assert_weighted_line(ax, 'state_A:packing', expected_packing)
        labels = {line.get_label() for line in ax.get_lines()}
        assert 'state_A:sequence' not in labels
        assert 'system_energy' in labels
    finally:
        plt.close(fig)


def test_plot_energies_best_weighted(analyzer):
    ax = analyzer.plot_energies(weighted=True, use_best=True)
    fig = ax.figure
    try:
        expected = analyzer.best_energies_df['state_A:hydrophobicity'] * 2.0
        _assert_weighted_line(ax, 'state_A:hydrophobicity', expected)
    finally:
        plt.close(fig)
