import random
import fire
import bagel as bg
import os
import logging
import pathlib as pl

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main(
    use_modal: bool = False,
    binder_sequence: str = None,
    optimization_params: dict = None,
    output_dir: str = 'data/CA4-binder'
):

    # Check
    print(f'Whether to use modal: {use_modal}')

    # PART 1: Define the target protein
    # EGFR: UniProt P00533
    target_sequence = "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS"
    target_sequence = target_sequence[304:481] # only consider part of one of the epitopes
    # CEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQV

    # Now define the mutability of the residues, all immutable in this case since this is the target sequence
    mutability = [False for _ in range(len(target_sequence))]

    # Now define the chain
    residues_target = [
        bg.Residue(name=aa, chain_ID='EGFR', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]

    # Now define residues in the hotspot where you want to bind.
    residue_ids = [49, 50, 51, 52, 53, 130, 131, 132, 134, 135, 136]

    residues_hotspot = [residues_target[i] for i in residue_ids]
    target_chain = bg.Chain(residues=residues_target)

    # For the binder, start with a random sequence of amino acids selecting randomly from the 30 amino acids
    binder_length = 30

    # Jakub: RESTART!
    if binder_sequence is None:
        binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])
    else:
        assert len(binder_sequence) == binder_length, 'Binder sequence must be of length 30'


    # Now define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability = [True for _ in range(len(binder_sequence))]
    # Now define the chain
    residues_binder = [
        bg.Residue(name=aa, chain_ID='BIND', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability))
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # Now define the folding algorithm, run locally not on modal
    config = {
        'output_pdb': False,
        'output_cif': False,
        'glycine_linker': "",
        'position_ids_skip': 512,
    }

    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal, config=config
    )

    # Now define the energy terms to be applied to the chain. In this example, all terms apply to all residues
    energy_terms = [
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=2.0
        ),
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            weight=1.0
            ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=4.0,
        ),
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=0.2,
        ),
    ]

    state = bg.State(
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
        name='state',
    )

    # Now define the system
    initial_system = bg.System(
        states=[state],
        name='EGFR-binder'
    )

    # Now define the minimizer
    mutator = bg.mutation.Canonical()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    print(f'Current directory: {current_dir}')

    # Use optimization parameters if provided, otherwise use defaults
    if optimization_params is None:
        optimization_params = {
            'high_temperature': 1,
            'low_temperature': 1e-3,
            'n_steps_high': 100,
            'n_steps_low': 400,
            'n_cycles': 100,
        }

    # Note: the result in the paper comes from a parallel tempering implementation in previous version of the codebase
    # which is not yet implemented, but the simulated tempering below should produce equivalent results
    # The difference is only in the optimization; not in the energy terms, i.e. the definition of the System energy
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=mutator,
        high_temperature=optimization_params['high_temperature'],
        low_temperature=optimization_params['low_temperature'],
        n_steps_high=optimization_params['n_steps_high'],
        n_steps_low=optimization_params['n_steps_low'],
        n_cycles=optimization_params['n_cycles'],
        preserve_best_system_every_n_steps=optimization_params['n_steps_high'] + optimization_params['n_steps_low'],
        log_frequency=1,
        log_path=pl.Path(os.path.join(current_dir, output_dir)),
    )

    # Run optimization and return the best system
    best_system = minimizer.minimize_system(system=initial_system)
    return best_system


if __name__ == '__main__':
    fire.Fire(main)
