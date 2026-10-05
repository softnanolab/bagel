#!/usr/bin/env python3
"""Offline verification: check imports, system structure, and basic setup."""

import sys
sys.path.insert(0, '/home/user/bagel/src')

print('Verifying TNF binder design setup...\n')

try:
    print('1. Checking BAGEL imports...')
    import bagel as bg
    print('   ✓ bagel')
    
    print('2. Checking custom modules...')
    from utils import ipSAEEnergy, make_darpin_scaffold
    print('   ✓ utils.ipSAEEnergy')
    print('   ✓ utils.make_darpin_scaffold')
    
    from ph_switch_histidine import BuriedHistidineEnergy, HistidineCarboxylateContactEnergy
    print('   ✓ ph_switch_histidine.BuriedHistidineEnergy')
    print('   ✓ ph_switch_histidine.HistidineCarboxylateContactEnergy')
    
    print('\n3. Creating test structures...')
    # Test DARPin scaffold
    darpin_residues = make_darpin_scaffold(length=124, n_variable=12)
    assert len(darpin_residues) == 124
    mutable_count = sum(1 for r in darpin_residues if r.mutable)
    assert mutable_count == 12, f'Expected 12 mutable, got {mutable_count}'
    print(f'   ✓ DARPin scaffold: 124 residues, {mutable_count} mutable')
    
    # Test TNF sequences
    TNF_HUMAN = 'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'
    assert len(TNF_HUMAN) == 157
    print(f'   ✓ TNF-human: {len(TNF_HUMAN)} residues')
    
    TNF_MOUSE = 'LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL'
    assert len(TNF_MOUSE) == 156
    print(f'   ✓ TNF-mouse: {len(TNF_MOUSE)} residues')
    
    # Test chain creation
    print('\n4. Creating chains and states...')
    generic_length = 80
    generic_seq = 'A' * generic_length
    generic_residues = [bg.Residue(name='A', chain_ID='B', index=i, mutable=True) for i in range(generic_length)]
    generic_chain = bg.Chain(residues=generic_residues)
    print(f'   ✓ Generic binder: {generic_length} residues')
    
    tnf_residues = [bg.Residue(name=aa, chain_ID='T', index=i, mutable=False) for i, aa in enumerate(TNF_HUMAN)]
    tnf_chain = bg.Chain(residues=tnf_residues)
    print(f'   ✓ TNF target: {len(tnf_residues)} residues (immutable)')
    
    print('\n5. Checking energy term instantiation...')
    # We can't create energy terms without an oracle, but we can verify the classes exist
    print(f'   ✓ ipSAEEnergy: {ipSAEEnergy.__name__}')
    print(f'   ✓ BuriedHistidineEnergy: {BuriedHistidineEnergy.__name__}')
    print(f'   ✓ HistidineCarboxylateContactEnergy: {HistidineCarboxylateContactEnergy.__name__}')
    
    print('\n' + '='*60)
    print('✓ All offline verifications PASSED')
    print('='*60)
    print('\nReady to run full designs with Modal authentication.')
    print('To authenticate: modal token new')
    
except Exception as e:
    print(f'\n✗ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
