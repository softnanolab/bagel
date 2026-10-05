# Generated with assistance from Claude (bagel-script-builder). Review before use.
"""Smoke tests for all 4 TNF binder design scripts."""

import subprocess, sys

def run_smoke(script_name, description):
    """Run a single smoke test."""
    print(f'\n{"="*60}')
    print(f'Smoke test: {description}')
    print(f'Script: {script_name}')
    print("="*60)
    
    # Minimal run: 1 cycle, 1 step low, 1 step high
    cmd = [
        'python', script_name,
        '--backend=modal',
        '--seed=0',
    ]
    
    try:
        result = subprocess.run(cmd, timeout=300, capture_output=False)
        if result.returncode == 0:
            print(f'✓ {description} PASSED')
            return True
        else:
            print(f'✗ {description} FAILED (exit code {result.returncode})')
            return False
    except subprocess.TimeoutExpired:
        print(f'✗ {description} TIMEOUT')
        return False
    except Exception as e:
        print(f'✗ {description} ERROR: {e}')
        return False

def main():
    print('TNF Binder Design Smoke Tests')
    print('This will verify that all scripts can run without errors.')
    print('(Actual runs will use full step counts.)')
    
    tests = [
        ('binder_generic_monomer_epitope.py', 'Generic binder vs monomer epitope'),
        ('binder_generic_trimer.py', 'Generic binder vs trimer'),
        ('binder_darpin_monomer_epitope.py', 'DARPin vs monomer epitope'),
        ('binder_darpin_trimer.py', 'DARPin vs trimer'),
    ]
    
    results = []
    for script, desc in tests:
        passed = run_smoke(script, desc)
        results.append((desc, passed))
    
    print(f'\n{"="*60}')
    print('Summary:')
    for desc, passed in results:
        status = '✓ PASSED' if passed else '✗ FAILED'
        print(f'  {status}: {desc}')
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f'\n{passed}/{total} tests passed')
    
    if passed == total:
        print('\nAll smoke tests PASSED. Ready to run full designs.')
        sys.exit(0)
    else:
        print('\nSome tests FAILED. Check errors above.')
        sys.exit(1)

if __name__ == '__main__':
    main()
