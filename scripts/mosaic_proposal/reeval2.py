"""
Re-fold sequences with bagel K times each (the oracle is noisy, see docs section 11b).
Usage: reeval2.py JOBS.json OUT.json  where JOBS.json is a list of {"name", "target", "sequence"}; results are appended to OUT.json.
"""

import json
import pathlib as pl
import sys

import numpy as np

from bagel.oracles import OraclesResultDict
from common import build_system, make_oracle

K = 8


def main() -> None:
    jobs = json.loads(pl.Path(sys.argv[1]).read_text())
    out = pl.Path(sys.argv[2])
    results = json.loads(out.read_text()) if out.exists() else {}
    oracle = make_oracle()
    for job in jobs:
        if job['name'] in results:
            continue
        state = build_system(oracle, job['sequence'], job['target']).states[0]
        vals = []
        for _ in range(K):
            res = OraclesResultDict()
            res[oracle] = oracle.predict(chains=state.chains)
            vals.append(float(sum(float(np.asarray(t.compute(res)[1]).item()) for t in state.energy_terms)))
        results[job['name']] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)), 'values': vals}
        out.write_text(json.dumps(results, indent=1))
        print(f"{job['name']}: re-folded mean {np.mean(vals):.3f} sd {np.std(vals, ddof=1):.3f}", flush=True)


if __name__ == '__main__':
    main()
