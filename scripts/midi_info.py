"""Inspect a MIDI's tempo map and instrument layout (debug helper)."""
import sys
from pathlib import Path

import numpy as np
import pretty_midi

mid = Path(sys.argv[1])
pm = pretty_midi.PrettyMIDI(str(mid))
tc = pm.get_tempo_changes()
if isinstance(tc, tuple):
    tidx, tvals = np.asarray(tc[0], dtype=float), np.asarray(tc[1], dtype=float)
else:
    tc = np.asarray(tc, dtype=float)
    tidx, tvals = (tc[:, 0], tc[:, 1]) if tc.ndim == 2 else (np.array([0.0]), tc)
print(f"tempos: n={len(tidx)} min={np.min(tvals):.2f} max={np.max(tvals):.2f} bpm first={tvals[0]:.2f} @ {tidx[0]:.3f}s")
print(f"end_time: {pm.get_end_time():.1f} s")
for i, inst in enumerate(pm.instruments):
    print(f" inst{i}: program={inst.program} drum={inst.is_drum} notes={len(inst.notes)}")
notes = [n for i in pm.instruments if not i.is_drum for n in i.notes]
if notes:
    starts = sorted(n.start for n in notes)
    print(f" non-drum notes: {len(notes)}  first onset {starts[0]:.3f}s  last {starts[-1]:.3f}s")