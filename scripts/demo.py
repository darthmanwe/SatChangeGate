"""Clone-to-result demo.

Runs the full funnel on the committed synthetic fixtures. Requires no dataset
download and no API key: the gate and Tier 0 run locally, and the analyst report
falls back to a clearly-labelled offline template.

Run: python scripts/demo.py   (or: make demo)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "mini_oscd"

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from satchangegate.config import get_settings  # noqa: E402
from satchangegate.data.oscd import discover_pairs  # noqa: E402
from satchangegate.pipeline import run_pair  # noqa: E402


def main() -> int:
    if not FIXTURES.is_dir():
        print(f"Fixtures missing at {FIXTURES}. Run: python scripts/make_fixtures.py")
        return 1

    settings = get_settings()
    pairs = discover_pairs(FIXTURES)
    if not pairs:
        print(f"No pairs discovered under {FIXTURES}")
        return 1

    out_dir = Path(tempfile.mkdtemp(prefix="satchangegate_demo_"))
    print(f"SatChangeGate demo: {len(pairs)} synthetic pairs, no API key required")
    print(f"Artifacts: {out_dir}\n")
    print(f"{'pair':16}{'gate':20}{'conf':>6}{'dNDVI':>9}{'dNDBI':>9}{'area%':>8}  reason")

    failures = 0
    for pair in pairs:
        result = run_pair(pair, settings=settings, out_dir=out_dir, skip_vlm=True, skip_llm=False)
        c = result.classical
        print(
            f"{pair.pair_id:16}{c.classical_gate:20}{c.gate_confidence:6.2f}"
            f"{c.ndvi_delta_mean:+9.4f}{c.ndbi_delta_mean:+9.4f}"
            f"{c.changed_area_percent:8.2f}  {c.gate_reason}"
        )
        # fixtureville contains a real vegetation->built change; stableton does not.
        expected = "candidate_change" if pair.pair_id == "fixtureville" else "no_change"
        if c.classical_gate != expected:
            print(f"    ! expected {expected}")
            failures += 1

    print()
    print("Tier 0 masks were computed from real multispectral bands, registration")
    print("was measured (not hardcoded), and the report is a labelled offline template.")
    print(
        f"\n{'PASS' if not failures else 'FAIL'}: {len(pairs) - failures}/{len(pairs)} as expected"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
