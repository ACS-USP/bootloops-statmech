#!/usr/bin/env python3
"""replicant selftest: discovers tests/test_*.py functions without pytest.

  python3 tools/replicant/selftest.py [--full]

@slow tests run only with --full.  Exit 0 only if every test passes.
"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools/, so `import replicant` works
sys.path.insert(0, os.path.join(HERE, "tests"))


def summary_line(passed, failed, skipped):
    return (f"replicant selftest: {passed} passed, {failed} failed, {skipped} skipped "
            "(slow; use --full, about 6-7 minutes on a laptop, measured 6 min 42 s)")


def main(argv):
    full = "--full" in argv
    names = sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "tests"))
                   if f.startswith("test_") and f.endswith(".py"))
    passed = failed = skipped = 0
    for mod_name in names:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if not (name.startswith("test_") and callable(fn)):
                continue
            if getattr(fn, "slow", False) and not full:
                skipped += 1
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
    print(summary_line(passed, failed, skipped))
    return 1 if failed or not passed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
