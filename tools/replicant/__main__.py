"""`python3 -m replicant ...` (from tools/): see replicant/cli.py."""
import sys

from replicant.cli import main

sys.exit(main(sys.argv[1:]))
