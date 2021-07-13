import sys

from vowpalwabbit import pyvw


opts = sys.argv[1:]
model = pyvw.vw(" ".join(opts))
