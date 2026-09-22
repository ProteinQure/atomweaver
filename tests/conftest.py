"""Use the released element vocabulary for inference regressions."""

import os

os.environ["ATOMWEAVER_ELEMENT_VOCAB"] = "5"
os.environ["ATOMWEAVER_SAMPLE_RECYCLES"] = "1"

os.environ["NTHREADS"] = "2"
