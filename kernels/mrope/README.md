## Setup
1. Install liger_kernel
```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
python3 -m pip install -e .
```

## Run instructions
1. Check for correctness: `pytest test_correctness.py -v > test_correctness.log 2>&1`
2. Benchmark:
- Quick benchmark: `python benchmark.py --quick --output quick_results.json`
- Full benchmark: `python benchmark.py --output full_results.json`
