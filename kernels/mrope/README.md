
### Version 1
`python3 test_mrope_simple.py`

### Version 2
#### Run full benchmark suite
`python3 test_mrope.py`

#### Run with custom parameters
`python3 test_mrope.py --warmup 5 --benchmark 50 --output-dir my_results`

#### Test specific models
`python3 test_mrope.py --models Qwen2-VL-2B Qwen2.5-VL-3B`

#### Run pytest tests
`pytest test_mrope.py::TestMRoPEKernels -v`