# gpu-leak-finder

Profiles AI training jobs to find idle, underutilized and wasted GPU time. Correlates utilization, data-loading stalls and memory usage, then reports the cost of each bottleneck with suggested optimizations.

> **Status:** early development. The collectors, analyzers and validation experiments are being built in the open. See the [Roadmap](#roadmap).

---

## Why this exists

GPUs are the most expensive part of training an AI model, and they are often **waiting instead of working**. Common reasons include:

- The **data loader** is too slow, so the GPU sits idle between batches.
- The **batch size** is too small to keep the GPU busy.
- The code **waits on the CPU** (for example, synchronizing too often or doing work on the CPU between steps).
- **Communication** between GPUs in multi-GPU jobs takes more time than expected.
- The GPU is allocated but **nothing is running** (a notebook left open, a job waiting between stages).

The usual tools do not make this easy:

- A single "GPU utilization" number is misleading. It only shows that *some* kernel was running during a sample period, not that the GPU was used *efficiently.*
- Full profilers give a lot of detail, but need expertise to read.
- Almost no tool says what a problem **costs** in money.

`gpu-leak-finder` aims to turn raw measurements into a short, readable answer: *where is GPU time being wasted, how much does it cost, and what should I try first?*

- **Lightweight collection** of GPU, CPU and data-loading metrics during a training run.
- **Bottleneck detection** that separates the main causes of waste.
- **Cost estimation** using a GPU price per hour that the user provides.
- **Suggested fixes** ranked by likely impact.
- **Validated on controlled experiments** where the true cause is known.

---

## What it reports

| Output | Meaning |
|---|---|
| **Idle time** | Share of time the GPU was doing nothing, with timestamps |
| **Underutilization** | Time the GPU was busy but at low efficiency (for example, tiny kernels, low memory bandwidth use) |
| **Data-loading stalls** | Time the training loop waited for the next batch |
| **CPU-side waits** | Time lost to synchronization and CPU work between steps |
| **Memory pressure** | Peak and average memory use, and headroom for a larger batch |
| **Cost of waste** | Estimated money lost per run or per day, from idle and underutilized time |
| **Suggestions** | Ranked list of fixes to try, such as more data-loader workers, larger batch size, mixed precision, or pinned memory |

All costs are estimates based on the GPU price per hour that the user enters.

---

## Data and measurement sources

The tool is designed to work with standard, widely available measurement sources:

- **NVIDIA management library (NVML)** for utilization, memory, power and clocks
- **PyTorch profiler** (and similar framework profilers) for step-level timing and data-loading time
- **System metrics** such as CPU usage and disk and network throughput

It does not need special access to model code beyond a small wrapper or callback. No training data is read or stored, only timing and resource measurements.

For testing, the project includes **controlled benchmark scripts** that create known problems on purpose (slow data loader, tiny batch size, frequent synchronization, idle gaps), so detection accuracy can be measured against the truth.

---

## Planned approach

### Detection

- Build a **timeline** of each training step: data wait, host-to-device copy, compute, communication, idle
- Compare GPU activity with CPU and data-loader activity to attribute each idle gap to a cause
- Estimate efficiency beyond raw utilization (for example, achieved throughput versus a reference, and model FLOPs utilization when the model's cost is known)

### Cost model

- Cost of waste = wasted GPU-hours x price per GPU-hour (user supplied)
- Reported per run, and projected per week for recurring jobs

### Metrics (for checking the tool itself)

- **Cause attribution accuracy** on controlled benchmarks with known problems
- **Waste estimate error** compared with measured speedup after the problem is fixed
- **Overhead:** how much the profiler slows the training job (target: small)
- **Stability:** repeated runs give similar reports

---

## How it works

```
training job (PyTorch, others later)
          |
          v
  lightweight collector
  (GPU metrics + step timing + data-loader timing + CPU metrics)
          |
          v
   per-step timeline
          |
          v
  bottleneck analysis -> cause per idle gap
          |
          v
  cost estimate + ranked suggestions + report
```

---

## Planned repository structure

```
gpu-leak-finder/
├── gpu_leak/
│   ├── collectors/       # NVML, framework profiler, system metrics
│   ├── timeline.py       # per-step timeline builder
│   ├── analyze.py        # bottleneck attribution
│   ├── cost.py           # cost model
│   ├── suggest.py        # ranked suggestions
│   └── report.py         # text and HTML report
├── benchmarks/
│   ├── controlled/       # scripts with known injected problems
│   └── results/          # saved outputs
├── examples/             # sample training scripts and reports
├── notebooks/            # exploration and plots
├── tests/
├── docs/
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Quickstart (planned)

```bash
git clone https://github.com/amoltathesoftwareengineer/gpu-leak-finder.git
cd gpu-leak-finder
pip install -r requirements.txt
```

Adding it to a PyTorch training script (illustrative):

```python
from gpu_leak import Monitor

monitor = Monitor(gpu_price_per_hour=2.50)

for epoch in range(num_epochs):
    for batch in train_loader:
        with monitor.step():
            loss = train_step(batch)

monitor.report()
```

Run the controlled benchmarks:

```bash
python -m benchmarks.controlled.run_all
```

---

## Results

Results will be added once the controlled benchmarks are complete.

| Injected problem | Detected | Cause attributed correctly | Estimated waste vs measured |
|---|---|---|---|
| _coming soon_ | - | - | - |

---

## Roadmap

- [ ] Build the NVML and step-timing collectors
- [ ] Build the per-step timeline
- [ ] Build controlled benchmark scripts with known problems
- [ ] Implement data-loading stall and idle-gap detection
- [ ] Add CPU-side wait and synchronization detection
- [ ] Add memory headroom analysis
- [ ] Add cost model and ranked suggestions
- [ ] Publish the first validated results
- [ ] Add multi-GPU communication analysis
- [ ] Add support for more frameworks

---

## Limitations and responsible use

- Reported cost is an **estimate.** It depends on the GPU price you enter and on the assumption that wasted time could have been used.
- Suggestions are starting points. Always re-measure after making a change.
- Profiling adds some overhead. Run it on a short, representative part of a job first.
- Very large multi-node jobs have extra causes (network, scheduling) that are outside the first version.
- This project is not affiliated with NVIDIA or any cloud provider. It is for research and education.

---

## Contributing

Contributions are welcome, especially:

- New controlled benchmark scenarios
- Support for more frameworks and GPU vendors
- Better efficiency metrics and cost models
- Bug reports and reproducibility fixes

Please open an issue to discuss before sending a large pull request.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

**Amol Tathe**
[LinkedIn](https://linkedin.com/in/amoltathesoftwareengineer) · [GitHub](https://github.com/amoltathesoftwareengineer)
