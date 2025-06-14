# GoldenTransformer: LLM Fault Injection Framework

GoldenTransformer is a research framework for analyzing the resiliency of Large Language Models (LLMs) and other neural networks through systematic fault injection and accuracy/latency measurement. It enables you to inject faults, run controlled experiments, and visualize the impact on model performance.

## Features

- **Flexible Fault Injection**: Inject faults at the layer, weight, or activation level
- **Comprehensive Metrics**: Measure accuracy, latency, and more
- **Experiment Management**: Reproducible, logged, and saved experiments
- **Visualization**: Built-in support for result visualization

## Installation

```bash
pip install -r requirements.txt
```

## End-to-End Example

Below is a minimal example using a HuggingFace model and the IMDB dataset. This will run a baseline and two types of faults, and print results.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from goldentransformer.core.experiment_runner import ExperimentRunner
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.layer_fault import LayerFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.latency import LatencyMetric

def prepare_dataset(tokenizer, max_length=128):
    dataset = load_dataset("imdb", split="test[:100]")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
    labels = torch.tensor(dataset["label"])
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
dataset = prepare_dataset(tokenizer)

injector = FaultInjector(model)
faults = [
    LayerFault(layer_idx=0, severity=0.2),  # May not work for all models, see below
    WeightCorruption(corruption_rate=0.1)
]
metrics = [Accuracy(), LatencyMetric(num_runs=3)]

runner = ExperimentRunner(
    injector=injector,
    faults=faults,
    metrics=metrics,
    dataset=dataset,
    batch_size=16,
    num_samples=50,
    device=device
)
results = runner.run()
print(results)
```

## Model and Fault Compatibility

- **WeightCorruption**: Works with most PyTorch models (including HuggingFace transformers)
- **LayerFault/ActivationFault**: Designed for GPT-style models with a `transformer.h` or `transformer` attribute as a `ModuleList`. These may not work out-of-the-box with models like DistilBERT or BERT. You may need to adapt the fault class for your model architecture.

## Visualizing Results

After running an experiment, results are saved as a `results.json` file in a timestamped directory (e.g., `experiment_results_YYYYMMDD_HHMMSS/results.json`).

Here's how to visualize accuracy and latency from a results file:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

with open('experiment_results_20250614_170427/results.json') as f:
    results = json.load(f)

# Collect data
labels = ['Baseline'] + [f["fault_info"] for f in results["fault_results"]]
accuracies = [results["baseline"]["Accuracy"]["Accuracy"]] + [f["metrics"]["Accuracy"]["Accuracy"] for f in results["fault_results"]]
latencies = [results["baseline"]["LatencyMetric"]["avg_latency_ms"]] + [f["metrics"]["LatencyMetric"]["avg_latency_ms"] for f in results["fault_results"]]

x = np.arange(len(labels))
fig, ax1 = plt.subplots(figsize=(10,5))

color = 'tab:blue'
ax1.set_xlabel('Experiment')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(x-0.2, accuracies, width=0.4, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Avg Latency (ms)', color=color)
ax2.bar(x+0.2, latencies, width=0.4, color=color, label='Latency')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Fault Injection Impact on Accuracy and Latency')
plt.show()
```

## Example Output

```
Experiment Results:
==================

Baseline Results:
Accuracy: {'Accuracy': 0.0}
LatencyMetric: {'latency_ms': 3.79, 'confidence': 0.72, 'avg_latency_ms': 3.56, 'min_latency_ms': 3.21, 'max_latency_ms': 3.79, 'avg_confidence': 0.80}

Fault Results:

Fault: LayerFault(layers=[0], type=attention_mask, severity=0.2)
Accuracy: {'Accuracy': 0.0}
LatencyMetric: {'latency_ms': 3.49, 'confidence': 0.69, 'avg_latency_ms': 3.52, 'min_latency_ms': 3.34, 'max_latency_ms': 3.64, 'avg_confidence': 0.67}

Fault: WeightCorruption(severity=0.1)
Accuracy: {'Accuracy': 0.0}
LatencyMetric: {'latency_ms': 3.63, 'confidence': 0.60, 'avg_latency_ms': 3.47, 'min_latency_ms': 3.09, 'max_latency_ms': 3.63, 'avg_confidence': 0.76}
```

## Troubleshooting

- **LayerFault/ActivationFault errors**: If you see errors like `Model must have transformer.h or transformer as ModuleList`, your model architecture is not compatible with these faults. Use `WeightCorruption` or adapt the fault class for your model.
- **Accuracy is always 0**: Check that your dataset and labels are compatible with your model's output.
- **CUDA out of memory**: Lower the batch size or use a smaller model/dataset.
- **Import errors**: Make sure you installed all dependencies from `requirements.txt`.

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use GoldenTransformer in your research, please cite:

```bibtex
@software{goldentransformer2024,
  author = {Luke Howard},
  title = {GoldenTransformer: LLM Fault Injection Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/FuzzyNum/goldentransformer}
}
``` 