"""
Simple example demonstrating the GoldenTransformer framework.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import LayerFault, WeightCorruption
from goldentransformer.metrics import Accuracy, LatencyMetric

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create fault injector
    injector = FaultInjector(model)
    
    # Create test dataset
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?",
        "This is a test of the fault injection framework.",
        "The weather is beautiful outside.",
        "I love programming and machine learning."
    ]
    
    # Tokenize texts
    encodings = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )
    
    # Define faults
    faults = [
        # Layer-wise attention mask corruption
        LayerFault(
            layer_idx=5,
            fault_type="attention_mask",
            severity=0.3
        ),
        # Weight corruption
        WeightCorruption(
            pattern="random",
            corruption_rate=0.1
        )
    ]
    
    # Create metrics
    metrics = [
        Accuracy(),
        LatencyMetric(num_runs=3)
    ]
    
    # Create experiment runner
    experiment = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=2,
        output_dir="example_results"
    )
    
    # Run experiment
    print("Starting experiment...")
    results = experiment.run()
    
    # Print results
    print("\nBaseline Results:")
    for metric_name, metric_results in results["baseline"].items():
        print(f"\n{metric_name}:")
        for key, value in metric_results.items():
            print(f"  {key}: {value:.4f}")
    
    print("\nFault Results:")
    for i, fault_result in enumerate(results["fault_results"]):
        print(f"\nFault {i + 1}: {fault_result['fault_info']}")
        for metric_name, metric_results in fault_result["metrics"].items():
            print(f"\n{metric_name}:")
            for key, value in metric_results.items():
                print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main() 