import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import LayerFault, WeightCorruption
from goldentransformer.metrics import Accuracy, LatencyMetric, Perplexity
from goldentransformer.visualization.plotter import plot_results

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    injector = FaultInjector(model)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?",
        "This is a test of the fault injection framework.",
        "The weather is beautiful outside.",
        "I love programming and machine learning."
    ]
    encodings = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )
    faults = [
        WeightCorruption(
            pattern="bit_flip",
            corruption_rate=0.0000000000000000000000000000000000000000001,
            target_layers=[1,2]
        )
    ]
    metrics = [
        Perplexity(),
    ]
    experiment = ExperimentRunner(injector, faults, metrics, dataset)
    print("Starting experiment...")
    results = experiment.run()
    
    # Visualization: plot and save results
    plot_results(results, str(experiment.output_dir))
    
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