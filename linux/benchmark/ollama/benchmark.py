import requests
import sys

def run_generation(model_name, prompt):
    url = 'http://localhost:11434/api/generate'

    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def main():
    if len(sys.argv) > 2:
        print("Usage: python script.py [model_name]")
        sys.exit(1)

    model_name = sys.argv[1] if len(sys.argv) == 2 else (print("Usage: python benchmark.py model_name") or sys.exit(1))    

    # Define prompts for each run
    prompts = [
        "Write me a C++ code that solves the fibonacci without recursion",
        "Explain the concept of quantum entanglement in simple terms",
        "Compose a poem inspired by the idea of artificial intelligence"
    ]

    # Run the generation three times with different prompts and capture results
    results = []
    for i in range(3):
        prompt = prompts[i]
        results.append(run_generation(model_name, prompt))

    # Calculate average results
    avg_eval_count = sum(int(result.get('eval_count', 0)) for result in results) / len(results)
    avg_eval_duration = sum(float(result.get('eval_duration', 0)) / (10**6) for result in results) / len(results)

    # Print the average performance metrics
    print("\nAverage Results:")
    print(f"Average eval_count: {avg_eval_count:.2f}")
    print(f"Average eval_duration: {avg_eval_duration:.2f} ms")

    if avg_eval_duration != 0:
        avg_performance = avg_eval_count / avg_eval_duration * 1000
        print(f"Average Performance: {avg_performance:.2f}(tokens/s)")
    else:
        print("Average eval_duration is zero. Cannot calculate average performance.")

if __name__ == "__main__":
    main()
