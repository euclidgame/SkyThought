#!/usr/bin/env python3
"""
Script to check the accuracy of LiveCodeBench inference results.

Usage:
    python check_livecodebench_accuracy.py --results_file path/to/results.json [--start 0] [--end 80] [--max_workers 32]

This script loads the LiveCodeBench dataset and your inference results, then uses the 
LiveCodeBenchTaskHandler to check the correctness of each response.
"""

import argparse
import json
import os
import sys
import time
import logging
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Dict, List

from tqdm import tqdm

# Add the skythought directory to the path
sys.path.append('skythought')

from skythought_evals.tasks import TASK_NAMES_TO_YAML, TaskConfig
from skythought_evals.tasks.livecodebench.livecodebench_handler import LiveCodeBenchTaskHandler
from skythought_evals.util.metrics import pass_at_k


def load_inference_results(results_file: str) -> List[Dict]:
    """Load the inference results from a JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Check accuracy of LiveCodeBench inference results")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the JSON file containing inference results"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for dataset (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=80,
        help="End index for dataset (default: 80)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=128,
        help="Maximum number of worker processes (default: 32)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output file to save detailed results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Process only first 10 problems for quick testing"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Validate input file
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)

    print(f"Loading LiveCodeBench task configuration...")

    # Initialize the task handler
    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML['livecodebench'])
    handler = LiveCodeBenchTaskHandler(task_config)

    # Adjust end for quick test
    if args.quick_test:
        args.end = min(args.end, args.start + 10)
        print(f"Quick test mode: processing problems {args.start} to {args.end}")

    print(f"Loading dataset from index {args.start} to {args.end}...")

    # Load the dataset
    dataset = handler.load_and_filter_dataset(
        start=args.start,
        end=args.end,
        split=None,
        subset=None,
        difficulty=None,
        args=None
    )

    print(f"Loaded {len(dataset)} problems from dataset")

    # Load inference results
    print(f"Loading inference results from {args.results_file}...")
    inference_results = load_inference_results(args.results_file)

    print(f"Loaded {len(inference_results)} inference results")

    # Validate that we have matching numbers
    if len(dataset) != len(inference_results):
        print(f"Warning: Dataset has {len(dataset)} problems but results have {len(inference_results)} entries")
        min_len = min(len(dataset), len(inference_results))
        print(f"Will process first {min_len} entries")
        dataset = dataset.iloc[:min_len]
        inference_results = inference_results[:min_len]

    # Convert dataset to list of dicts for easier processing
    remaining_data = []
    for _, problem_row in dataset.iterrows():
        remaining_data.append(problem_row.to_dict())

    print(f"Processing {len(remaining_data)} problems...")

    # Prepare to collect results
    total_correct = 0
    total_finish = 0
    all_results = []
    problem_results = {}

    start_time = time.time()
    logging.info("Starting correctness check...")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        logging.info("ProcessPoolExecutor initialized")

        future_to_task = {}          # Future -> (problem_idx, response_idx, response_text)
        future_submit_time = {}      # Future -> submission timestamp
        future_run_time = {}         # Future -> first time .running() was True
        num_futures = 0

        # Submit all tasks
        for idx, result_entry in enumerate(inference_results):
            problem = remaining_data[idx]

            responses = result_entry.get('response', [])
            if not isinstance(responses, list):
                responses = [responses]

            for response_idx, response in enumerate(responses):
                future = executor.submit(handler.update_results, problem, response)
                future_to_task[future] = (idx, response_idx, response)
                future_submit_time[future] = time.time()
                future_run_time[future] = None
                num_futures += 1

        # Initialize progress bar
        progress = tqdm(
            total=num_futures,
            desc="Processing Generations",
            ncols=80,
            mininterval=0.1
        )

        pending = set(future_to_task.keys())

        while pending:
            # Wait up to 2 seconds for any future to complete
            done, not_done = wait(pending, timeout=2, return_when=FIRST_COMPLETED)

            # Process newly completed futures
            for future in done:
                idx, response_idx, response = future_to_task[future]
                try:
                    response_entry: dict = future.result()
                except Exception as e:
                    response_entry = {
                        "content": response,
                        "correctness": False,
                        "reason": f"Task failed with error: {str(e)}"
                    }

                total_correct += int(response_entry.get("correctness", False))
                total_finish += 1

                if idx not in problem_results:
                    problem_results[idx] = []
                problem_results[idx].append(response_entry.get("correctness", False))

                prob = remaining_data[idx]
                truncated_resp = (
                    response_entry["content"][:200] + "..."
                    if response_entry["content"] and len(response_entry["content"]) > 200
                    else response_entry["content"] if response_entry["content"] else "No response"
                )
                all_results.append({
                    "problem_idx": idx,
                    "response_idx": response_idx,
                    "problem_id": prob.get(handler.question_key, f"problem_{idx}"),
                    "correctness": response_entry.get("correctness", False),
                    "reason": response_entry.get("reason", ""),
                    "response": truncated_resp
                })

                progress.update(1)
                pending.remove(future)

            # Check running status and timeouts for not-yet-done futures
            now = time.time()
            for future in list(not_done):
                # Record first time it transitions to running
                if future.running() and future_run_time[future] is None:
                    future_run_time[future] = now

                run_start = future_run_time[future]
                if run_start is not None and (now - run_start) > 120:
                    idx, response_idx, response = future_to_task[future]
                    logging.error(f"Task timed out for problem {idx}, response {response_idx}")

                    response_entry = {
                        "content": response,
                        "correctness": False,
                        "reason": "Task timed out after 60 seconds of execution"
                    }
                    total_finish += 1
                    if idx not in problem_results:
                        problem_results[idx] = []
                    problem_results[idx].append(False)

                    prob = remaining_data[idx]
                    truncated_resp = (
                        response[:200] + "..."
                        if response and len(response) > 200
                        else response if response else "No response"
                    )
                    all_results.append({
                        "problem_idx": idx,
                        "response_idx": response_idx,
                        "problem_id": prob.get(handler.question_key, f"problem_{idx}"),
                        "correctness": False,
                        "reason": "Task timed out after 60 seconds of execution",
                        "response": truncated_resp
                    })

                    future.cancel()
                    progress.update(1)
                    pending.remove(future)

            # If no futures moved from pending to done/timeout, loop again

        progress.close()

    # All futures are now either done or cancelled
    print(f"Final acc: {total_correct}/{total_finish}")

    # Calculate overall accuracy
    accuracy = total_correct / total_finish if total_finish > 0 else 0

    print(f"\n{'='*60}")
    print("LIVECODEBENCH ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"Total responses processed: {total_finish}")
    print(f"Correct responses: {total_correct}")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Calculate pass@k metrics if we have multiple responses per problem
    responses_per_problem = len(inference_results[0].get("response", [])) if inference_results else 1
    if responses_per_problem > 1:
        print(f"\nPass@k Metrics (with {responses_per_problem} responses per problem):")

        pass_at_k_data = {}
        for prob_idx, results in problem_results.items():
            pass_at_k_data[str(prob_idx)] = results

        k_values = (
            [1, 5, 10, 20, 32, 64]
            if responses_per_problem >= 64
            else [1, 5, 10, min(responses_per_problem, 20)]
        )
        k_values = [k for k in k_values if k <= responses_per_problem]

        pass_at_k_results = pass_at_k(responses_per_problem, {0.0: pass_at_k_data})
        print(json.dumps({"pass_at_k": pass_at_k_results}))

    problem_level_correct = sum(1 for results in problem_results.values() if any(results))
    problem_level_accuracy = problem_level_correct / len(problem_results) if problem_results else 0
    print(f"\nProblem-level accuracy (at least one correct): {problem_level_accuracy:.4f} ({problem_level_accuracy*100:.2f}%)")

    elapsed_time = time.time() - start_time
    print(f"\nPerformance:")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average time per response: {elapsed_time / total_finish:.2f} seconds")

    if args.output_file:
        output_data = {
            "summary": {
                "total_responses": total_finish,
                "correct_responses": total_correct,
                "overall_accuracy": accuracy,
                "problem_level_accuracy": problem_level_accuracy,
                "responses_per_problem": responses_per_problem,
                "processing_time": elapsed_time,
            },
            "detailed_results": all_results,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output_file}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()