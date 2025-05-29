import argparse
import json
import os
import subprocess
import warnings

from .tasks import TASK_NAMES_TO_YAML


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process model path, prompt format, and evals to run."
    )
    parser.add_argument("--model", required=True, type=str, help="Path to the model.")
    parser.add_argument(
        "--evals",
        required=True,
        type=str,
        help=f"Comma-separated list of evals to run (no spaces). We currently support the following tasks {list(TASK_NAMES_TO_YAML.keys())}",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Difficulty for the dataset. Options: 'easy', 'medium', 'hard'",
    )
    parser.add_argument("--subset", type=str, help="Subset for the dataset.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0],
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples generated per problem."
    )
    parser.add_argument(
        "--result-dir", type=str, default=".", help="Directory to save result files."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for the dataset.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index for the dataset.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="normal",
        choices=["thinking_r1", "no_thinking_r1", "normal", "no_thinking_r1_2", "no_thinking_r1_3", "no_thinking_r1_4"],
        help="Prompt style for the model.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Jinja file for the chat template.",
    )
    parser.add_argument(
        "--continue_final_message",
        action="store_true",
        help="Continue the final message from the model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Max tokens for the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the model.",
    )
    parser.add_argument(
        "--budget_force",
        action="store_true",
        help="Force the budget of the model.",
    )
    parser.add_argument(
        "--online-inference",
        action="store_true",
        help="Use online inference.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=16, help="Max workers for the model."
    )
    parser.add_argument(
        "--inference", action="store_true", help="Use inference mode."
    )
    return parser.parse_args()


def extract_accuracy_from_output(output):
    # Iterate through all lines from the end to the beginning
    lines = output.splitlines()[::-1]
    for line in lines:
        try:
            # Attempt to parse a JSON object from the line
            data = json.loads(line.replace("'", '"'))
            if "acc" in data:
                return data["acc"]
        except json.JSONDecodeError:
            continue
    return None


def write_logs_to_file(logs, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(logs)
        print(f"Logs successfully written to {output_file}")
    except IOError as e:
        print(f"Failed to write logs to file {output_file}: {e}")


def main():
    args = parse_arguments()
    # Extract the arguments
    model_path = args.model
    evals = args.evals.split(",")
    tp = args.tp
    temperatures = [str(t) for t in args.temperatures]

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_and_check.py"
    )

    # Run the Python command for each eval and collect logs
    for eval_name in evals:
        eval_name = eval_name.lower()
        assert (
            eval_name in TASK_NAMES_TO_YAML.keys()
        ), f"Task {eval_name} not found, should be one of {TASK_NAMES_TO_YAML.keys()}"
        command = [
            "python",
            script_path,
            "--model",
            model_path,
            "--task",
            eval_name,
            "--tp",
            str(tp),
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--prompt_style",
            args.prompt_style,
            "--max_tokens",
            str(args.max_tokens),
            "--max-workers",
            str(args.max_workers),
            "--temperatures",
        ]
        command.extend(temperatures)  # Add temperatures as separate arguments
        command.extend(
            [
                "--n",
                str(args.n),
                "--result-dir",
                args.result_dir,
            ]
        )
        if args.inference:
            command.append("--inference")
        if args.budget_force:
            command.append("--budget_force")
        if args.online_inference:
            command.append("--online-inference")
        if args.seed:
            command.append("--seed")
            command.append(str(args.seed))
        if args.continue_final_message:
            command.append("--continue_final_message")

        if args.chat_template:
            command.append("--chat_template")
            command.append(args.chat_template)
        if args.difficulty:
            command.append("--difficulty")
            command.append(args.difficulty)

        print(f"Running eval {eval_name} with command {command}")
        try:
            subprocess.run(command, check=True)
            # TODO (sumanthrh): cleanup code here but provide a way to extract accuracy from all the runs
            # ideally, we should invoke functions from inference_and_check and juust have then return
            # metrics.
            # with subprocess.Popen(
            #     command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            # ) as proc:
            #     output_lines = []
            #     for line in proc.stdout:
            #         print(line, end="")  # Stream output to the console
            #         output_lines.append(line)
            #         all_logs += line
            #     proc.wait()
            #     if proc.returncode != 0:
            #         raise subprocess.CalledProcessError(proc.returncode, command)

            #     # Capture output for post-processing
            #     output = "".join(output_lines)
            #     accuracy = extract_accuracy_from_output(output)
            #     results[eval_name] = accuracy

        except subprocess.CalledProcessError as e:
            error_message = f"Error occurred while running eval {eval_name}: {e}\n"
            print(error_message)

    print(f"Evals for tasks {args.evals} ran successfully.")


if __name__ == "__main__":
    main()
