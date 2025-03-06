import subprocess

commands = [
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime25 --tp=4 --temperatures 0.6 --n 16 --prompt_style no_thinking_r1 --chat_template chat_template.jinja --result-dir results_1/ --continue_final_message",
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime24,aime25 --tp=4 --temperatures 0.6 --n 16 --prompt_style no_thinking_r1 --chat_template chat_template.jinja --result-dir results_2/ --seed 135 --continue_final_message",
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime24,aime25 --tp=4 --temperatures 0.6 --n 16 --prompt_style no_thinking_r1 --chat_template chat_template.jinja --result-dir results_3/ --seed 690 --continue_final_message",
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime24,aime25 --tp=4 --temperatures 0.6 --n 16 --prompt_style no_thinking_r1 --chat_template chat_template.jinja --result-dir results_4/ --seed 123 --continue_final_message",
    # "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B-Instruct --evals=aime24,aime25 --tp=4 --temperatures 0.7 --n 16 --prompt_style normal --result-dir results_1/",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B-Instruct --evals=aime24,aime25 --tp=4 --temperatures 0.7 --n 16 --prompt_style normal --result-dir results_2/ --seed 135",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B-Instruct --evals=aime24,aime25 --tp=4 --temperatures 0.7 --n 16 --prompt_style normal --result-dir results_3/ --seed 690",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B-Instruct --evals=aime24,aime25 --tp=4 --temperatures 0.7 --n 16 --prompt_style normal --result-dir results_4/ --seed 123",
]

for command in commands:
    print(f"Running {command}")
    subprocess.run(command, shell=True)