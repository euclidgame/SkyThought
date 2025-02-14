import subprocess

commands = [
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=2 --output_file=results_math_dr1_thinking.txt --temperatures 0.6 --prompt_style=thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=2 --output_file=results_math_dr1_no_thinking.txt --temperatures 0.6 --prompt_style=no_thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=2 --output_file=results_math_Qwen.txt --temperatures 0.6 --prompt_style=normal --chat_template chat_template.jinja --continue_final_message False",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=livecodebench --tp=2 --output_file=results_lcb_qwen.txt --temperatures 0.6 --prompt_style=normal --chat_template chat_template.jinja --continue_final_message False",
]

for command in commands:
    print(f"Running {command}")
    subprocess.run(command, shell=True)
