import subprocess

commands = [
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime25,aime24 --tp=8 --result-dir=results_aime_dr1_thinking --temperatures 0.6 --prompt_style=thinking --chat_template chat_template.jinja --continue_final_message True",
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime25,aime24 --tp=8 --result-dir=results_aime_dr1_no_thinking --temperatures 0.6 --prompt_style=no_thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=aime25,aime24 --tp=8 --result-dir=results_aime_qwen --temperatures 0.6 --prompt_style=normal --chat_template chat_template.jinja --continue_final_message False",
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=livecodebench --tp=8 --result-dir=results_lcb_dr1_thinking --temperatures 0.6 --prompt_style=thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=livecodebench --tp=8 --result-dir=results_lcb_dr1_no_thinking --temperatures 0.6 --prompt_style=no_thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=livecodebench --tp=8 --result-dir=results_lcb_qwen --temperatures 0.6 --prompt_style=normal --chat_template chat_template.jinja --continue_final_message False",
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=8 --result-dir=results_math_dr1_thinking --temperatures 0.6 --prompt_style=thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=8 --result-dir=results_math_dr1_no_thinking --temperatures 0.6 --prompt_style=no_thinking --chat_template chat_template.jinja --continue_final_message True",
    "python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=gpqa_diamond,amc23,olympiadbench_math_en --tp=8 --result-dir=results_math_Qwen --temperatures 0.6 --prompt_style=normal --chat_template chat_template.jinja --continue_final_message False",
]

for command in commands:
    print(f"Running {command}")
    subprocess.run(command, shell=True)