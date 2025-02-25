import subprocess

commands = [
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime25,aime24 --tp=8 --result-dir=results_aime_dr1_thinking --temperatures 0.6 --prompt_style=thinking --chat_template chat_template.jinja --continue_final_message True",
    # "python -m skythought_evals.eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=aime25,aime24 --tp=8 --result-dir=results_aime_dr1_no_thinking --temperatures 0.6 --prompt_style=no_thinking --chat_template chat_template.jinja --continue_final_message True",
    # "CUDA_VISIBLE_DEVICES=3,4,6,7 python -m skythought_evals.eval --model xiaomama2002/s1.1-direct-32B --evals=aime25,aime24,amc23,gpqa_diamond,olympiadbench_math_en,math500 --tp=4 --prompt_style no_thinking --result-dir s1-math --temperatures 0.6",
    # "CUDA_VISIBLE_DEVICES=3,4,6,7 python -m skythought_evals.eval --model xiaomama2002/s1.1-direct-32B --evals=aime25,aime24,amc23,gpqa_diamond,olympiadbench_math_en,math500 --tp=4 --prompt_style normal --result-dir s1-math --temperatures 0.6",
    # "CUDA_VISIBLE_DEVICES=3,4,6,7 python -m skythought_evals.eval --model simplescaling/s1.1-32B --evals=aime25,aime24,amc23,gpqa_diamond,olympiadbench_math_en,math500 --tp=4 --prompt_style no_thinking --result-dir s1-math-new --temperatures 0.6 --continue_final_message",
    "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m skythought_evals.eval --model Qwen/Qwen2.5-32B --evals=aime25,aime24,amc23,gpqa_diamond,olympiadbench_math_en,math500,livecodebench --tp=4 --prompt_style normal --result-dir qwen-32b-base-1 --temperatures 0.6",
]

for command in commands:
    print(f"Running {command}")
    subprocess.run(command, shell=True)