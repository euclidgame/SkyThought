# Get the process reward score of an answer
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import torch
from tqdm import tqdm
import argparse
import json
import time
import os
import sys
import re

from vllm import SamplingParams, LLM

def process_reward_score(prompts_and_answers, llm, tokenizer, candidate_tokens):
    all_conversations = []
    answer_lengths = []  # To keep track of steps for each answer

    answers_per_problem = len(prompts_and_answers[0][1])  # Number of answers per problem

    # Prepare all conversations at once
    for prompt, answers in prompts_and_answers:
        problem_lengths = []  # Track lengths for each answer in this problem
        for ans in answers:
            ans_list = list(map(str.strip, ans.split("\n\n")))
            step_conversations = []
            
            # Create conversation for each step
            current_conversation = []
            for k, step in enumerate(ans_list):
                if k == 0:
                    text = prompt + " " + step
                else:
                    text = step
                current_conversation.append({"content": text, "role": "user"})
                current_conversation.append({"content": "+", "role": "assistant"})
                # Make a copy of the conversation at each step
            # step_conversations.append(current_conversation)
            
            # Add all step conversations for this answer
            all_conversations.append(current_conversation)
        #     problem_lengths.append(len(step_conversations))
        # answer_lengths.append(problem_lengths)

    # Run inference using vLLM in batch
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=10
    )

    all_outputs = llm.chat(messages=all_conversations, sampling_params=sampling_params)

    # Process results
    current_idx = 0
    all_step_scores = []
    
    # Process each problem
    for i in range(len(prompts_and_answers)):
        problem_scores = []
        answer_outputs = all_outputs[current_idx:current_idx + answers_per_problem]
        for output in answer_outputs:
            print("Output: ", output.outputs[0].text)
            if not output.outputs[0].logprobs:
                score = -float('inf')
            else:
                logprobs = output.outputs[0].logprobs[0]
                token_probs = torch.tensor([logprobs.get(token, float('-inf')).logprob for token in candidate_tokens])
                score = token_probs.softmax(dim=-1)[0].item()
            problem_scores.append(score)
        all_step_scores.append(problem_scores)
        current_idx += answers_per_problem
    return all_step_scores

    # for problem_lengths in answer_lengths:
    #     problem_scores = []
        
    #     # Process each answer in the problem
    #     for length in problem_lengths:
    #         answer_outputs = all_outputs[current_idx:current_idx + length]
    #         single_step_scores = []
            
    #         # Process each step in the answer
    #         for output in answer_outputs:
    #             logprobs = output.outputs[0].logprobs[0]
    #             if not isinstance(logprobs, dict):
    #                 print(f"Warning: logprobs is not a dict: {type(logprobs)}")
    #                 continue

    #             token_probs = torch.tensor([logprobs.get(str(token), float('-inf')) for token in candidate_tokens])
    #             score = token_probs.softmax(dim=-1)[0].item()
    #             single_step_scores.append(score)
            
    #         problem_scores.append(single_step_scores)
    #         current_idx += length
        
    #     all_step_scores.append(problem_scores)
    
    # return all_step_scores

import json

def process_and_select_answers(json_file_path, llm, tokenizer, candidate_tokens):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Prepare all prompts and answers
    prompts_and_answers = []
    problem_ids = []
    answers_list = []
    
    for problem_id, problem_data in data.items():
        prompt = problem_data.get('prompt', '')
        answers = problem_data.get('responses', {}).get('0.6', [])
        
        if not answers:
            continue
            
        answer_texts = [ans.get('content', '') for ans in answers]
        prompts_and_answers.append((prompt, answer_texts))
        problem_ids.append(problem_id)
        answers_list.append(answers)

    # Get reward scores for all problems at once
    all_step_scores = process_reward_score(prompts_and_answers, llm, tokenizer, candidate_tokens)
    
    # Process results
    results = {}
    for problem_id, answers, step_scores in zip(problem_ids, answers_list, all_step_scores):
        # Add debug print
        # print(f"Processing problem {problem_id}")
        # print(f"Step scores length: {len(step_scores)}")
        
        # Handle empty step_scores
        if not step_scores:
            print(f"Warning: No scores available for problem {problem_id}")
            continue

        # average_scores = [sum(score_list)/len(score_list) if score_list else -float('inf') for score_list in step_scores]
            
        # last_scores = [score_list[-1] if score_list else -float('inf') for score_list in step_scores]
        
        # Check if last_scores is empty
        # if not last_scores:
        #     print(f"Warning: No valid last scores for problem {problem_id}")
        #     continue
            
        best_answer_idx = int(np.argmax(step_scores))

        # print(f"Best score: {step_scores[best_answer_idx]}")

        print(f"Best answer index: {best_answer_idx}")
        
        selected_answer = answers[best_answer_idx]
        is_correct = selected_answer.get('correctness', False)
        
        results[problem_id] = {
            'selected_answer_index': best_answer_idx,
            'selected_answer': selected_answer.get('content', ''),
            'is_correct': is_correct,
            'scores': step_scores[best_answer_idx],
            'all_scores': step_scores
        }
    
    # Save results
    output_file = json_file_path.replace('.json', '_processed.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

    # Calculate accuracy
    correct_count = sum(1 for item in results.values() if item['is_correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Overall accuracy: {correct_count}/{total_count} = {accuracy:.2%}")
    
    return results

if __name__ == "__main__":

    model = LLM(
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        dtype=torch.bfloat16,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=32768,
        enable_prefix_caching=True
    )

    tokenizer = AutoTokenizer.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")

    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]
    # print(candidate_tokens)

    process_and_select_answers('results_1/aime25/no_thinking_r1/temp_0.6/Pass_at_16/metrics/DeepSeek-R1-Distill-Qwen-32B_s0_e-1.json', model, tokenizer, candidate_tokens)