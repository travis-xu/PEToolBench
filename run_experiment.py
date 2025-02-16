from collections import defaultdict
import time
import pandas as pd
# from sentence_transformers import SentenceTransformer, util
import json
import re
from utils_xqc import standardize, standardize_category, change_name, set_seed, load_json, save_json
import random
import numpy as np
import json
import os
# import logging
from tqdm import tqdm 
# from inference.interact import *
from prompts.prompt_template import FORMAT_ENTRY_INPUT
from func_timeout import func_timeout, FunctionTimedOut
import unicodedata
from openai import OpenAI
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from peft import PeftModel


api_key = ""

def interact_openai(entries, model_name, prompt_type, results_file, lora_adapter_path=None):
    # 最大等待时间（秒）
    max_wait_time = 10

    # 加载已完成的响应
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []

    # 提取已处理的输入
    completed_entry_ids = {result['entry_id'] for result in results}

    client = OpenAI(
        api_key=api_key,
        base_url="https://api3.apifans.com/v1"
    )
    
    # 处理每个输入
    for entry in entries:
        if entry["entry_id"] in completed_entry_ids:
            # print(f"Skipping already processed entry: {entry["entry_id"]}")
            continue
        
        system_prompt = entry[prompt_type]
        user_prompt = entry["query"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                timeout=max_wait_time,  # 设置超时时间
            )
            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            results.append({
                "query": entry["query"],
                "api_call_ground_truth": entry["api_call_ground_truth"],
                "response": content,
                "entry_id": entry["entry_id"],
                "history_length": entry["history_length"]
            })

        except Exception as e:
            print(f"Error for '{entry['entry_id']}': {e}")

        # 实时写入JSON文件
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print("All responses have been processed and saved.")
    return results

eos_tokens = {
    "Meta-Llama-3-8B": "<|eot_id|>",
    "Meta-Llama-3-8B-Instruct": "<|eot_id|>",
    "llama3-lora": "<|eot_id|>",
    "Llama-2-7b-hf": "<|eot_id|>",
    "Llama-2-7b-chat-hf": "<|eot_id|>",
    "Qwen2.5-7B-Instruct": "<|endoftext|>",
    "Mistral-7B-Instruct-v0.3": "<|endoftext|>",
    # "claude": interact_other,
    "vicuna-7b-v1.5": "</s>",
}

# Define a function to interact with Hugging Face models
def interact_hf(entries, model_name, prompt_type, results_file, lora_adapter_path=None):
    eos_token = eos_tokens.get(model_name, "")
    # 加载已完成的响应
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []

    # 提取已处理的输入
    completed_entry_ids = {result['entry_id'] for result in results}
    
    model_path = "/data/models/"
    
    if model_name == "llama3-lora_sft" or model_name == "llama3-lora_dpo" or model_name == "llama3-lora_sft-dpo":
        # 基础模型路径
        base_model_path = model_path + "Meta-Llama-3-8B-Instruct"
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,  # 使用 BF16 精度
            device_map=0)
        # 加载 LoRA 适配器并合并到基础模型
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重到基础模型

        eos_token = eos_tokens.get("Meta-Llama-3-8B-Instruct", "")


    else:
        base_model_path = model_path + model_name

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,  # 使用 BF16 精度
            device_map=0)
        
        eos_token = eos_tokens.get(model_name, "")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # 将模型设置为评估模式
    model.eval()
    # 将模型移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     # model=base_model_path,
        #     model=model,
        #     tokenizer=tokenizer,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device=0,
        #     # device_map="auto",
        # )

    for entry in entries:
        if entry["entry_id"] in completed_entry_ids:
            # print(f"Skipping already processed entry: {entry["entry_id"]}")
            continue
        
        system_prompt = entry[prompt_type]
        user_prompt = entry["query"]

        if model_name == "Mistral-7B-Instruct-v0.3":
            messages = [
                {"role": "user", "content": system_prompt + "\nInstruction: " + user_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        if model_name == "vicuna-7b-v1.5":
            prompt = system_prompt + "Instruction: " + user_prompt
            prompt = f"USER: {prompt} ASSISTANT:"
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(eos_token)
        ]

        # if model_name == "llama3-lora_sft" or model_name == "llama3-lora_dpo" or model_name == "llama3-lora_sft-dpo":
        # 将输入文本编码为模型输入
        inputs = tokenizer(prompt, 
                            truncation=True,  # 自动截断
                            max_length=4000,  # 设置最大长度
                            return_tensors="pt").to(device)
        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        # 生成输出
        with torch.no_grad():  # 禁用梯度计算
            outputs = model.generate(
                **inputs,
                # input_ids=inputs["input_ids"],
                max_new_tokens=256,
                # max_length=50,  # 生成的最大长度
                num_return_sequences=1,  # 返回的序列数
                temperature=0.1,  # 控制生成多样性
                # top_p=0.9,  # 核采样参数
                eos_token_id=terminators,
            )
        # 解码生成结果
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(truncated_text):]
        # response = tokenizer.decode(outputs[0])[len(truncated_text):]
    
        # else:
        #     inputs = tokenizer(prompt, 
        #                        truncation=True,  # 自动截断
        #                        max_length=4000,  # 设置最大长度
        #                        return_tensors="pt").to(device)
            # inputs = tokenizer(prompt, 
            #                    truncation=True,  # 自动截断
            #                    max_length=4000,  # 设置最大长度
            #                    return_tensors="pt").to(device)
            # truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            # with torch.no_grad():  # 禁用梯度计算
            #     outputs = model.generate(
            #         **inputs,
            #         # input_ids=inputs["input_ids"],
            #         max_new_tokens=256,
            #         # max_length=50,  # 生成的最大长度
            #         # do_sample=True,
            #         # num_return_sequences=1,  # 返回的序列数
            #         temperature=0.1,  # 控制生成多样性
            #         # top_p=0.1,  # 核采样参数
            #         eos_token_id=terminators,
            #     )
            # response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(truncated_text):]

            # with torch.no_grad():
            #     output = pipeline(
            #         prompt,
            #         max_new_tokens=256,
            #         truncation=True,
            #         eos_token_id=terminators,
            #         do_sample=True,
            #         temperature=0.1,
            #         # top_p=0.9,
            #     )
            # response = output[0]["generated_text"][len(prompt):]

        results.append({
            "query": entry["query"],
            "api_call_ground_truth": entry["api_call_ground_truth"],
            "response": response,
            "entry_id": entry["entry_id"],
            "history_length": entry["history_length"]
        })
        
        # 实时写入JSON文件
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("All responses have been processed and saved.")
    return results


def calculate_scores(results, history_type, model_name):
    scores_tool = 0
    scores_parameters = 0
    for entry in results:
        if not entry["response"]:
            continue

        # response = json.loads(entry["response"])
        pattern = r'\{.*"tool_name":\s*".*?",\s*"parameters":\s*\{.*?\}\}'
        match = re.search(pattern, entry["response"])
        # '{"tool_name": "<Data>.<Pet Store>.<getInventory>", "parameters": "{}"}'
        # {'tool_name': '<Database>.<aaaa>.<Get API Current Version>', 'parameters': {'apiId': ''}}

        if match:
            json_str = match.group(0)
            try:
                # 将字符串解析为 JSON 对象
                response = json.loads(json_str)
                # print(response)
            except json.JSONDecodeError:
                print("Not valid JSON format")
                continue
        else:
            print(f"No valid JSON found in response for entry {entry['entry_id']}")
            continue

        # json_match = re.search(r'\{.*?\}', entry["response"])
        # if json_match:
        #     response = json.loads(json_match.group())
        # else:
        #     print(f"No valid JSON found in response for entry {entry['entry_id']}")
        #     continue
        
        if entry["api_call_ground_truth"]["tool_name"] == response["tool_name"]:
            scores_tool += 1
        if entry["api_call_ground_truth"]["parameters"] == response["parameters"]:
            scores_parameters += 1

    print("Model:", model_name)
    print(f"Tool accuracy ({history_type}):", f"{scores_tool/len(results)}")    # f"({scores_tool / len(results):.2%})"
    print(f"Parameters accuracy ({history_type}):", f"{scores_parameters/len(results)}")    # f"({scores_parameters / len(results):.2%})")

    # return scores

interact_functions = {
    "gpt-4o": interact_openai,
    "gpt-4o-mini": interact_openai,
    "gpt-3.5-turbo": interact_openai,
    # "mistral": interact_hf,
    "Meta-Llama-3-8B": interact_hf,
    "Meta-Llama-3-8B-Instruct": interact_hf,
    "llama3-lora_sft": interact_hf,
    "llama3-lora_dpo": interact_hf,
    "llama3-lora_sft-dpo": interact_hf,
    "Llama-2-7b-hf": interact_hf,
    "Llama-2-7b-chat-hf": interact_hf,
    "Qwen2.5-7B-Instruct": interact_hf,
    "Mistral-7B-Instruct-v0.3": interact_hf,
    "vicuna-7b-v1.5": interact_hf,
}

prompt_types = {
    "p": "instruction_preferred",
    "r": "instruction_ratings",
    "c": "instruction_chronological",
    "wo_history": "instruction_wo_history"
}

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--model_name", default=None, type=str, required=True, help="The base model name.")
    parser.add_argument("--history_type", default=None, type=str, required=True, help="")
    parser.add_argument("--lora_adapter_path", default=None, type=str, required=False, help="")
    # parser.add_argument("--num_epochs", default=5, type=int, required=True,
    #                     help="Train epochs.")
    args = parser.parse_args()

    # a = {"tool_name": "Airbnb Search","api_name": "Search Property by_place","parameters": {"place": 50}}
    # a = str(a)
    # b = eval(a)
    
    user_entries = load_json("user_entries_test.json")  # [:20]

    model_name = args.model_name   # gpt-4o-mini, gpt-4o, gpt-3.5-turbo, Meta-Llama-3-8B-Instruct
    # if model_name == "llama3-8b":
    #     model_name = "Meta-Llama-3-8B-Instruct"
    history_type = args.history_type    # preferred, ratings, chronological
    lora_adapter_path = args.lora_adapter_path
    # prompt_template = FORMAT_ENTRY_INPUT
    # results = []

    interact_func = interact_functions.get(model_name, lambda a,b,c,d,e: f"Model {model_name} not recognized.")
    prompt_type = prompt_types.get(history_type, f"History type {history_type} not recognized")

    # 输出文件名
    results_file = f"{history_type}_{model_name}.json"
    results_path="results/"
    
    results = interact_func(user_entries, model_name, prompt_type, results_path+results_file, lora_adapter_path)
    calculate_scores(results, history_type, model_name)
    
    save_json(results, results_file, file_path=results_path)



    # for entry in user_entries[:20]:  
    #     try:
    #         response = interact_func(model_name, messages)
    #     except FunctionTimedOut:
    #         print('task func_timeout')
    #         continue

    
    print()


    # prompt = "Explain the significance of uncertainty estimation in AI models."
    # models = ["gpt-4", "gpt-3.5-turbo", "mistral", "llama", "qwen", "claude"]

    # outputs = run_experiment(models, prompt)

    # for model, output in outputs.items():
    #     print(f"\nModel: {model}\nOutput: {output}\n")
