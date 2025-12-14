import json
import re
import subprocess
import threading
import time
from snownlp import SnowNLP
import os
import numpy as np
from openai import OpenAI
import requests
import tiktoken
import yaml
tokenizer = tiktoken.get_encoding("cl100k_base")
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0
def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text
def create_if_not_exist(path):
    if not os.path.exists(path):  # If the directory does not exist, create it recursively.
        os.makedirs(path, exist_ok=True)

def dicts_almost_equal(dict1, dict2, tolerance=1e-6):
    # Allow floating error when comparing dictionaries
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        # If the value is a list, compare each element one by one.
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(value1, value2):
                if isinstance(v1, float) and isinstance(v2, float):
                    if abs(v1 - v2) > tolerance:  # floating tolerance
                        return False
                elif v1 != v2:
                    return False
        # If the value is a floating-point number, compare directly.
        elif isinstance(value1, float) and isinstance(value2, float):
            if abs(value1 - value2) > tolerance:  # floating tolerance
                return False
        # Other types of values ​​are compared directly.
        elif value1 != value2:
            return False
    return True


def custom_lower_fast(s):
    """Lowercase conversion compatible with both Chinese and English"""
    return s.lower() if s.isascii() else s  # Chinese remains unchanged


def is_word_boundary(text, start, end):
    """Adaptive Chinese and English word boundary detection"""
    # Determine if the text contains Chinese characters (including extended CJK characters).
    has_chinese = re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]", text)

    if has_chinese:
        # Chinese mode: Use snowNLP word segmentation to detect word boundaries
        snow = SnowNLP(text)
        words = snow.words
        current_pos = 0
        boundaries = set()

        # Build word boundary set
        for word in words:
            boundaries.add(current_pos)  # word starting position
            boundaries.add(current_pos + len(word))  # word end position
            current_pos += len(word)

        # Check whether the input position is on a word segmentation boundary
        return start in boundaries or end in boundaries
    else:
        # English Mode: Detecting word boundaries using regular expressions
        word_chars = r"\w"  # Only letters, numbers, and underscores

        # pre-character check
        prev_is_word = False
        if start > 0:
            prev_char = text[start - 1]
            prev_is_word = re.match(f"[{word_chars}]", prev_char, re.UNICODE)

        # post character check
        next_is_word = False
        if end < len(text):
            next_char = text[end]
            next_is_word = re.match(f"[{word_chars}]", next_char, re.UNICODE)

        return not prev_is_word and not next_is_word


def read_jsonl(file_path):
    """
    Read a jsonl file and return a list containing JSON objects for each line.
    
    :param file_path: Path to .jsonl file
    :return: List containing JSON objects for each row
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove blank lines
                if line.strip():
                    json_obj = json.loads(line.strip())  # Parse the JSON object for each row
                    data.append(json_obj)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def write_jsonl(data, path, mode="a",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
def write_jsonl_force(data, path, mode="w+",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
def check_test(entities):## Initially used to detect whether there are boundary errors
    e_l=[]
    for layer in entities:
        temp_e=[]
        if type(layer) != list:
            temp_e.append(layer['entity_name'])
            e_l.append(temp_e)
            continue
        for item in layer:
            temp_e.append(item['entity_name'])
        e_l.append(temp_e)
        
    for index,layer in enumerate(entities):
        if type(layer) != list or len(layer) == 1:
            break
        for item in layer:
            if item['parent'] not in e_l[index+1]:
                print(item['entity_name'],item['parent'])
                
class InstanceManager:
    def __init__(self, url,ports, gpus, generate_model,startup_delay=30):
        self.ports = ports
        self.gpus = gpus
        self.base_url=url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # for polling strategy
        self.generate_model=generate_model
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0
        for port, gpu in zip(self.ports, self.gpus):
            self.instances.append({"port": port, "load": 0})
    def reset_token_cost(self):
        """Reset total token consumption and number of API calls"""
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0
    def get_tokens_cosumption(self):
        
        return self.TOTAL_TOKEN_COST, self.TOTAL_API_CALL_COST
        
        # time.sleep(startup_delay)  # Wait for all instances to start
    

    def get_available_instance(self):
        """Use polling strategy to obtain an available instance"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance["port"]  # return port

    def generate_text(self, prompt,system_prompt=None, history_messages=[], **kwargs):
        """Send request to selected instance"""
        port = self.get_available_instance()
        base_url = f"{self.base_url}:{port}/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------
       
        if len(history_messages)>1:
            history_messages[0]['content'] = truncate_text(history_messages[0]['content'], max_tokens=3000)
            history_messages[1]['content'] = truncate_text(history_messages[1]['content'], max_tokens=25000)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            cur_token_cost = len(tokenizer.encode(messages[0]['content']))
            if cur_token_cost>31000:
                cur_token_cost = 31000
                messages[0]['content'] = truncate_text(messages[0]['content'], max_tokens=31000)
            
            # logging api call cost
            self.TOTAL_API_CALL_COST += 1
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": self.generate_model,
                    "messages": messages,
                    **kwargs,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
                #timeout=300
        )
            response.raise_for_status()
            res=json.loads(response.content)
            self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
            response_message = res["choices"][0]["message"]['content']#Post-process the results
        except Exception as e:
            print(f"Retry for Error: {e}")
            response = ""    
            response_message=""
                
        return response_message
    
    async def generate_text_asy(self, prompt,system_prompt=None, history_messages=[], **kwargs):
        """Send request to the selected instance"""
        port = self.get_available_instance()
        base_url = f"{self.base_url}:{port}/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------
       
        if len(history_messages)>1:
            history_messages[0]['content'] = truncate_text(history_messages[0]['content'], max_tokens=3000)
            history_messages[1]['content'] = truncate_text(history_messages[1]['content'], max_tokens=25000)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            # encode_text = messages[0]['content']
            # print(encode_text)
            # tokens = tokenizer.encode(encode_text)
            # cur_token_cost = len(tokens)
            # if cur_token_cost>31000:
            #     cur_token_cost = 31000
            messages[0]['content'] = truncate_text(messages[0]['content'], max_tokens=31000)
            
            # logging api call cost
            self.TOTAL_API_CALL_COST += 1
            response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": self.generate_model,
                "messages": messages,
                **kwargs,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=600
        )
            response.raise_for_status()
            res=json.loads(response.content)
            self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
            response_message = res["choices"][0]["message"]['content'] #Post-processing of results
        except Exception as e:
            print(f"Retry for Error: {e}")
            response = ""    
            response_message=""
        
        return response_message

