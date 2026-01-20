import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
import json
import os
from tqdm import tqdm
from prompt import PROMPTS
import tiktoken
from tools.utils import read_jsonl, write_jsonl, create_if_not_exist,InstanceManager

from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

import logging
import yaml
import copy

logger=logging.getLogger(__name__)

threshold=50

def summarize_entity(entity_name, description, summary_prompt, threshold, tokenizer, use_llm):
    tokens = len(tokenizer.encode(description))
    if tokens > threshold:
        exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
        response = use_llm(exact_prompt)
        return entity_name, response
    return entity_name, description  # If no summary is needed, return the original description.


def truncate_data(working_dir, output_path):
    # relation_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/relation.jsonl"
    # relation_output_path="/cpfs04/user/zhangyaoze/workspace/trag/late_data/relation.jsonl"
    # entity_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/entity.jsonl"
    # entity_output_path="/cpfs04/user/zhangyaoze/workspace/trag/late_data/entity.jsonl"
    relation_path=f"{working_dir}/chunk/relation.jsonl"
    relation_output_path=f"{output_path}/tripple/relation.jsonl"
    entity_path=f"{working_dir}/chunk/entity.jsonl"
    entity_output_path=f"{output_path}/tripple/entity.jsonl"
    res=[]
    i=0
    with open(relation_path,"r", encoding="utf-8") as f:
        for uline in f:
                line=json.loads(uline)
                res.append(line)
                i+=1
                if i==20000:
                    break
    write_jsonl(res,relation_output_path)
    
    res=[]
    i=0
    with open(entity_path,"r", encoding="utf-8") as f:
        for uline in f:
                line=json.loads(uline)
                if "wtr20" in line[0]['source_id']:
                    res.append(line)
                    i+=1
                    if i==20000:
                        break
    write_jsonl(res,entity_output_path)

def deal_duplicate_entity(working_dir,output_path, use_llm):
    relation_path=f"{working_dir}/chunk/relation.jsonl"
    relation_output_path=f"{output_path}/duplicates/relation.jsonl"
    entity_path=f"{working_dir}/chunk/entity.jsonl"
    entity_output_path=f"{output_path}/duplicates/entity.jsonl"
    
    all_entities=[]
    all_relations=[]
    e_dic={}
    r_dic={}
    summary_prompt=PROMPTS['summary_entities']
    with open(entity_path,"r", encoding="utf-8")as f:
        for xline in f:
            line=json.loads(xline)
            entity_name=str(line[0]['entity_name']).replace("\"","")
            entity_type=line[0]['entity_type'].replace("\"","")
            description=line[0]['description'].replace("\"","")
            source_id=line[0]['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name]=dict(
                    entity_name=str(entity_name),
                    entity_type=entity_type,
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description']+=" | "+ description
                if e_dic[entity_name]['source_id']!= source_id:
                    e_dic[entity_name]['source_id']+= "|"+source_id
                    
                  
                    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    to_summarize = []
    for k,v in e_dic.items():
        v['source_id']="|".join(set(v['source_id'].split("|")))
        description=v['description']
        tokens = len(tokenizer.encode(description))
        if tokens > threshold:
            to_summarize.append((k, description))
        else:
            all_entities.append(v)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(summarize_entity, k, desc, summary_prompt, threshold, tokenizer, use_llm): k
            for k, desc in to_summarize
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing descriptions"):
            k, summarized_desc = future.result()
            e_dic[k]['description'] = summarized_desc
            all_entities.append(e_dic[k])

    save_entity=[]
    for v in all_entities:
        x = []
        x.append(v)
        save_entity.append(x)
    write_jsonl(save_entity, entity_output_path)

    with open(relation_path,"r", encoding="utf-8")as f:
        for xline in f:
            line=json.loads(xline)
            src_tgt=str(line[0]['src_id']).replace("\"","")
            tgt_src=str(line[0]['tgt_id']).replace("\"","")
            description=line[0]['description'].replace("\"","")
            weight=1
            source_id=line[0]['source_id']
            # r_dic[(src_tgt,tgt_src)]={
            #     'src_tgt':str(src_tgt),
            #     'tgt_src':str(tgt_src),
            #     'description':description,
            #     'weight':weight,
            #     'source_id':source_id
            # }
            all_relations.append(dict(
                src_tgt=src_tgt,
                tgt_src=tgt_src,
                description=description,
                weight=weight,
                source_id=source_id
            ))
            # e_dic[src_tgt]['degree']+=1
            # e_dic[tgt_src]['degree']+=1

    save_relations=[]
    for v in all_relations:
        x = []
        x.append(v)
        save_relations.append(x)
    write_jsonl(save_relations, relation_output_path)

    
def process_triple(file_path, output_path, use_llm):
    create_if_not_exist(output_path)
    with open(file_path,"r", encoding="utf-8") as f:
        entities={}
        relations=[]
        for uline in f:
            line=json.loads(uline)
            triple=line[0]['triple'].split("\t")
            doc_name=line[0]['doc_name']
            page_idx=line[0]['page_idx']
            paragraph_idx=line[0]['paragraph_idx']
            source_id = doc_name + "_" + str(page_idx) + "_" + str(paragraph_idx)
            head_entity=triple[0][1:-1]
            head_description=triple[1][1:-1]
            relation=triple[2][1:-1]
            relation_description=triple[3][1:-1]
            tail_entity=triple[4][1:-1]
            tail_description=triple[5][1:-1]
            
            if head_entity not in entities.keys():
                entities[head_entity]=dict(
                    entity_name=str(head_entity),
                    description=head_description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                entities[head_entity]['description']+=" | "+ head_description
                if entities[head_entity]['source_id']!= source_id:
                    entities[head_entity]['source_id']+= "|"+source_id
            if tail_entity not in entities.keys():
                entities[tail_entity]=dict(
                    entity_name=str(tail_entity),
                    description=tail_description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                entities[tail_entity]['description']+=" | "+ tail_description
                if entities[tail_entity]['source_id']!= source_id:
                    entities[tail_entity]['source_id']+= "|"+source_id
            relations.append(dict(
                src_tgt=head_entity,
                tgt_src=tail_entity,
                source=relation,
                description=relation_description,
                weight=1,
                source_id=source_id
            ))
    write_jsonl(relations,f"{output_path}/tripple/relation.jsonl") 
    res_entity=[]           
    tokenizer = tiktoken.get_encoding("cl100k_base")
    to_summarize = []
    summary_prompt=PROMPTS['summary_entities']
    for k,v in entities.items():
        v['source_id']="|".join(set(v['source_id'].split("|")))
        description=v['description']
        tokens = len(tokenizer.encode(description))
        if tokens > threshold:
            to_summarize.append((k, description))
        else:
            res_entity.append(v)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(summarize_entity, k, desc, summary_prompt, threshold, tokenizer, use_llm): k
            for k, desc in to_summarize
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing descriptions"):
            k, summarized_desc = future.result()
            entities[k]['description'] = summarized_desc
            res_entity.append(entities[k])

    write_jsonl(res_entity,f"{output_path}/tripple/entity.jsonl")


def main():
    ## Read configuration file
    conf_path = "config.yaml" 
    with open(conf_path, "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    logger.info(f"args:\n{args}\n")    

    llm_url = args["llm_conf"]["llm_url"] # "http://10.0.101.102" # "http://localhost" # "http://172.31.224.1" #
    llm_port = args["llm_conf"]["llm_port"] #  11434 # 1234 # 
    llm_model = args["llm_conf"]["llm_model"]  ## Task parameters

    NUM=1

    working_dir= args["task_conf"]["output_dir"]
    output_path= args["task_conf"]["output_dir"]


    instanceManager=InstanceManager(
        url=llm_url,
        ports=[llm_port for i in range(NUM)],
        gpus=[i for i in range(NUM)],
        generate_model=llm_model,
        startup_delay=30
    )
    use_llm=instanceManager.generate_text
    # truncate_data(working_dir=working_dir,output_path=output_path)
    deal_duplicate_entity(working_dir=working_dir,output_path=output_path, use_llm=use_llm)
    # file_path="create_kg/data/processed_wtr_reports-kg-test/wtr03_e_by_page_block-head_20/new_triples_wtr03_e_by_page_block-head_20_descriptions.jsonl"
    # process_triple(file_path,output_path, use_llm)





if __name__=="__main__":
    main()