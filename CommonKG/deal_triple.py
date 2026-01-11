import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
import json
import os
from tqdm import tqdm
from prompt import PROMPTS
import tiktoken
import sys 
from tools.utils import read_jsonl, write_jsonl, create_if_not_exist,InstanceManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import logging
import yaml

logger=logging.getLogger(__name__)


threshold=50

def summarize_entity(entity_name, description, summary_prompt, threshold, tokenizer, use_llm):
    tokens = len(tokenizer.encode(description))
    if tokens > threshold:
        exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
        response = use_llm(exact_prompt)
        return entity_name, response
    return entity_name, description  # If no summary is needed, return the original description.



def process_triple(file_path,output_path, use_llm):
    create_if_not_exist(output_path)
    triple_path=os.path.join(file_path,f"new_triples_{os.path.basename(file_path)}_descriptions.jsonl")
    
    with open(triple_path,"r", encoding="utf-8") as f:
        entities={}
        relations=[]
        for uline in f:
            line=json.loads(uline)
            triple=line['triple'].split("\t")
            doc_name=line['doc_name']
            source_id=line['source_id']
            head_entity=triple[0][1:-1]
            head_description=triple[1][1:-1]
            head_type=triple[2][1:-1]
            relation=triple[3][1:-1]
            relation_description=triple[4][1:-1]
            tail_entity=triple[5][1:-1]
            tail_description=triple[6][1:-1]
            tail_type=triple[7][1:-1]
            
            if head_entity not in entities.keys():
                entities[head_entity]=dict(
                    entity_name=str(head_entity),
                    description=head_description,
                    type=head_type,
                    source_id=source_id,
                    doc_name=doc_name,
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
                    type=tail_type,
                    source_id=source_id,
                    doc_name=doc_name,
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
    write_jsonl(relations,f"{output_path}/relation.jsonl") 
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

    write_jsonl(res_entity,f"{output_path}/entity.jsonl")


def main():
    ## Read configuration file
    conf_path = "config.yaml"  # "CommonKG/config/create_kg_conf_test.yaml"
    with open(conf_path, "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    logger.info(f"args:\n{args}\n")    

    llm_url = args["llm_conf"]["llm_url"] # "http://10.0.101.102" # "http://localhost" # "http://172.31.224.1" #
    llm_port = args["llm_conf"]["llm_port"] #  11434 # 1234 # 
    llm_model = args["llm_conf"]["llm_model"]  ## Task parameters

    num=1
    instanceManager=InstanceManager(
        url=llm_url,
        ports=[llm_port for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model=llm_model,
        startup_delay=30
    )
    use_llm = instanceManager.generate_text

    file_path= args["task_conf"]["output_dir"]  # Path(Path(__file__).parent.parent.__str__(), "ckg_data/mix_chunk3/mix_chunk3")
    output_path= args["task_conf"]["output_dir"]  # Path(Path(__file__).parent.parent.__str__(), "ckg_data/mix_chunk3/mix_chunk3")
    process_triple(file_path.__str__(),output_path.__str__(), use_llm)

if __name__=="__main__":
    main()