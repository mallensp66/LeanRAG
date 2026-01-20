import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
import json
import time
from tqdm import tqdm
from corpus import Corpus
import os
import yaml
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed 
import shutil
from tools.logger_factory import setup_logger
from tools.utils import read_jsonl, write_jsonl, create_if_not_exist
from llm_infer import LLM_Processor
from triple import Triple

logger = setup_logger("create_KG")


def write_txt(path: str, data, mode="a"):
    with open(path, mode=mode, encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, list) or isinstance(data, set):
            for line in data:
                f.write(line+"\n")

def read_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def process_llm_batch(item_batch, llm_processer, ref_kg_path):
    """
    Processing LLM requests in a single batch
    """
    doc_name, source_id, text, match_words = (
        item_batch["doc_name"],
        item_batch["source_id"],
        item_batch["text"],
        item_batch["match_words"]
    )
    
    # Generate large model inference input files
    prompt = llm_processer.extract_triple_prompt(text, match_words, ref_kg_path)
    # Large model reasoning
    response = llm_processer.infer(prompt)
    # Post-processing of inference results (triple filtering)
    infer_triples, head_entities, tail_entities = Triple.get_triple(match_words, response)
    # The entity is validated again by calling the large model.
    verify_entities = llm_processer.entity_evaluate(tail_entities)
    
    return {
        "doc_name": doc_name,
        "source_id": source_id,
        "infer_triples": infer_triples,
        "head_entities": head_entities,
        "verify_entities": verify_entities
    }


def extract_desc(triple_path, corpus_path, task_conf, llm_processer):
    """
    Extract descriptions from triples (supports multi-threading for acceleration)
    """
    start_time = time.time()
    desc_output_path = str(triple_path).replace(".jsonl", "_descriptions.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus=json.load(f)
    corpus_dict = {item["hash_code"]: item["text"] for item in corpus}
        
    # Read triple data
    triples = read_jsonl(triple_path)
    logger.info(f"Total triples to add description: {len(triples)}")

    for item in triples:
        # Add context information to each triple
        source_id = item["source_id"]
        item["text"] = corpus_dict.get(source_id, "")

    
    # Thread pool configuration
    max_workers = task_conf["num_processes_infer"] if task_conf["num_processes_infer"] != -1 else multiprocessing.cpu_count()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_triple = {
            executor.submit(process_single_description, triple, llm_processer): triple 
            for triple in triples
        }
        
        # Use tqdm to display the processing progress.
        for future in tqdm(as_completed(future_to_triple), total=len(triples), desc="Extracting descriptions..."):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing description: {str(e)}")
                continue

    # Write all results to the output file  
    write_jsonl(data=all_results, path=desc_output_path, mode="w")

    end_time = time.time()
    logger.info(f"Description extraction completed in {end_time - start_time} seconds")

    # The number of described triples was successfully extracted.
    description_count = {"total": len(all_results), "with_description": 0, "without_description": 0}
    for r in all_results:
        if len(r["triple"].split("\t")) == 6:
            description_count["with_description"] += 1
        else:
            description_count["without_description"] += 1

    logger.info(f"Description extraction statistics: {description_count}")


def process_single_description(triple, llm_processor) -> str:
    """
    Description extraction for processing single triples
    """
    try:
        # Construct prompt
        text = triple["text"]
        triple_str = triple["triple"]
        prompt = llm_processor.extract_description_prompt(text, triple_str)

    except Exception as e:
        logger.error(f"extract_description prompt failed: {str(e)}")

    try:

        # Call LLM 
        response = llm_processor.infer(prompt, output_json=True) 
        if response == '':
            logger.error(f"extract_description failed, 2nd try")
            response = llm_processor.infer(prompt, output_json=True) 
        if response == '':
            logger.error(f"extract_description failed")
    except Exception as e:
        logger.error(f"extract_description failed: {str(e)}")

    try:
        result = Triple.parse_description_plustype_response(triple_str, response)

        triple["triple"] = result

        return triple

    except Exception as e:
        logger.error(f"parse_description_plustype_response failed: {str(e)}")

        return triple


def process_single_file(corpus_path, task_conf, llm_processer, output_dir="output"):
    """Triple extraction from a single file"""
    start_time = time.time()
    pedia_entity_path = task_conf["pedia_entity_path"]  # Header entity path

    try:
    # Dynamically generate output path
        file_name = Path(corpus_path).stem
        output_subdir = Path(output_dir) / file_name
        if output_subdir.exists():
            logger.info(f"Target files: {file_name} already exists, overwrite\n")
        else:
            output_subdir.mkdir(parents=True, exist_ok=True)
        #Move chunk file
        shutil.copy(corpus_path, output_subdir)
        
        # Initialize the output file path
        result_triple_path = output_subdir / f"new_triples_{file_name}.jsonl"
        next_layer_entities_path = output_subdir / f"next_layer_entities_{file_name}.txt"
        all_entities_path = output_subdir / f"all_entities_{file_name}.txt"
        match_words_path = output_subdir / f"match_words_{file_name}.jsonl"  ## Match result path


        # When triple extraction is not skipped, multi-level entity matching and triple extraction are performed.
        if not task_conf["skip_extract_triple"]:

            # The data is loaded into the header entity path for processing. Assuming there is a layer 0, the header entity is directly matched using `next_layer_entities_path`.
            head_entities = read_txt(pedia_entity_path)
            next_layer_entities = list(set([item.strip() for item in head_entities]))
            write_txt(next_layer_entities_path, next_layer_entities, mode="w")
            logger.info(f"Initialize next_layer_entities num: {len(next_layer_entities)}")
            
            # Initialize entity and triple files
            write_jsonl(data="", path=result_triple_path, mode="w")
            write_txt(data="", path=all_entities_path, mode="w")
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpusfiles=json.load(f)
            # Read corpus file
            # corpusfiles = read_jsonl(corpus_path)
            logger.info(f"corpus paragraph num: {len(corpusfiles)}")

            for iter in range(task_conf["level_num"]):
                logger.info(f"Processing {file_name} | Iteration {iter+1}/{task_conf['level_num']}")
                layer_head_cnt, layer_tail_cnt, layer_triple_cnt = 0, 0, 0

                logger.info(f"[num_iteration]: {iter+1} ---------------------\n")
                logger.info("[corpus matching]-----------------------------------\n")

                # Check if the file exists; if it does, delete it.
                if os.path.exists(match_words_path):
                    os.remove(match_words_path)
                            
                next_layer_entities = read_txt(next_layer_entities_path)
                next_layer_entities = [entity.strip("\n") for entity in next_layer_entities]

                source_id = "hash_code"
                text_key = "text"

                tasks_for_matching = [
                    (item, file_name, next_layer_entities, source_id, text_key) 
                    for item in corpusfiles
                ]
                
                num_processes = task_conf["num_processes_match"] if task_conf["num_processes_match"] != -1 else multiprocessing.cpu_count()
                logger.info(f"Starting AC matching for {len(tasks_for_matching)} paragraphs in {file_name} (Iter {iter+1}) using {num_processes} processes.")
                match_start_time = time.time()
                all_match_words = []
                if tasks_for_matching: # Ensure there are tasks to process
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        results_iterator = pool.imap(_process_paragraph_for_matching, tasks_for_matching)
                        all_match_words = list(tqdm(results_iterator, total=len(tasks_for_matching), 
                                                desc=f"AC Matching: {file_name} | Iter {iter+1}/{task_conf['level_num']}"))
                
                logger.info(f"[corpus match finished for {file_name} | Iteration {iter+1}]-----------------------------------\n")
                match_end_time = time.time()
                logger.info(f"Match time taken: {match_end_time - match_start_time} seconds")
                # The head entity matching results of each layer are written to a file (the next layer overwrites the previous layer).
                write_jsonl(data=all_match_words, path=match_words_path, mode="w")
                logger.info(f"Save current match result to: {match_words_path}")


                logger.info("[LLM response]-----------------------------------\n")
                ref_kg_path = task_conf["ref_kg_path"] 
                
                # Initialize the next level entity file
                if os.path.exists(next_layer_entities_path):
                    os.remove(next_layer_entities_path)

                # Initialize the counter and result set
                layer_head_cnt, layer_tail_cnt, layer_triple_cnt = 0, 0, 0
                current_all_triple = set()
                current_all_entity = set()

                # Read existing triples and entities
                current_all_triple_item = read_jsonl(result_triple_path)
                current_all_triple = set([item["triple"].lower() for item in current_all_triple_item])
                current_all_entity = set([item.strip().lower() for item in read_txt(all_entities_path)])

                # Use a thread pool to process LLM requests in parallel.
                max_workers = task_conf["num_processes_infer"] if task_conf["num_processes_infer"] != -1 else multiprocessing.cpu_count()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:  
                    # Submit all tasks to the thread pool
                    future_to_item = {
                        executor.submit(process_llm_batch, item, llm_processer, ref_kg_path): item 
                        for item in all_match_words
                    }
                    
                    # Use tqdm to display the processing progress.
                    for future in tqdm(as_completed(future_to_item), 
                                    total=len(all_match_words),
                                    desc=f"LLM Processing: {file_name} | Iter {iter+1}/{task_conf['level_num']}"):
                        try:
                            result = future.result()
                            
                            # Processing triples
                            new_triples_item = []
                            if result["infer_triples"] is not None:
                                for triple in result["infer_triples"]:
                                    if triple not in current_all_triple:
                                        layer_triple_cnt += 1
                                        current_all_triple.add(triple)
                                        triple_json = Triple.triple_json_format(
                                            triple, 
                                            result["doc_name"],
                                            result["source_id"]
                                        )
                                        new_triples_item.append(triple_json)
                                
                                if new_triples_item:
                                    write_jsonl(data=new_triples_item, path=result_triple_path, mode="a")
                                    logger.info(f"Add {len(new_triples_item)} triples to: {result_triple_path}")

                            # Handle header entity
                            if result["head_entities"] is not None:
                                head_entities_cnt = 0
                                for entity in result["head_entities"]:
                                    if entity not in current_all_entity:
                                        current_all_entity.add(entity)
                                        head_entities_cnt += 1
                                        layer_head_cnt += 1

                                # Update the complete entity list, overwrite
                                if head_entities_cnt > 0:
                                    write_txt(data=current_all_entity, path=all_entities_path, mode="w")
                                    logger.info(f"Add {head_entities_cnt} entities to: {all_entities_path}")

                            # Handle validation entities
                            if result["verify_entities"] is not None:
                                tmp_next_layer_entities = set()
                                for entity in result["verify_entities"]:
                                    entity_lower = entity.strip().lower()
                                    if entity_lower not in current_all_entity:
                                        current_all_entity.add(entity_lower)
                                        tmp_next_layer_entities.add(entity_lower)
                                        layer_tail_cnt += 1
                                
                                if tmp_next_layer_entities:
                                    write_txt(data=tmp_next_layer_entities, path=next_layer_entities_path, mode="a")
                                    logger.info(f"Save {len(tmp_next_layer_entities)} entities to: {next_layer_entities_path}")
                        
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            continue

                logger.info(f"layer: {iter+1}, add head: {layer_head_cnt}, tail: {layer_tail_cnt}, triple: {layer_triple_cnt}")
        

        # Extracting descriptions for triples
        if task_conf["extract_desc"]:
            if not os.path.getsize(result_triple_path) > 0:
              logger.warning(f"No triples found in {result_triple_path}, skip extracting descriptions")

            else:
                extract_desc(result_triple_path, corpus_path, task_conf, llm_processer)
        
        end_time = time.time()
        logger.info(f"Total time taken: {end_time - start_time} seconds")
        
        return True

    except Exception as e:
        logger.error(f"Error processing {corpus_path}: {str(e)}")
        return False


# Helper function for multiprocessing Aho-Corasick matching
def _process_paragraph_for_matching(args_tuple):
    """
    Worker function to process a single paragraph for entity matching.
    Unpacks arguments, creates a Corpus object, and performs matching.
    """
    item, file_name, local_next_layer_entities, source_id,  text_key = args_tuple
    corpus = Corpus(
        doc_name=file_name,
        source_id=item[source_id],
        corpus=item[text_key]
    )
    match_words = corpus.get_match_words(local_next_layer_entities)
    return match_words


def main():
    ## Read configuration file
    conf_path = "config.yaml"
    with open(conf_path, "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    logger.info(f"args:\n{args}\n")    

    task_conf = args["task_conf"]  ## Task parameters

    # Number of iterations to extract
    llm_conf = args["llm_conf"]  ## llm parameters
    llm_processer = LLM_Processor(llm_conf)

    
    # Input path processing (supports files/folders)
    dataset_root = args["dataset"]["root"]
    dataset = args["dataset"]["dataset"]
    input_path = f"{dataset_root}/{dataset}_chunk.json"   # task_conf["corpus_path"]  
    output_dir = task_conf["output_dir"]
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        files_to_process = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".jsonl")]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    # Batch processing
    success_count = 0
    
    # Move chunk file
    for corpus_path in tqdm(files_to_process, desc="Processing files"):
        if process_single_file(corpus_path, task_conf, llm_processer, output_dir):
            success_count += 1

    logger.info(f"Processed {success_count}/{len(files_to_process)} files successfully")
     


if __name__ == "__main__":
    main()
