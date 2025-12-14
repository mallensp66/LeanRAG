import os
import heapq
import re
import os
import json
# import ijson
# from rdflib import Graph



# def get_ttl(input_file, out_file):
#     cnt = 0    
#     # Create a graph object
#     g = Graph()
#     # Parsing .ttl files
#     try:
#         g.parse(input_file, format="ttl")
#     except Exception as e:
#         print(f"Error parsing file: {e}")

#     with open(out_file, "w", encoding="utf-8") as outfile:
#         # Traverse all triples in the graph
#         for subj, pred, obj in g.triples((None, None, None)):
#             print(f"Subject: {subj}, Predicate: {pred}, Object: {obj}")
#             previous_item = None

#             cnt += 1       
#             # Write output file
#             outfile.write(f"<{subj}>\t<{pred}>\t<{obj}>\n")    
#             # Output processing progress, once every 10,000 objects processed.
#             progress = cnt + 1
#             if progress % 10000 == 0:
#                 print(f"Processing progress: {progress}")



def remove_text_in_brackets(input_file, output_file, strong_clean_flag = False):
    def process(line):
        """  
        Remove the brackets and their contents from the string.
        This includes parentheses () and Chinese parentheses ().
        :param line: raw string
        :return: The string after deleting the parentheses
        """
        # Use regular expressions to match parentheses and their contents.
        # Matches parentheses () or Chinese parentheses (), and the content within them.
        return re.sub(r"[\(\（][^\)\）]*[\)\）]", "", line)

    def remove_non_alpha(string):
        # Use regular expressions to delete non-alphanumeric characters at the beginning and end.
        cleaned_string = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', string)
        return cleaned_string

    # Ensure the input file exists.
    if not os.path.exists(input_file):
        print(f"The input file {input_file} does not exist!")
        return

    try:
        # Open input and output files
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            cnt = 0
            # Read the input file line by line.
            for line in infile:
                cnt += 1       
                # Delete the brackets and their contents in the current line.
                line_wo_bracket = process(line)
                if strong_clean_flag == True:
                    # Numbers and symbols before and after deletion
                    processed_line = remove_non_alpha(line_wo_bracket)
                # Write to output file
                outfile.write(processed_line + "\n")
                if cnt % 10000 == 0:
                    print(f"Processing progress: {cnt}")

        print(f"Processing complete! Results saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def sort_large_file(
    input_file, output_file, chunk_size=100000
):  # chunk_size Adjustable based on memory
    chunks = []
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            while True:
                lines = infile.readlines(chunk_size)
                if not lines:
                    break
                lines.sort()  # Sort each piece of data

                chunk_filename = f"temp_chunk_{len(chunks)}.txt"
                chunks.append(chunk_filename)
                with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
                    chunk_file.writelines(lines)

        # Merged and sorted blocks
        cnt = 1
        with open(output_file, "w", encoding="utf-8") as outfile:
            files = [open(f, "r", encoding="utf-8") for f in chunks]
           
            for line in heapq.merge(*files):
                cnt += 1
                outfile.write(line)
                if cnt % 10000 == 0:
                    print(f'merging process: {cnt}/{len(chunks)*chunk_size}')
            for f in files:
                f.close()

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean temporary files
        for chunk in chunks:
            try:
                os.remove(chunk)
            except FileNotFoundError:
                pass


def remove_duplicates(input_file, output_file):
    """
    Remove duplicate lines from a sorted text file.

    Args:
        input_file: The sorted input file paths.
        output_file: Output file paths without duplicate lines.
    """
    try:
        duplicate_cnt = 0
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            # Initialize the content of the previous line
            previous_line = None
            index = 0
            for current_line in infile:
                index += 1
                # If the current line differs from the previous line, write it to the output file.
                if current_line != previous_line:
                    outfile.write(current_line)
                    duplicate_cnt += 1
                    previous_line = current_line
                progress = index + 1
                if progress % 10000 == 0:
                    print(f"Processing progress: {progress}")
        
        print(f"The removal of duplicate rows {duplicate_cnt} is complete.")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred while processing the file:{e}")


def str_full_to_half_width(input_file, output_file):
    def process(ustring):
        result = []
        for char in ustring:
            # Full-width spaces are directly converted to half-width spaces.
            if char == "\u3000":
                result.append(" ")
            # Handle other full-width characters (based on Unicode range)
            elif "\uff01" <= char <= "\uff5e":
                # Subtract the offset 0xFEE0 from the Unicode code of the full-width character to get the half-width character.
                result.append(chr(ord(char) - 0xFEE0))
            else:
                # Retain characters that were originally half-width.
                result.append(char)
        return "".join(result)

    # Ensure the input file exists.
    if not os.path.exists(input_file):
        print(f"The input file {input_file} does not exist!")
        return

    try:
        # Open input and output files
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:

            # Read the input file line by line.
            for line in infile:
                # Convert the content of the current line from full-width to half-width characters.
                converted_line = process(line)
                # Write to output file
                outfile.write(converted_line)
        print(f"Processing complete! The converted content has been saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def file_split(input_file, num):
        # Assuming your filename is 'data.txt'
        # input_file = 'input_file.txt'
        input_name = input_file.split('.')[0]
        file_type = input_file.split('.')[1]
        output_files = []
        for i in range(num):
            output_files.append(f'{input_name}_part{i}.{file_type}')

        # Calculate the total number of lines in the file
        with open(input_file, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file)

        # Number of lines in each section
        lines_per_part = total_lines // 4

        # Read and write line by line
        with open(input_file, 'r', encoding='utf-8') as file:
            current_part = 0
            current_line = 0
            output_file = open(output_files[current_part], 'w', encoding='utf-8')
            
            for line in file:
                output_file.write(line)
                current_line += 1
                
                # Check if you need to switch to the next file.
                if current_line >= lines_per_part and current_part < 3:
                    output_file.close()
                    current_part += 1
                    output_file = open(output_files[current_part], 'w', encoding='utf-8')
                    current_line = 0
            
            # Close last file
            output_file.close()

def print_file(infile, n=10):
    # Open the file and read the first 10 lines.
    with open(infile, 'r') as file:
        for i in range(n):
            line = file.readline()
            if line:  # Check if there are still lines to read.
                print(line.strip())
            else:
                break  # If the file has fewer than 10 lines, exit early.

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        line_count = 0
        for line in file:
            line_count += 1
            if line_count % 1000000 == 0:
                print(line_count)

    print(f"文件 {filename} 中的行数为: {line_count}")


# def json_pass(infilename, outfilename):
#     with open(infilename, "r", encoding="utf-8") as infile, open(outfilename, "w", encoding="utf-8") as outfile:
#         for index, item in enumerate(ijson.items(infile, 'item')):  # Assume the top layer is an array
#             try:                
#                 # Extract the content of the "en" field
#                 extracted = {
#                     "type": item.get("type", {}),
#                     "id": item.get("id", {}),
#                     "labels": item.get("labels", {}).get("en", ""),
#                     "descriptions": item.get("descriptions", {}).get("en", ""),
#                     "aliases": item.get("aliases", {}).get("en", [])
#                 }
                
#                 # Write the extracted results to the output file.
#                 outfile.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            
#                 # Output processing progress, once every 10,000 objects processed.
#                 progress = index + 1
#                 if progress % 10000 == 0:
#                     print(f"Processing progress: {progress}")
#             except:
#                 pass

#     print(f"Extraction complete, results saved to {outfile}")

def get_entities_from_triples(infilename, outfilename, signpass=[]):
    cnt = 0
    with open(infilename, 'r', encoding='utf-8') as infile, open(outfilename, 'w', encoding='utf-8') as outfile:
        # Initialize the content of the previous line
        previous_item = None
        for line in infile:
            cnt += 1
            # Split each row and extract the first column.
            current_item = line.split("\t")[0]
            # Filter out special characters
            for sign in signpass:
                current_item = current_item.strip(sign)
            current_item = re.sub(r"_+", " ", current_item)
            if current_item != previous_item:
                # Write output file
                outfile.write(current_item + '\n')    
                previous_item = current_item      
            # Output processing progress, once every 10,000 objects processed.
            progress = cnt + 1
            if progress % 10000 == 0:
                print(f"Processing progress: {progress}")
def get_entities_from_wikidata(infilename, outfilename1, outfilename2, outfilename3):
    with open(infilename, "r", encoding="utf-8") as infile, open(outfilename1, "w", encoding="utf-8") as outfile1, open(outfilename2, "w", encoding="utf-8") as outfile2, open(outfilename3, "w", encoding="utf-8") as outfile3:
        index = 0
        for line in infile:  # Assuming the top layer is an array, try:  
            index += 1    
            item = json.loads(line)
            # Extract the content of the "en" field
            try:
                entity = item.get("labels", {}).get("value", "")
                aliases_dic = item.get("aliases", [])
                aliases = []
                for item in aliases_dic:
                    aliases.append(item.get("value", ""))
                descriptions = item.get("descriptions", "")
                if descriptions != "":
                    descriptions = descriptions.get("value", "")
            except:
                pass
                            
            # Write the extracted results to the output file.
            outfile1.write(entity + "\n")
            outfile2.write(entity + "\n")
            for item in aliases:
                outfile2.write(item + "\n")
            outfile3.write(f"<{entity}>\t<{aliases}>\t<{descriptions}>\n")
        
            # Output processing progress, once every 10,000 objects processed.
            progress = index + 1
            if progress % 10000 == 0:
                print(f"Processing progress: {progress}")

def main():
    # infile = '/data/H-RAG/H_RAG/data/kg/wikidata-20240101-all.json'
    # outfile = '/data/H-RAG/H_RAG/data/kg/extracted.json'
    # data1 = "/data/H-RAG/H_RAG/data/kg/triples.txt"
    # Example usage
    # input_path = "CN-DBpediaMention2Entity.txt"  # Replace with your sorted input file path
    # temp1 = "temp1.txt"
    # temp2 = "temp2.txt"
    # output_path = "CN-DBpedia_pass.txt"  # Replace with the file path you want to output
    # remove_duplicates('triple_data/wiki_entities_sorted.txt', output_file = 'triple_data/wiki_entities.txt')
    # remove_duplicates('triple_data/wiki_aliases_sorted.txt', output_file = 'triple_data/wiki_aliases.txt')
    # remove_duplicates('triple_data/wiki_descriptions_sorted.txt', output_file = 'triple_data/wiki_descriptions.txt')
    # str_full_to_half_width(temp1, temp2)
    # remove_duplicates(temp2, output_path)
    # file_split('CN-DBpedia_pass.txt', 4)
    # print_file('/data/H-RAG/H_RAG/data/kg/triples.txt', 1000)
    # count_lines('/data/H-RAG/H_RAG/data/kg/triples.txt')
    # get_entities_from_triples("triples.txt", "triples_entities.txt", ['>', '<'])
    # get_entities_from_wikidata('/data/H-RAG/H_RAG/data/kg/extracted.json', 'wiki_entities.txt', 'wiki_aliases.txt', 'wiki_descriptions.txt')
    # yago-beyond-wikipedia, yago-facts, yago-schema, yago-taxonomy, yago-schema
    # get_ttl('/data/H-RAG/H_RAG/data/kg/yago-schema.ttl', 'yago_wiki_triples.txt')
    # json_pass(infile, outfile)
    # remove_text_in_brackets('triple_data/dbpedia_entities.txt', 'triple_data/dbpedia_entities_clean.txt', True)
    remove_duplicates('triple_data/temp.txt', 'triple_data/dbpedia_entities_clean.txt')


if __name__ == "__main__":
    main()
