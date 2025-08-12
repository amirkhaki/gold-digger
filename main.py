import os
import json
import time
import requests
import sys
import argparse
import google.generativeai as genai
from tqdm import tqdm
import subprocess

# --- Configuration ---
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# --- Caching ---
def load_cache(cache_file):
    if os.path.exists(cache_file + ".tmp"):
        print("Found temporary cache file, attempting to load it.")
        try:
            with open(cache_file + ".tmp", "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Could not load temporary cache file.")
            pass # Fall through to loading the main cache file

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Could not parse cache file. Starting with an empty cache.")
            return {}
    return {}

def save_cache(cache, cache_file):
    tmp_file = cache_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(cache, f, indent=4)
    os.rename(tmp_file, cache_file)

def to_bibtex(papers):
    """
    Converts a list of paper details to a BibTeX string.
    """
    bibtex_entries = []
    for paper in papers:
        # Prefer the BibTeX from citationStyles if available
        if paper.get("citationStyles") and paper["citationStyles"].get("bibtex"):
            bibtex_entries.append(paper["citationStyles"]["bibtex"])
            continue

        # Fallback to manual creation
        entry_type = "@article" # Default to article
        if paper.get("venue"):
            # Heuristic to determine if it's a conference paper
            if "proceedings" in paper.get("venue", "").lower() or "conference" in paper.get("venue", "").lower():
                entry_type = "@inproceedings"
        
        citation_key = paper.get("paperId", "")
        
        fields = []
        if paper.get("title"):
            fields.append(f"  title     = {{{paper.get('title')}}}")
        if paper.get("authors"):
            authors = " and ".join([author["name"] for author in paper.get("authors", [])])
            fields.append(f"  author    = {{{authors}}}")
        if paper.get("year"):
            fields.append(f"  year      = {{{paper.get('year')}}}")
        if paper.get("venue"):
            if entry_type == "@article":
                fields.append(f"  journal   = {{{paper.get('venue')}}}")
            else:
                fields.append(f"  booktitle = {{{paper.get('venue')}}}")
        
        external_ids = paper.get("externalIds", {})
        if external_ids.get("DOI"):
            fields.append(f"  doi       = {{{external_ids.get('DOI')}}}")

        if paper.get("abstract"):
            fields.append(f"  abstract  = {{{paper.get('abstract')}}}")
            
        bibtex_entry = f"{entry_type}{{{citation_key},\n" + ",\n".join(fields) + "\n}"
        bibtex_entries.append(bibtex_entry)
        
    return "\n\n".join(bibtex_entries)


# --- Semantic Scholar API ---
def get_paper_details_batch(paper_ids, cache_file, retry_on_400=0):
    """
    Fetches paper details in batch from Semantic Scholar API, using cache if available.
    """
    cache = load_cache(cache_file)
    
    papers_to_fetch = [pid for pid in paper_ids if pid not in cache]
    
    if papers_to_fetch:
        print(f"Fetching details for {len(papers_to_fetch)} papers in a batch.")
        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

        retries = retry_on_400
        while retries >= 0:
            try:
                response = requests.post(
                    'https://api.semanticscholar.org/graph/v1/paper/batch',
                    params={'fields': 'title,authors,year,abstract,citations,references,citationStyles,externalIds'},
                    json={"ids": papers_to_fetch},
                    headers=headers
                )
                response.raise_for_status()
                
                response_data = response.json()
                for original_id, paper_data in zip(papers_to_fetch, response_data):
                    if paper_data: # API returns None for papers not found
                        # Ensure citations and references are lists
                        if 'citations' not in paper_data:
                            paper_data['citations'] = []
                        if 'references' not in paper_data:
                            paper_data['references'] = []
                        
                        # Cache by both the canonical ID and the original ID
                        cache[paper_data['paperId']] = paper_data
                        cache[original_id] = paper_data

                save_cache(cache, cache_file)
                break  # Success, exit the loop

            except requests.exceptions.RequestException as e:
                if e.response is not None and e.response.status_code == 400 and retries > 0:
                    print(f"Client error 400, retrying in 1 second... ({retries} retries left)")
                    time.sleep(1)
                    retries -= 1
                else:
                    print(f"Error fetching batch of papers: {e}")
                    break # Unrecoverable error or no retries left

    return {pid: cache.get(pid) for pid in paper_ids}

# --- Filtering ---
def or_filter(papers, sub_filters):
    """
    Applies a list of filters with OR logic.
    """
    print(f"Applying OR filter to {len(papers)} papers...")
    accepted_papers = set()
    for filter_func, filter_arg in sub_filters:
        filtered_papers = filter_func(papers, filter_arg)
        for paper in filtered_papers:
            accepted_papers.add(paper['paperId'])
    
    # Return the full paper objects
    final_list = [paper for paper in papers if paper['paperId'] in accepted_papers]
    print(f"Found {len(final_list)} matching papers after OR filter.")
    return final_list

def filter_by_field(papers, field_keyword_tuple):
    """
    Filters a list of papers based on a keyword in a specific field.
    """
    field, keyword = field_keyword_tuple
    print(f"Filtering {len(papers)} papers with keyword '{keyword}' in field '{field}'...")
    
    filtered_papers = []
    for paper in papers:
        field_value = paper.get(field, '') or ''
        if isinstance(field_value, str) and keyword.lower() in field_value.lower():
            filtered_papers.append(paper)

    print(f"Found {len(filtered_papers)} matching papers.")
    return filtered_papers

def filter_by_year(papers, year_op_tuple):
    """
    Filters a list of papers by publication year using an operator.
    """
    op, year_str = year_op_tuple
    try:
        year = int(year_str)
    except ValueError:
        print(f"Invalid year format: {year_str}. Year must be an integer.")
        return []

    print(f"Filtering {len(papers)} papers with year {op} {year}...")
    
    ops = {
        "lt": lambda x, y: x < y,
        "gt": lambda x, y: x > y,
        "eq": lambda x, y: x == y,
        "le": lambda x, y: x <= y,
        "ge": lambda x, y: x >= y,
    }

    if op not in ops:
        print(f"Invalid operator: {op}. Supported operators are: {list(ops.keys())}")
        return []

    filtered_papers = []
    for paper in papers:
        paper_year = paper.get('year')
        if paper_year and ops[op](paper_year, year):
            filtered_papers.append(paper)

    print(f"Found {len(filtered_papers)} matching papers.")
    return filtered_papers

def filter_by_author(papers, author_name):
    """
    Filters a list of papers by author name.
    """
    print(f"Filtering {len(papers)} papers by author '{author_name}'...")
    
    filtered_papers = []
    for paper in papers:
        authors = paper.get('authors', [])
        for author in authors:
            if author_name.lower() in author.get('name', '').lower():
                filtered_papers.append(paper)
                break  # Move to the next paper once a match is found

    print(f"Found {len(filtered_papers)} matching papers.")
    return filtered_papers

def filter_papers_with_llm(papers, criterion, llm_provider='gemini-api', gemini_cli_path='gemini', batch_size=5):
    """
    Filters a list of papers based on a criterion using an LLM provider.
    """
    print(f"Filtering {len(papers)} papers with LLM ({llm_provider})...")

    if llm_provider == 'gemini-api':
        # Configure the Gemini API
        # TODO: Replace "YOUR_API_KEY" with your actual Gemini API key.
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')

    filtered_papers = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        
        prompt = "Here are several papers:\n\n"
        for j, paper in enumerate(batch):
            prompt += f"Paper {j+1} Title: {paper.get('title', '')}\n"
            prompt += f"Paper {j+1} Abstract: {paper.get('abstract', '')}\n\n"
        
        prompt += f"Criterion: {criterion}\n\n"
        prompt += "Which of these papers (listed by their number) meet the criterion? Please return a comma-separated list of numbers (e.g., 1, 3, 5)."

        try:
            if llm_provider == 'gemini-api':
                response = model.generate_content(prompt)
                response_text = response.text
            elif llm_provider == 'gemini-cli':
                process = subprocess.run([gemini_cli_path, "-p", prompt], capture_output=True, text=True)
                if process.returncode != 0:
                    print(f"gemini-cli failed with error: {process.stderr}")
                    continue
                response_text = process.stdout
            else:
                print(f"Unknown LLM provider: {llm_provider}")
                continue

            # Extract numbers from the response
            try:
                paper_numbers = [int(n.strip()) for n in response_text.split(',') if n.strip().isdigit()]
                for num in paper_numbers:
                    if 1 <= num <= len(batch):
                        filtered_papers.append(batch[num-1])
            except (ValueError, IndexError) as e:
                print(f"Could not parse LLM response: {response_text}. Error: {e}")

        except Exception as e:
            print(f"An error occurred while processing a batch: {e}")

    print(f"Found {len(filtered_papers)} matching papers.")
    return filtered_papers

def filter_papers_with_llm_from_file(papers, file_path, llm_provider='gemini-api', gemini_cli_path='gemini', batch_size=5):
    """
    Filters papers based on a criterion read from a file.
    """
    try:
        with open(file_path, 'r') as f:
            criterion = f.read()
    except FileNotFoundError:
        print(f"Criterion file not found: {file_path}")
        return []
    
    return filter_papers_with_llm(papers, criterion, llm_provider, gemini_cli_path, batch_size)


# --- Core Logic ---

def snowball_literature(starting_papers, filters, cache_file, batch_size=10, retry_on_400=0):
    """
    Main function to perform the literature snowballing.
    """
    papers_to_check = set(starting_papers)
    done_papers = {}

    with tqdm(total=len(papers_to_check)) as pbar:
        try:
            while papers_to_check:
                # Create a batch of papers to process
                batch_ids = []
                while papers_to_check and len(batch_ids) < batch_size:
                    paper_id = papers_to_check.pop()
                    if paper_id not in done_papers:
                        batch_ids.append(paper_id)

                if not batch_ids:
                    continue

                paper_details_batch = get_paper_details_batch(batch_ids, cache_file, retry_on_400)

                for paper_id, paper_details in paper_details_batch.items():
                    if not paper_details:
                        pbar.update(1)
                        continue

                    done_papers[paper_id] = paper_details

                    # Combine citations and references for processing
                    related_papers = paper_details.get("citations", []) + paper_details.get("references", [])

                    if related_papers:
                        # Apply filters in sequence
                        filtered_papers = related_papers
                        for filter_func, filter_arg in filters:
                            filtered_papers = filter_func(filtered_papers, filter_arg)

                        new_papers = 0
                        for paper in filtered_papers:
                            if paper['paperId'] not in done_papers:
                                papers_to_check.add(paper['paperId'])
                                new_papers += 1
                        pbar.total += new_papers
                    
                    pbar.update(1)
                    pbar.set_description(f"Processed: {paper_id}")

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            sys.exit(0)

    print("Literature snowballing complete.")
    return done_papers


# --- Main Execution ---

FILTER_MAPPING = {
    "field": filter_by_field,
    "year": filter_by_year,
    "author": filter_by_author,
    "llm": filter_papers_with_llm,
    "llm_from_file": filter_papers_with_llm_from_file,
}

def parse_filter_args(filter_args, filter_mapping=FILTER_MAPPING):
    """
    Parses the filter arguments from the command line.
    """
    filter_pipeline = []
    or_group = None

    for args in filter_args:
        filter_name = args[0]

        if filter_name == "or_start":
            if or_group is not None:
                print("Nested ORs are not supported.")
                return None
            or_group = []
            continue
        
        if filter_name == "or_end":
            if or_group is None:
                print("or_end found without a matching or_start.")
                return None
            filter_pipeline.append((or_filter, or_group))
            or_group = None
            continue

        if filter_name not in filter_mapping:
            print(f"Unknown filter: {filter_name}")
            return None

        filter_func = filter_mapping[filter_name]
        filter_arg = tuple(args[1:])
        
        # Special handling for single-argument filters
        if len(filter_arg) == 1:
            filter_arg = filter_arg[0]

        if or_group is not None:
            or_group.append((filter_func, filter_arg))
        else:
            filter_pipeline.append((filter_func, filter_arg))

    if or_group is not None:
        print("or_start found without a matching or_end.")
        return None

    return filter_pipeline

def main(initial_papers, output_file, filter_args, cache_file, llm_provider, gemini_cli_path, batch_size, llm_batch_size, retry_on_400, output_format):
    """
    Main function to run the literature snowballing process.
    """
    
    llm_filter = lambda papers, criterion: filter_papers_with_llm(papers, criterion, llm_provider, gemini_cli_path, llm_batch_size)
    llm_from_file_filter = lambda papers, file_path: filter_papers_with_llm_from_file(papers, file_path, llm_provider, gemini_cli_path, llm_batch_size)

    local_filter_mapping = FILTER_MAPPING.copy()
    local_filter_mapping['llm'] = llm_filter
    local_filter_mapping['llm_from_file'] = llm_from_file_filter

    filter_pipeline = parse_filter_args(filter_args, local_filter_mapping)
    if filter_pipeline is None:
        return

    results = snowball_literature(initial_papers, filter_pipeline, cache_file, batch_size, retry_on_400)

    # Prepare results for saving
    papers_to_save = []
    for paper in results.values():
        # Create a copy to avoid modifying the original data
        paper_copy = paper.copy()
        paper_copy.pop('citations', None)  # Remove citations if they exist
        paper_copy.pop('references', None) # Remove references if they exist
        papers_to_save.append(paper_copy)

    # Write results to a file
    if output_format == 'json':
        with open(output_file, "w") as f:
            json.dump(papers_to_save, f, indent=4)
        print(f"Results saved to {output_file}")
    elif output_format == 'bibtex':
        bibtex_data = to_bibtex(papers_to_save)
        # Ensure the output file has a .bib extension
        if not output_file.endswith('.bib'):
            output_file = os.path.splitext(output_file)[0] + '.bib'
        with open(output_file, "w") as f:
            f.write(bibtex_data)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Snowball literature search tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--initial-papers", nargs='+', help="List of initial paper IDs. Can be a Semantic Scholar Paper ID, DOI, or arXiv ID.")
    parser.add_argument("--output-file", default="snowball_results.json", help="Output file for the results.")
    parser.add_argument("--output-format", choices=['json', 'bibtex'], default='json', help="Output format for the results.")
    parser.add_argument("--cache-file", default="semantic_scholar_cache.json", help="Cache file for Semantic Scholar data.")
    parser.add_argument("--batch-size", type=int, default=10, help="Maximum number of papers to fetch in a single batch.")
    parser.add_argument("--llm-batch-size", type=int, default=5, help="Maximum number of papers to process in a single batch with the LLM filter.")
    parser.add_argument("--retry-on-400", type=int, default=0, help="Number of retries on HTTP 400 errors.")
    parser.add_argument("--llm-provider", choices=['gemini-api', 'gemini-cli'], default='gemini-api', help="LLM provider to use.")
    parser.add_argument("--gemini-cli-path", default='gemini', help="Path to the gemini-cli executable.")
    parser.add_argument("--filter", nargs='+', action='append', 
                        help="""Filter to apply. Can be used multiple times.

Available filters:
- field <field_name> <keyword>: Filter by keyword in a field (e.g., title, abstract).
- year <lt|gt|eq|le|ge> <year>: Filter by publication year.
- author <author_name>: Filter by author name.
- llm <criterion>: Filter with a custom criterion using an LLM.
- llm_from_file <file_path>: Filter with a criterion from a file.
- or_start / or_end: Group filters with OR logic.""")
    parser.add_argument("--convert-to-bibtex", help="Convert an existing JSON results file to BibTeX format.")

    args = parser.parse_args()

    if args.convert_to_bibtex:
        try:
            with open(args.convert_to_bibtex, 'r') as f:
                papers = json.load(f)
            
            bibtex_data = to_bibtex(papers)
            output_file = os.path.splitext(args.convert_to_bibtex)[0] + '.bib'
            
            with open(output_file, 'w') as f:
                f.write(bibtex_data)
            
            print(f"Successfully converted {args.convert_to_bibtex} to {output_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error converting file: {e}")
        sys.exit(0)

    if not args.initial_papers:
        parser.error("--initial-papers is required when not using --convert-to-bibtex")

    if not args.filter:
        parser.error("--filter is required when not using --convert-to-bibtex")

    main(args.initial_papers, args.output_file, args.filter, args.cache_file, args.llm_provider, args.gemini_cli_path, args.batch_size, args.llm_batch_size, args.retry_on_400, args.output_format)
