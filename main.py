import os
import json
import time
import requests
import sys
import argparse
import google.generativeai as genai
from tqdm import tqdm

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

# --- Semantic Scholar API ---
def get_paper_details(paper_id, cache_file):
    """
    Fetches paper details from Semantic Scholar API, using cache if available.
    """
    cache = load_cache(cache_file)
    if paper_id in cache:
        print(f"Using cached details for paper: {paper_id}")
        return cache[paper_id]

    print(f"Fetching details for paper: {paper_id}")
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

    try:
        # Request DOI field from the API
        response = requests.get(f"https://api.semanticscholar.org/v1/paper/{paper_id}?fields=title,authors,year,abstract,citations,references,doi", headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        paper_data = response.json()

        # Ensure citations and references are lists
        if 'citations' not in paper_data:
            paper_data['citations'] = []
        if 'references' not in paper_data:
            paper_data['references'] = []
            
        cache[paper_id] = paper_data
        save_cache(cache, cache_file)
        return paper_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper {paper_id}: {e}")
        return None

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
    year, op = year_op_tuple
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

def filter_papers_with_llm(papers, criterion):
    """
    Filters a list of papers based on a criterion using the Gemini API in batch mode.
    """
    print(f"Filtering {len(papers)} papers with LLM in batch mode...")

    # Configure the Gemini API
    # TODO: Replace "YOUR_API_KEY" with your actual Gemini API key.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    filtered_papers = []
    BATCH_SIZE = 5

    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i+BATCH_SIZE]
        
        prompt = "Here are several papers:\n\n"
        for j, paper in enumerate(batch):
            prompt += f"Paper {j+1} Title: {paper.get('title', '')}\n"
            prompt += f"Paper {j+1} Abstract: {paper.get('abstract', '')}\n\n"
        
        prompt += f"Criterion: {criterion}\n\n"
        prompt += "Which of these papers (listed by their number) meet the criterion? Please return a comma-separated list of numbers (e.g., 1, 3, 5)."

        try:
            response = model.generate_content(prompt)
            # Extract numbers from the response
            try:
                paper_numbers = [int(n.strip()) for n in response.text.split(',') if n.strip().isdigit()]
                for num in paper_numbers:
                    if 1 <= num <= len(batch):
                        filtered_papers.append(batch[num-1])
            except (ValueError, IndexError) as e:
                print(f"Could not parse LLM response: {response.text}. Error: {e}")

        except Exception as e:
            print(f"An error occurred while processing a batch: {e}")

    print(f"Found {len(filtered_papers)} matching papers.")
    return filtered_papers

def filter_papers_with_llm_from_file(papers, file_path):
    """
    Filters papers based on a criterion read from a file.
    """
    try:
        with open(file_path, 'r') as f:
            criterion = f.read()
    except FileNotFoundError:
        print(f"Criterion file not found: {file_path}")
        return []
    
    return filter_papers_with_llm(papers, criterion)


# --- Core Logic ---

def snowball_literature(starting_papers, filters, cache_file):
    """
    Main function to perform the literature snowballing.
    """
    papers_to_check = set(starting_papers)
    done_papers = {}

    with tqdm(total=len(papers_to_check)) as pbar:
        try:
            while papers_to_check:
                paper_id = papers_to_check.pop()
                if paper_id in done_papers:
                    pbar.update(1)
                    continue

                paper_details = get_paper_details(paper_id, cache_file)
                
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

def parse_filter_args(filter_args):
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

        if filter_name not in FILTER_MAPPING:
            print(f"Unknown filter: {filter_name}")
            return None

        filter_func = FILTER_MAPPING[filter_name]
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

def main(initial_papers, output_file, filter_args, cache_file):
    """
    Main function to run the literature snowballing process.
    """
    filter_pipeline = parse_filter_args(filter_args)
    if filter_pipeline is None:
        return
    print(filter_pipeline)

    results = snowball_literature(initial_papers, filter_pipeline, cache_file)

    # Write results to a file
    with open(output_file, "w") as f:
        json.dump(list(results.values()), f, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Snowball literature search tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--initial-papers", nargs='+', required=True, help="List of initial paper IDs.")
    parser.add_argument("--output-file", default="snowball_results.json", help="Output file for the results.")
    parser.add_argument("--cache-file", default="semantic_scholar_cache.json", help="Cache file for Semantic Scholar data.")
    parser.add_argument("--filter", nargs='+', action='append', required=True, 
                        help="""Filter to apply. Can be used multiple times.

Available filters:
- field <field_name> <keyword>: Filter by keyword in a field (e.g., title, abstract).
- year <lt|gt|eq|le|ge> <year>: Filter by publication year.
- author <author_name>: Filter by author name.
- llm <criterion>: Filter with a custom criterion using an LLM.
- llm_from_file <file_path>: Filter with a criterion from a file.
- or_start / or_end: Group filters with OR logic.""")

    args = parser.parse_args()
    main(args.initial_papers, args.output_file, args.filter, args.cache_file)
