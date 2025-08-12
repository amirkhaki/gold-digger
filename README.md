# gold-digger
literature snowballing

``` shell

usage: main.py [-h] --initial-papers INITIAL_PAPERS [INITIAL_PAPERS ...] [--output-file OUTPUT_FILE] [--cache-file CACHE_FILE] --filter FILTER [FILTER ...]

Snowball literature search tool.

options:
  -h, --help            show this help message and exit
  --initial-papers INITIAL_PAPERS [INITIAL_PAPERS ...]
                        List of initial paper IDs.
  --output-file OUTPUT_FILE
                        Output file for the results.
  --cache-file CACHE_FILE
                        Cache file for Semantic Scholar data.
  --filter FILTER [FILTER ...]
                        Filter to apply. Can be used multiple times.
                        
                        Available filters:
                        - field <field_name> <keyword>: Filter by keyword in a field (e.g., title, abstract).
                        - year <lt|gt|eq|le|ge> <year>: Filter by publication year.
                        - author <author_name>: Filter by author name.
                        - llm <criterion>: Filter with a custom criterion using an LLM.
                        - llm_from_file <file_path>: Filter with a criterion from a file.
                        - or_start / or_end: Group filters with OR logic.
```
