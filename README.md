# Gold Digger

Gold Digger is a command-line tool for snowballing literature search, starting from a few initial papers. It uses the Semantic Scholar API to find citations and references, and provides a flexible filtering mechanism to narrow down the results.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/gold-digger.git
    cd gold-digger
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

### Semantic Scholar API Key

The tool can be used without a Semantic Scholar API key, but it is recommended to use one to get higher rate limits. You can get a key from the [Semantic Scholar website](https://www.semanticscholar.org/product/api).

Once you have a key, you can set it as an environment variable:

```bash
export SEMANTIC_SCHOLAR_API_KEY="YOUR_API_KEY"
```

### Gemini API Key

The `llm` filter can use the Gemini API for filtering. To use it, you need a Gemini API key. You can get one from [Google AI Studio](https://aistudio.google.com/).

Set the API key as an environment variable:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Usage

```bash
python main.py --initial-papers <paper_id_1> <paper_id_2> ... --filter <filter_1> --filter <filter_2> ...
```

**Note:** If you interrupt the process with `Ctrl-C`, the papers that have been processed so far will be saved to the output file.

### Arguments

*   `--initial-papers`: A list of initial paper IDs to start the snowballing from. These can be Semantic Scholar Paper IDs, DOIs, or arXiv IDs. (Required, unless using `--convert-to-bibtex`)
*   `--output-file`: The file to save the results to. (Default: `snowball_results.json`)
*   `--output-format`: The format to save the results in. (Choices: `json`, `bibtex`; Default: `json`)
*   `--cache-file`: The file to use for caching Semantic Scholar API responses. (Default: `semantic_scholar_cache.json`)
*   `--batch-size`: The maximum number of papers to fetch in a single batch from Semantic Scholar. (Default: 10)
*   `--llm-batch-size`: The maximum number of papers to process in a single batch with the LLM filter. (Default: 5)
*   `--retry-on-400`: The number of times to retry on HTTP 400 errors. (Default: 0)
*   `--llm-provider`: The LLM provider to use for the `llm` filter. (Choices: `gemini-api`, `gemini-cli`; Default: `gemini-api`)
*   `--gemini-cli-path`: The path to the `gemini-cli` executable. (Default: `gemini`)
*   `--filter`: A filter to apply to the papers. This argument can be used multiple times. (Required, unless using `--convert-to-bibtex`)
*   `--convert-to-bibtex`: Convert an existing JSON results file to BibTeX format. When this option is used, no snowballing is performed.

## Filters

Filters are used to narrow down the results of the literature search.

### `field`

Filters papers by a keyword in a specific field.

**Syntax:** `--filter field <field_name> <keyword>`

*   `<field_name>`: The field to search in (e.g., `title`, `abstract`).
*   `<keyword>`: The keyword to search for.

### `year`

Filters papers by publication year.

**Syntax:** `--filter year <operator> <year>`

*   `<operator>`: One of `lt` (less than), `gt` (greater than), `eq` (equal to), `le` (less than or equal to), `ge` (greater than or equal to).
*   `<year>`: The year to compare against.

### `author`

Filters papers by author name.

**Syntax:** `--filter author <author_name>`

*   `<author_name>`: The name of the author to search for.

### `llm`

Filters papers using a custom criterion with an LLM. The papers are processed in batches, which can be controlled with the `--llm-batch-size` argument.

**Syntax:** `--filter llm <criterion>`

*   `<criterion>`: The criterion to use for filtering. This will be sent to the LLM.

### `llm_from_file`

Filters papers using a criterion from a file. The papers are processed in batches, which can be controlled with the `--llm-batch-size` argument.

**Syntax:** `--filter llm_from_file <file_path>`

*   `<file_path>`: The path to the file containing the criterion.

### `or` logic

You can group filters with OR logic using `or_start` and `or_end`.

**Syntax:**
```bash
--filter or_start \
--filter <filter_1> \
--filter <filter_2> \
... \
--filter or_end
```

## Converting to BibTeX

You can convert an existing JSON results file to BibTeX format using the `--convert-to-bibtex` argument.

```bash
python main.py --convert-to-bibtex snowball_results.json
```

This will create a `snowball_results.bib` file in the same directory.

## Examples
```

### Basic Example

Find papers related to "machine learning" in the title, published after 2020.

```bash
python main.py \
    --initial-papers 10.1109/CVPR.2016.90 \
    --filter field title "machine learning" \
    --filter year gt 2020
```

### Using OR logic

Find papers with "attention" in the title OR "transformer" in the abstract.

```bash
python main.py \
    --initial-papers 10.1109/CVPR.2016.90 \
    --filter or_start \
    --filter field title "attention" \
    --filter field abstract "transformer" \
    --filter or_end
```

### Using the LLM filter

Find papers that are relevant to "explainable AI".

```bash
python main.py \
    --initial-papers 10.1109/CVPR.2016.90 \
    --filter llm "papers relevant to explainable AI"
```

### Using the `gemini-cli` tool

```bash
python main.py \
    --initial-papers 10.1109/CVPR.2016.90 \
    --llm-provider gemini-cli \
    --filter llm "papers relevant to explainable AI"
```