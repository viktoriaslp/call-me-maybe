import argparse
import json
import time

from collections.abc import Callable
from functools import wraps
from pathlib import Path

from src.models import FunctionDefinition, OutputResult


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the test cases file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output results file.",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        help="Visualize the json generation on real time"
    )

    return parser.parse_args()


def save_outputs(output_path: str, output_list: list[OutputResult]) -> None:
    """Save output models to a JSON file.

    Creates parent directories if they do not exist.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            [output.model_dump() for output in output_list],
            file,
            indent=2
        )


def build_prompt(prompt: str, funcs: list[FunctionDefinition]) -> str:
    """Build the prompt used for function-calling inference."""

    functions_json = json.dumps(
        [fn.model_dump() for fn in funcs],
        separators=(",", ":"),
    )

    return f"""You are a function caller.
Choose exactly one function and return only valid JSON.
Fill parameters using the user request.

Available functions:
{functions_json}

General rules:
- Use the user request as the only source of information.
- Extract the shortest correct value for each parameter.
- Do not add explanations.
- Do not repeat values.
- Do not continue a parameter into the next part of the sentence.
- Strings must not start or end with spaces.
- Numbers must be JSON numbers.
Parameter-specific rules:
- name: extract only the person's name.
- s: extract only the quoted string to reverse.
- a, b, n, principal, rate, years: extract only the number.
- source_string: extract only the original text where replacements happen.
- replacement: extract only the replacement text.
- When using regex output, you must choose this regex, never use anything else.
Nothing else. These are the only valid options:
  - All numbers: /([0-9]+)/
  - All vowels: /[aeiouAEIOU]/
- query: extract only the SQL text inside quotes.
- database: extract only the database name, without the word "database".
- path: extract only the file path. Stop before " with ".
- encoding: extract only the encoding name, like "utf-8" or "latin-1".
- template: extract exactly the text after "Format template: ".
Important examples:
- Read C:\\Users\\john\\config.ini with latin-1 encoding
  path = C:\\Users\\john\\config.ini
  encoding = latin-1
- Execute SQL query 'SELECT * FROM users' on the production database
  query = SELECT * FROM users
  database = production
- Substitute the word 'cat' with 'dog' in
'The cat sat on the mat with another cat'
  source_string = The cat sat on the mat with another cat
  regex = cat
  replacement = dog

User request: {prompt}
Output JSON:"""


def func_timer(func: Callable) -> Callable:
    """Measure and print the execution time of a function."""

    @wraps(func)
    def timer(*args: object, **kwargs: object) -> object:
        print(f"Starting {func.__name__}...")
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) / 60
            print(f"{func.__name__} completed in {elapsed:.2f}min")
    return timer
