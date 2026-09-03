*This project has been created as part of the 42 curriculum by vslyunko.*

# 📞 Call Me Maybe

Function calling with constrained decoding using a custom finite-state decoder.

## 🛠 Tech Stack

* Python 3.10+
* NumPy
* Pydantic
* uv
* pytest
* Qwen3-0.6B (llm_sdk)


## 📖 Description

This project implements a function-calling system powered by a small Language Model.

Instead of generating free-form text, the model produces structured JSON describing:

* which function should be called;
* which parameters should be passed;
* the correct type for every parameter.

To guarantee valid output, the generation process is constrained token by token using a custom finite-state decoder.

## ✨ Features

* Constrained decoding
* 100% valid JSON generation
* Schema-aware parameter validation
* Support for:
    * string
    * integer
    * number
    * boolean
* Graceful error handling
* Full type hints
* Pydantic validation
* Unit tests with pytest

## 📂 Project Structure

src/
├── decoder.py
├── generator.py
├── parser.py
├── models.py
├── utils.py
└── main.py
tests/
data/
├── input/
└── output/

## 🚀 Instructions

Install dependencies

make install

Run

make run

or

uv run python -m src

Run with custom files

uv run python -m src \
    --functions_definition path/to/functions.json \
    --input path/to/prompts.json \
    --output path/to/results.json
    --visualize True/False

Default paths

Argument	Default value
--functions_definition	data/input/functions_definition.json
--input	data/input/function_calling_tests.json
--output	data/output/function_calling_results.json
--visualize False

## ⚙️ Algorithm Explanation

The decoder generates the output one token at a time while ensuring that every generated token keeps the JSON valid.

Prompt
   │
   ▼
Build Prompt
   │
   ▼
Tokenize
   │
   ▼
State Machine
   │
   ▼
Compute Allowed Tokens
   │
   ▼
Mask Invalid Logits
   │
   ▼
Greedy Selection
   │
   ▼
Append Token
   │
   ▼
Advance State
   │
   └───────────────┐
                   │
                Until DONE

Generation loop

For every generated token:

1. Encode the prompt.
2. Determine the current decoder state.
3. Compute the set of valid next tokens.
4. If only one token is possible, emit it directly.
5. Otherwise:
    * obtain the model logits;
    * mask every invalid token;
    * select the highest remaining logit.
6. Append the selected token.
7. Update the decoder state.
8. Repeat until the JSON object is complete.

The decoder never allows tokens that would break either the JSON syntax or the expected schema.

## 💡 Design Decisions

Decision	Reason
Finite-state machine	Keeps the JSON structure under control during generation
Greedy decoding	Fast and deterministic
NumPy for logit masking	Efficient filtering of valid tokens
Token cache	Avoid repeated tokenization of identical strings
Skip logits when only one token is valid	Reduces unnecessary LLM computations
Pydantic models	Automatic validation and cleaner data structures

Performance was prioritised over prompt complexity. Instead of relying on a long prompt, the implementation uses constrained decoding to guarantee correct output.

## 📈 Performance Analysis

Accuracy

* Function selection is performed by the LLM.
* JSON validity is guaranteed by constrained decoding.
* Parameter types always follow the function schema.

Speed

Several optimisations were introduced:

* logits are only computed when multiple token choices exist;
* repeated tokenization is avoided through caching;
* NumPy is used for efficient masking operations.

These optimisations significantly reduce the amount of work performed during generation.

Reliability

Every generated token satisfies both:

* valid JSON syntax;
* the expected function schema.

As a result, every generated output can always be parsed successfully.

## 🚧 Challenges Faced

Understanding constrained decoding

The biggest challenge was understanding how constrained decoding works internally. Coming from a C background, concepts such as logits, tokenization and token-level generation were completely new.

Detecting the end of strings

The most difficult implementation detail was determining when a generated string was actually complete while correctly handling escaped characters.

Writing Python instead of C

Several parts of the implementation were redesigned to follow a more Pythonic approach instead of my initial C-style solutions.

## 🧪 Testing Strategy

The implementation was validated using:

* unit tests with pytest;
* malformed JSON files;
* missing input files;
* invalid schemas;
* prompts containing different parameter types;
* manual inspection of generated outputs.

Every generated JSON object is validated again using Pydantic before being written to the output file.

## ▶️ Example Usage

Input

What is the sum of 2 and 3?

Output

{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2,
        "b": 3
    }
}

Another example

Input

Greet John

Output

{
    "prompt": "Greet John",
    "name": "fn_greet",
    "parameters": {
        "name": "John"
    }
}

## 📚 Resources

Constrained Decoding & LLMs

* Controlling your LLM: Deep Dive into Constrained Generation
    https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20
* LLM Breakdown: Logits and Next-Token Prediction
    https://mikexcohen.substack.com/p/llm-breakdown-26-logits-and-next
* Constrained Decoding: Grammar-Guided Generation for Structured Output
    https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output
* Function Calling Internals: Grammars and Constrained Decoding
    https://www.salmanq.com/blog/llm-constrained-sampling/

Python

* pytest
    https://www.youtube.com/watch?v=mzlH8lp4ISA
* argparse
    https://www.youtube.com/watch?v=88pl8TuuKz0
* argparse (additional tutorial)
    https://www.youtube.com/watch?v=cdblJqEUDNo
* JSON module
    https://www.youtube.com/watch?v=4rmBOxn0PdI

Additional documentation consulted:

* Pydantic
* list comprehensions
* error handling
* uv package manager

## 🤖 AI Usage

AI was used as a learning assistant, not as a code generator.

It helped with:

* understanding concepts that were difficult to find in documentation;
* improving function, variable and class names;
* suggesting more Pythonic alternatives to C-style code;
* identifying repeated code that could be simplified;
* discussing implementation ideas when I was blocked.

Every suggestion was reviewed, understood and adapted before being included in the final implementation.