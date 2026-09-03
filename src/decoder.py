import json
from enum import Enum

import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, PrivateAttr

from src.models import FunctionDefinition, VarType


class State(Enum):
    START = "start"
    EXPECT_PROMPT = "expect_prompt"
    EXPECT_NAME = "expect_name"
    EXPECT_FUNCTION_NAME = "expect_function_name"
    EXPECT_PARAMETERS = "expect_parameters"
    EXPECT_PARAMETER_NAME = "expect_parameter_name"
    EXPECT_PARAMETER_VALUE = "expect_parameter_value"
    INSIDE_STRING = "inside_string"
    EXPECT_PARAMETER_SEPARATOR_OR_END = "expect_parameter_separator_or_end"
    EXPECT_END = "expect_end"
    DONE = "done"


class DecoderMachine(BaseModel):
    """Finite-state machine that constrains LLM generation to a valid
    function-call JSON.

    The decoder exposes, for each generation step, the set of valid next
    tokens according to the current parser state and the selected function
    schema.
    """

    _model: Small_LLM_Model = PrivateAttr()
    vocab_size: int = 0
    func_definitions: list[FunctionDefinition] = Field(default_factory=list)
    prompt: str

    current_state: State = State.START
    current_json: str = ""
    current_function: FunctionDefinition | None = None
    parameter_index: int = 0

    position: int = 0
    fixed_tokens: list[int] = Field(default_factory=list)
    current_choice_prefix: str = ""

    _token_cache: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _token_text_cache: dict[int, str] = PrivateAttr(default_factory=dict)
    _safe_string_token_ids_cache: np.ndarray | None = PrivateAttr(default=None)

    def _encode(self, text: str) -> list[int]:
        if text not in self._token_cache:
            encoded = self._model.encode(text).tolist()[0]
            self._token_cache[text] = encoded
        return self._token_cache[text]

    def token_text(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self._model.decode([token_id])
        return self._token_text_cache[token_id]

    def allowed_function_names(self) -> list[str]:
        return [json.dumps(f.name) for f in self.func_definitions]

    def current_param_name(self) -> str:
        if self.current_function is None:
            raise ValueError("No current function selected.")
        parameter_names = list(self.current_function.parameters.keys())
        return parameter_names[self.parameter_index]

    def current_param_type(self) -> VarType:
        assert self.current_function is not None
        return self.current_function.parameters[self.current_param_name()].type

    def still_has_parameters(self) -> bool:
        assert self.current_function is not None
        return self.parameter_index < len(
            list(self.current_function.parameters.keys())
        ) - 1

    def is_fixed_text_state(self) -> bool:
        return self.current_state in {
            State.START,
            State.EXPECT_PROMPT,
            State.EXPECT_NAME,
            State.EXPECT_PARAMETERS,
            State.EXPECT_PARAMETER_NAME,
            State.EXPECT_END,
        }

    def allowed_tokens(self) -> np.ndarray:
        """Return token ids that are valid in the current decoding state."""

        if self.is_fixed_text_state():
            return self.allowed_tokens_for_fixed_state()

        if self.current_state == State.EXPECT_FUNCTION_NAME:
            return self.allowed_tokens_for_function_name()

        if self.current_state == State.EXPECT_PARAMETER_VALUE:
            return self.allowed_tokens_for_parameter_value()

        if self.current_state == State.INSIDE_STRING:
            return self.allowed_tokens_inside_string()

        if self.current_state == State.EXPECT_PARAMETER_SEPARATOR_OR_END:
            return np.array(self.parameter_end_tokens(), dtype=np.int32)

        return np.array([], dtype=np.int32)

    def fixed_text_for_state(self) -> str | None:
        """Return the fixed JSON fragment expected for the current state.

        Returns:
            Literal text that must be emitted next, or ``None`` if the state
            allows dynamic token generation.
        """

        if self.current_state == State.START:
            return "{"

        if self.current_state == State.EXPECT_PROMPT:
            return f'"prompt": {json.dumps(self.prompt)},'

        if self.current_state == State.EXPECT_NAME:
            return '"name": '

        if self.current_state == State.EXPECT_PARAMETERS:
            return ',"parameters": {'

        if self.current_state == State.EXPECT_PARAMETER_NAME:
            param_name = self.current_param_name()
            return json.dumps(param_name) + ":"

        value_state = self.current_state == State.EXPECT_PARAMETER_VALUE
        value_type = self.current_param_type() == VarType.STRING
        if value_state and value_type:
            return '"'

        if self.current_state == State.EXPECT_END:
            return "}"

        return None

    def allowed_tokens_for_fixed_state(self) -> np.ndarray:
        fixed_text = self.fixed_text_for_state()
        if fixed_text is None:
            return np.array([], dtype=np.int32)

        tokens = self._encode(fixed_text)
        if self.position < len(tokens):
            return np.array([tokens[self.position]], dtype=np.int32)
        else:
            return np.array([], dtype=np.int32)

    def next_tokens_from_token_options(
            self, token_options: list[list[int]]
    ) -> list[int] | None:
        """Return the valid next token for each candidate sequence matching the
        current decoded prefix.

        Args:
            token_options: Candidate tokenized sequences.

        Returns:
            The next valid token ids, or ``None`` if no sequence matches the
            current prefix.
        """

        next_tokens: list[int] = []
        for option in token_options:
            if self.position >= len(option):
                continue
            decoded_prefix = self._model.decode(option[:self.position])
            if decoded_prefix != self.current_choice_prefix:
                continue
            next_tokens.append(option[self.position])
        return next_tokens if next_tokens else None

    def allowed_tokens_for_function_name(self) -> np.ndarray:
        function_names = self.allowed_function_names()
        function_tokens = [self._encode(name)for name in function_names]
        allowed_tokens = self.next_tokens_from_token_options(function_tokens)
        if allowed_tokens is None:
            return np.array([], dtype=np.int32)
        else:
            return np.array(allowed_tokens, dtype=np.int32)

    def allowed_tokens_for_boolean(self) -> np.ndarray:
        if self.current_choice_prefix in {"true", "false"}:
            return np.array(self.parameter_end_tokens(), dtype=np.int32)

        boolean_options = [
            self._encode("true"),
            self._encode("false"),
        ]

        bool_tokens = self.next_tokens_from_token_options(boolean_options)
        if bool_tokens is None:
            return np.array([], dtype=np.int32)
        else:
            return np.array(bool_tokens, dtype=np.int32)

    def parameter_end_tokens(self) -> list[int]:
        if self.still_has_parameters():
            return self._encode(",")
        return self._encode("}")

    def allowed_tokens_for_numeric_value(
        self, allow_decimal: bool,
    ) -> np.ndarray:
        tokens: list[int] = []

        for char in "0123456789":
            tokens.extend(self._encode(char))

        if self.current_choice_prefix == "":
            tokens.extend(self._encode("-"))

        if allow_decimal and "." not in self.current_choice_prefix:
            tokens.extend(self._encode("."))

        tokens.extend(self.parameter_end_tokens())

        return np.array(tokens, dtype=np.int32)

    def allowed_tokens_for_number(self) -> np.ndarray:
        if len(self.current_choice_prefix) > 40:
            return np.array(self._encode('"'), dtype=np.int32)

        return self.allowed_tokens_for_numeric_value(allow_decimal=True)

    def allowed_tokens_for_integer(self) -> np.ndarray:
        return self.allowed_tokens_for_numeric_value(allow_decimal=False)

    def allowed_tokens_for_parameter_value(self) -> np.ndarray:
        param_type = self.current_param_type()

        if param_type == VarType.STRING:
            return self.allowed_tokens_for_fixed_state()
        if param_type == VarType.NUMBER:
            return self.allowed_tokens_for_number()

        if param_type == VarType.INTEGER:
            return self.allowed_tokens_for_integer()

        if param_type == VarType.BOOLEAN:
            return self.allowed_tokens_for_boolean()

        return np.array([], dtype=np.int32)

    def allowed_tokens_inside_string(self) -> np.ndarray:
        """Return all token ids that keep the generated JSON string valid."""

        if len(self.current_choice_prefix) > 30:
            return np.array(self._encode('"'), dtype=np.int32)

        if self._safe_string_token_ids_cache is None:
            safe_string_token_ids = []

            for token_id in range(self.vocab_size):
                if self.safe_token(token_id):
                    safe_string_token_ids.append(token_id)

            safe_string_token_ids.extend(self._encode('"'))

            self._safe_string_token_ids_cache = np.asarray(
                safe_string_token_ids,
                dtype=np.int32
            )

        return self._safe_string_token_ids_cache

    def safe_token(self, id_token: int) -> bool:
        """Check whether a token can safely appear inside a JSON string.

        Rejects tokens that would produce invalid escaping,
        embedded newlines or malformed quotation marks.
        """

        text = self.token_text(id_token)
        if "\n" in text or "\r" in text or "\\w" in text:
            return False
        if text.count("\\") % 2 != 0:
            if '\\"' not in text:
                return False
        if '"' in text:
            if not text.endswith('"'):
                return False

            if text.count('"') > 1:
                return False

        return True

    def advance_state(self) -> None:
        transitions = {
            State.START: State.EXPECT_PROMPT,
            State.EXPECT_PROMPT: State.EXPECT_NAME,
            State.EXPECT_NAME: State.EXPECT_FUNCTION_NAME,
            State.EXPECT_PARAMETERS: State.EXPECT_PARAMETER_NAME,
            State.EXPECT_PARAMETER_NAME: State.EXPECT_PARAMETER_VALUE,
            State.EXPECT_END: State.DONE,
        }

        if self.current_state in transitions:
            self.current_state = transitions[self.current_state]

    def advance_fixed_state_if_needed(self) -> None:
        self.position += 1
        fixed_text = self.fixed_text_for_state()
        if fixed_text is None:
            return
        fixed_tokens = self._encode(fixed_text)

        if self.position < len(fixed_tokens):
            return

        self.position = 0
        self.advance_state()

    def finish_function_name(self) -> None:
        """Finalize the selected function and initialize parameter parsing."""

        selected_name = json.loads(self.current_choice_prefix)

        self.current_function = next(
            f for f in self.func_definitions
            if f.name == selected_name
        )

        self.parameter_index = 0
        self.current_choice_prefix = ""
        self.current_state = State.EXPECT_PARAMETERS

    def finish_parameter_value(self) -> None:
        """Finish parsing the current parameter value."""

        self.current_choice_prefix = ""
        self.position = 0
        self.current_state = State.EXPECT_PARAMETER_SEPARATOR_OR_END

    def advance_parameter_separator_or_end(self, decoded_token: str) -> None:
        """Handle the separator following a parameter value."""

        if decoded_token == ",":
            self.parameter_index += 1
            self.current_state = State.EXPECT_PARAMETER_NAME
        elif decoded_token == "}":
            self.current_state = State.EXPECT_END
        else:
            raise ValueError(f"Unexpected separator token: {decoded_token!r}")
        self.position = 0

    def advance_function_name_if_needed(self, decoded_token: str) -> None:
        self.position += 1
        self.current_choice_prefix += decoded_token

        if self.current_choice_prefix not in self.allowed_function_names():
            return

        self.position = 0
        self.finish_function_name()

    def advance_parameter_value_if_needed(self, decoded_token: str) -> None:
        param_type = self.current_param_type()

        if param_type == VarType.STRING:
            self.current_state = State.INSIDE_STRING
            self.current_choice_prefix = ""
            return

        if decoded_token in {",", "}"}:
            self.advance_parameter_separator_or_end(decoded_token)
            self.current_choice_prefix = ""
            return

        self.current_choice_prefix += decoded_token

        if param_type == VarType.BOOLEAN:
            self.position += 1

    def advance_inside_string(self, decoded_token: str) -> None:
        if decoded_token.endswith('"') and not decoded_token.endswith('\\"'):
            self.current_choice_prefix += decoded_token
            self.finish_parameter_value()
            return
        self.current_choice_prefix += decoded_token

    def advance_state_if_needed(self, decoded_token: str) -> None:
        """Update the decoder state after accepting a generated token."""

        if self.is_fixed_text_state():
            self.advance_fixed_state_if_needed()
            return

        if self.current_state == State.EXPECT_FUNCTION_NAME:
            self.advance_function_name_if_needed(decoded_token)
            return

        if self.current_state == State.EXPECT_PARAMETER_VALUE:
            self.advance_parameter_value_if_needed(decoded_token)
            return

        if self.current_state == State.INSIDE_STRING:
            self.advance_inside_string(decoded_token)
            return

        if self.current_state == State.EXPECT_PARAMETER_SEPARATOR_OR_END:
            self.advance_parameter_separator_or_end(decoded_token)
            return
