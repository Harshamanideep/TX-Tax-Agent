"""Tools the agent can call. Each docstring is what the LLM reads to decide
when to use the tool, so keep them clear and specific."""
import ast
import operator

from langchain_core.tools import tool

from src import config
from src.rag import format_context, get_store, hybrid_search


@tool
def search_tax_instructions(query: str, state: str | None = None) -> str:
    """Search the loaded state tax form instructions.
    Use for any question about due dates, rules, line instructions, credits or
    where to file. `state` is an optional two-letter code such as "UT" to limit
    the search. Returns text passages labelled [STATE form, p.N] for citing."""
    docs = hybrid_search(query, state, config.TOP_K)
    if not docs:
        return "No matching passages found."
    return format_context(docs)


@tool
def list_loaded_forms() -> str:
    """List which states and forms are loaded. Use when the user asks what you
    can answer about, or before searching if you are unsure a form is loaded."""
    metas = get_store().get(include=["metadatas"])["metadatas"]
    forms = sorted({f"{m['state']} {m['form']}" for m in metas})
    return "Loaded forms: " + ", ".join(forms) if forms else "No forms loaded."


# --- Calculator -------------------------------------------------------------
# Evaluates arithmetic safely by walking the expression tree instead of using
# eval(), so the LLM can't run arbitrary Python.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")  # stops "9**9**9" freezing the program
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** ( ) are allowed.")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "52000 * 0.0455" or "(1200 - 300) / 2".
    ALWAYS use this for any arithmetic instead of calculating yourself.
    Do not include commas, $ or % signs: write 4.55% as 0.0455."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Could not calculate '{expression}': {e}"
    return f"{expression} = {round(result, 4)}"


# --- Modulus 11 check digit ---------------------------------------------------
# A common Modulus 11 scheme: multiply digits by weights 2, 3, 4 ... 7, repeating,
# starting from the rightmost digit; check digit = (11 - sum % 11) % 11,
# with 10 meaning invalid. Agencies use different variants, so check the
# specific published spec before relying on it for a real form.
def mod11_check_digit(digits: str) -> int | None:
    total = sum(int(d) * (2 + i % 6) for i, d in enumerate(reversed(digits)))
    check = (11 - total % 11) % 11
    return None if check == 10 else check


@tool
def validate_check_digit(number: str) -> str:
    """Validate a number whose LAST digit is a Modulus 11 check digit (weights
    2-7 from the right), as used on payment scanlines. Spaces and dashes are ignored."""
    digits = "".join(c for c in number if c.isdigit())
    if len(digits) < 2:
        return "Need at least 2 digits."
    body, given = digits[:-1], int(digits[-1])
    expected = mod11_check_digit(body)
    if expected is None:
        return f"{number}: no valid Modulus 11 check digit exists for {body}."
    if expected == given:
        return f"{number}: VALID (check digit {given})."
    return f"{number}: INVALID. Check digit is {given}, expected {expected}."


TOOLS = [search_tax_instructions, list_loaded_forms, calculate, validate_check_digit]