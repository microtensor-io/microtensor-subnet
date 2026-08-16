from __future__ import annotations

import pytest

from microtensor.harness.limits import Limits, sandbox_available
from microtensor.scoring import execution
from microtensor.scoring.execution import (
    ExecutionUnavailable,
    execute_pass_rate,
    extract_code,
    parse_tests,
    values_match,
)
from microtensor.scoring.execution import TestCase as Case

TESTS = (
    Case(args=("",), expected=0),
    Case(args=("abc",), expected=3),
    Case(args=("hello",), expected=5),
    Case(args=("aa",), expected=2),
)

GOOD = "def measure(s):\n    return len(s)\n"
OFF_BY_ONE = "def measure(s):\n    return len(s) + (1 if s else 0)\n"
SPINNER = "def measure(s):\n    while True:\n        pass\n"
SMUGGLER = "import socket\ndef measure(s):\n    return len(s)\n"
ESCAPER = "def measure(s):\n    open('/etc/hostname').read()\n    return len(s)\n"


@pytest.fixture(autouse=True)
def _unsandboxed_ok() -> None:
    execution.configure(allow_unsandboxed=True)
    yield
    execution.configure(allow_unsandboxed=False)


def _tight() -> Limits:
    return Limits(cpu_seconds=1, wall_seconds=4, rss_bytes=256 * 1024**2)


def test_a_correct_solution_scores_one() -> None:
    assert execute_pass_rate(GOOD, "measure", TESTS) == 1.0


def test_an_off_by_one_scores_the_exact_fraction() -> None:
    assert execute_pass_rate(OFF_BY_ONE, "measure", TESTS) == pytest.approx(1 / 4)


def test_scores_are_identical_across_runs() -> None:
    first = execute_pass_rate(OFF_BY_ONE, "measure", TESTS)
    second = execute_pass_rate(OFF_BY_ONE, "measure", TESTS)
    assert first == second


def test_an_infinite_loop_scores_zero_without_stalling() -> None:
    assert execute_pass_rate(SPINNER, "measure", TESTS, limits=_tight()) == 0.0


def test_a_forbidden_import_scores_zero() -> None:
    assert execute_pass_rate(SMUGGLER, "measure", TESTS) == 0.0


def test_an_escape_through_open_fails_that_test() -> None:
    assert execute_pass_rate(ESCAPER, "measure", TESTS) == 0.0


def test_code_that_does_not_compile_scores_zero() -> None:
    assert execute_pass_rate("def broken(:\n", "measure", TESTS) == 0.0


def test_a_missing_entry_point_scores_zero() -> None:
    assert execute_pass_rate("def other(s):\n    return 1\n", "measure", TESTS) == 0.0


def test_a_whitelisted_import_is_allowed() -> None:
    code = "import math\ndef measure(s):\n    return int(math.floor(len(s)))\n"
    assert execute_pass_rate(code, "measure", TESTS) == 1.0


def test_no_tests_means_no_score() -> None:
    assert execute_pass_rate(GOOD, "measure", ()) == 0.0


def test_a_sandboxless_host_abstains_rather_than_scoring() -> None:
    if sandbox_available():
        pytest.skip("this host can enforce limits")
    execution.configure(allow_unsandboxed=False)
    with pytest.raises(ExecutionUnavailable):
        execute_pass_rate(GOOD, "measure", TESTS)


def test_extract_code_prefers_the_fenced_block() -> None:
    wrapped = "Here you go:\n```python\ndef measure(s):\n    return len(s)\n```\nEnjoy."
    assert extract_code(wrapped) == "def measure(s):\n    return len(s)"
    assert extract_code(GOOD) == GOOD.strip()


def test_fenced_output_scores_like_bare_output() -> None:
    wrapped = f"```python\n{GOOD}```"
    assert execute_pass_rate(wrapped, "measure", TESTS) == 1.0


def test_parse_tests_rejects_a_malformed_case() -> None:
    with pytest.raises(ValueError):
        parse_tests([{"args": [1]}])


def test_numeric_comparison_is_by_value_not_repr() -> None:
    assert values_match(2.0000000001, 2.0)
    assert values_match(2, 2.0)
    assert not values_match(True, 1)
    assert values_match([1, [2, 3]], [1, [2, 3]])
    assert not values_match({"a": 1}, {"a": 1, "b": 2})


def test_prints_do_not_break_scoring() -> None:
    noisy = "def measure(s):\n    print('thinking')\n    return len(s)\n"
    assert execute_pass_rate(noisy, "measure", TESTS) == 1.0
