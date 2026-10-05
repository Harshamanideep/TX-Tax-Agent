from src.tools import calculate, mod11_check_digit, validate_check_digit


def test_calculate_basic():
    assert calculate.invoke({"expression": "52000 * 0.045"}) == "52000 * 0.045 = 2340.0"


def test_calculate_rejects_code():
    assert "Only numbers" in calculate.invoke({"expression": "__import__('os')"})


def test_calculate_blocks_huge_exponent():
    assert "too large" in calculate.invoke({"expression": "9**9**9"})


def test_calculate_divide_by_zero():
    assert "division by zero" in calculate.invoke({"expression": "1/0"})


def test_check_digit_round_trip():
    body = "12345678"
    digit = mod11_check_digit(body)
    assert "VALID" in validate_check_digit.invoke({"number": body + str(digit)})
    assert "INVALID" in validate_check_digit.invoke({"number": body + str((digit + 1) % 10)})


def test_check_digit_ignores_dashes():
    assert "VALID" in validate_check_digit.invoke({"number": "1234-56785"})
