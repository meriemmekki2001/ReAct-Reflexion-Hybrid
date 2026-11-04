from agent_bench.benchmarks.metrics import exact_match, token_f1, numeric_accuracy
from agent_bench.tools.calculator import calculate
from agent_bench.tools.documents import DocumentEnvironment


def test_hotpot_metrics():
    assert exact_match('The Eiffel Tower', 'Eiffel Tower') == 1
    assert token_f1('Barack Obama', 'Obama') > 0
    assert exact_match('yes', 'no') == 0


def test_gsm8k_numeric():
    assert numeric_accuracy('Answer: $1,250', '1250') == 1
    assert numeric_accuracy('12 apples', '12') == 0


def test_calculator():
    assert calculate('(12 * 3) + 4') == '40'
    assert 'error' in calculate('1 / 0').lower()


def test_empty_documents():
    env = DocumentEnvironment([])
    assert 'No documents' in env.search('test')
