"""Tests for mock sweep behavior and triage I/O mapping."""

import csv
import pytest

from probes.behavioral.mock_client import MockClient
from probes.behavioral.css import compute_css
from probes.behavioral.cci import compute_cci
from scripts.run_contamination_triage import run_triage


def test_mock_client_complete_returns_expected_keys():
    """Test MockClient.complete() guarantees standard response objects mappings."""
    m = MockClient(response_map={"P1": "The answer is 5."})
    r = m.complete("P1", "some prompt")
    
    assert "response" in r
    assert "model" in r
    assert "problem_id" in r
    assert "prompt_tokens" in r
    assert "completion_tokens" in r
    
    assert r["model"] == "mock"
    assert r["response"] == "The answer is 5."


def test_mock_client_batch_length_preserved():
    """Test MockClient.complete_batch() accurately preserves indexing mapping lengths."""
    m = MockClient()
    problems = [{"problem_id": f"P{i}", "prompt": "prompt string"} for i in range(5)]
    results = m.complete_batch(problems)
    
    assert len(results) == 5
    assert all("response" in r for r in results)


def test_mock_client_default_response():
    """Test MockClient injects the global default when problem matches fail."""
    m = MockClient()
    r = m.complete("unknown_id", "prompt text")
    assert r["response"] == "The answer is 42."


def test_css_on_mock_responses():
    """Test compute_css parses Mock outputs correctly assigning float boundaries."""
    variant_responses = [
        {"variant_type": "W1", "model_answer": "42", "correct_answer": "42"},
        {"variant_type": "W2", "model_answer": "42", "correct_answer": "42"},
        {"variant_type": "W3", "model_answer": "42", "correct_answer": "42"},
    ]
    res = compute_css("P1", "42", variant_responses, "gsm")
    assert res["css"] == 1.0
    assert res["variants_evaluated"] == 3


def test_css_raises_on_w6():
    """Test compute_css intentionally crashes if mistakenly passed W6 Reversal variant flags."""
    variant_responses = [
        {"variant_type": "W6", "model_answer": "42", "correct_answer": "42"}
    ]
    with pytest.raises(ValueError):
        compute_css("P1", "42", variant_responses, "gsm")


def test_cci_on_matching_plan():
    """Test compute_cci matches generated planning tuples perfectly against execution states."""
    plan = ["move a from table to b", "move c from d to table"]
    res = compute_cci("P1", plan, plan)
    assert res["cci"] == 1.0


def test_cci_on_empty_inputs():
    """Test compute_cci reliably defaults to None without crashing against 0-length inputs."""
    res = compute_cci("P1", [], [])
    assert res["cci"] is None


def test_triage_writes_csv(tmp_path, monkeypatch):
    """Test run_triage parses problems directly to CSV structure blocking behavioral_correct flags."""
    input_file = tmp_path / "probe1_instances.csv"
    output_file = tmp_path / "contamination_triage.csv"
    
    # Write dummy input
    with input_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_id", "problem_text", "correct_answer", "problem_family"])
        writer.writeheader()
        writer.writerow({"problem_id": "P1", "problem_text": "text 1", "correct_answer": "a", "problem_family": "gsm"})
        writer.writerow({"problem_id": "P2", "problem_text": "text 2", "correct_answer": "b", "problem_family": "gsm"})
        
    def mock_score_problem(text):
        return {
            "max_ngram_length": 5, 
            "max_ngram_count": 10, 
            "contamination_score": 0.5
        }
        
    monkeypatch.setattr("scripts.run_contamination_triage.score_problem", mock_score_problem)
    
    run_triage(input_path=input_file, output_path=output_file, limit=2)
    
    assert output_file.exists()
    
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        assert len(rows) == 2
        assert "behavioral_correct" not in reader.fieldnames
