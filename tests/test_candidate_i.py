import pytest

from telecom_rca.candidate_i import ConsensusChoiceBackend, TrainingCaseRetriever


class SequenceBackend:
    def __init__(self, answers):
        self.answers = iter(answers)

    def answer(self, question, allowed):
        answer = next(self.answers)
        assert answer in allowed
        return answer


def test_consensus_accepts_quorum() -> None:
    backend = ConsensusChoiceBackend(SequenceBackend(["A", "B", "A"]))
    assert backend.answer("question", {"A", "B"}) == "A"


def test_consensus_rejects_three_way_disagreement() -> None:
    backend = ConsensusChoiceBackend(SequenceBackend(["A", "B", "C"]))
    with pytest.raises(ValueError, match="No Qwen consensus"):
        backend.answer("question", {"A", "B", "C"})


def test_retriever_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TrainingCaseRetriever().predict("question")
