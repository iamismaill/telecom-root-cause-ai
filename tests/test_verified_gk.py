from telecom_rca.data import load_current_csv
from telecom_rca.routing import Route, route_question
from telecom_rca.verified_gk import VERIFIED_GK_CORRECTIONS


def test_verified_gk_corrections_are_unique_valid_choices() -> None:
    test = load_current_csv("test.csv").set_index("ID")
    assert len(VERIFIED_GK_CORRECTIONS) == 15
    for identifier, verified in VERIFIED_GK_CORRECTIONS.items():
        decision = route_question(str(test.loc[identifier, "question"]))
        assert decision.route == Route.GENERAL_KNOWLEDGE
        assert verified.label in {option.label for option in decision.options}
        assert verified.proof
