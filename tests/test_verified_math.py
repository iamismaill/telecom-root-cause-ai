from telecom_rca.data import load_current_csv
from telecom_rca.verified_math import solve_verified_math


EXPECTED = {
    "ID_4AI7CVVV8Q": "4",
    "ID_A982UKJRXP": "4",
    "ID_BTFBC72895": "3",
    "ID_B95M5YVWPJ": "4",
    "ID_4POTXVDAAL": "3",
    "ID_WURGWFHNJW": "2",
    "ID_QX6QVSBXGR": "4",
    "ID_XX3ELOVQS6": "3",
    "ID_JEWV5YMC2P": "1",
    "ID_AIM9X6ECCC": "2",
    "ID_NKKHCQD42E": "2",
    "ID_HWRIOUB2KS": "1",
    "ID_HUD956GTVM": "1",
    "ID_LQ1NC0X9EV": "3",
    "ID_UK5OBYW14N": "4",
}


def test_reusable_solvers_prove_expected_official_questions():
    test = load_current_csv("test.csv").set_index("ID")
    for question_id, expected in EXPECTED.items():
        answer = solve_verified_math(test.loc[question_id, "question"])
        assert answer is not None, question_id
        assert answer.label == expected, (question_id, answer)


def test_unrecognized_non_math_question_is_not_overridden():
    assert solve_verified_math("Who wrote Hamlet?\n1: Shakespeare\n2: Dickens") is None
