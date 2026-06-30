"""
Smoke test for the Marginal API. Run the server first (python app.py),
then run this script in another terminal: python test_app.py
"""

import requests

BASE_URL = "http://127.0.0.1:5000"
NOTEBOOK = "test_notebook"


def test_upload():
    with open("apollo.txt", "w") as f:
        f.write(
            "Project Apollo was a series of NASA missions that landed humans on the Moon. "
            "The first crewed landing was Apollo 11 in July 1969, commanded by Neil Armstrong."
        )
    with open("errors.txt", "w") as f:
        f.write(
            "Error code SYS-9942 indicates a tracking anomaly in the test rig and must be "
            "cleared before the next launch attempt."
        )

    with open("apollo.txt", "rb") as f1, open("errors.txt", "rb") as f2:
        res = requests.post(
            f"{BASE_URL}/api/sources/upload",
            data={"notebook": NOTEBOOK},
            files=[("files", ("apollo.txt", f1, "text/plain")),
                   ("files", ("errors.txt", f2, "text/plain"))],
        )
    print("Upload:", res.status_code, res.json())
    assert res.status_code == 200


def test_chat_with_citations():
    res = requests.post(
        f"{BASE_URL}/api/chat",
        json={"notebook": NOTEBOOK, "query": "What is the status of code SYS-9942?", "top_k": 3},
    )
    print("Chat:", res.status_code, res.json())
    assert res.status_code == 200


def test_sources_list():
    res = requests.get(f"{BASE_URL}/api/sources", params={"notebook": NOTEBOOK})
    print("Sources:", res.status_code, res.json())
    assert res.status_code == 200


if __name__ == "__main__":
    test_upload()
    test_chat_with_citations()
    test_sources_list()
