import papermill as pm
import logging

from petface.const import ROOT_DIR


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "model_test_report.ipynb"


def main(animals: list[str]) -> None:
    for animal in animals:
        output_path = ROOT_DIR / "reports" / f"{animal}.ipynb"

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            pm.execute_notebook(
                input_path=NOTEBOOK_PATH,
                output_path=output_path,
                parameters=dict(ANIMAL="chinchilla")
            )
        except Exception:
            logger.exception("Encounterd an error evaluating models")
            


ANIMALS = [
    "chimp",
    "guineapig",
    "chinchilla",
    "degus",
    "ferret",
    "hamster",
    "hedgehog",
    "javasparrow",
    "parakeet",
    "pig",
    "rabbit"
]


if __name__ == "__main__":
    main(animals=ANIMALS)
