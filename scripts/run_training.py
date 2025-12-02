import logging

from petface.const import ROOT_DIR
from petface.face_identification.model import get_reidentification_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)


LOSSES: list[str] = [
    "cross_entropy",
    "arcface"
]


CLASSIFICATIONS: list[str] = [
    "softmax",
    "cosine_similarity"
]


def main(animals: list[str], *, force: bool = True) -> None:
    for animal in animals:
        for loss in LOSSES:
            for classification in CLASSIFICATIONS:
                try:
                    logger.info(f"Training network for animal={animal!r}, loss={loss!r}, classification={classification!r}")

                    if not force:
                        history_file = ROOT_DIR / ".out" / animal / loss / classification / "history" / "history.json"
                        if history_file.exists():
                            logger.info(f"History file [{history_file}] exists and force={force} - Not running again")
                            continue

                    model = get_reidentification_model(
                        animal=animal,
                        loss_type=loss,
                        classification_type=classification
                    )
                    
                    model.train()
                except Exception:
                    logger.exception("Encountered an error training models")


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
