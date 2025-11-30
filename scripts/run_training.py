import logging

from petface.models.face_identification.res_net.model import get_reidentification_model


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


def main(animals: list[str]) -> None:
    for animal in animals:
        for loss in LOSSES:
            for classification in CLASSIFICATIONS:
                try:
                    logger.info(f"Training network for animal={animal!r}, loss={loss!r}, classification={classification!r}")

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
    # "guineapig",
    # "chinchilla",
    # "degus",
    # "ferret",
    # "hamster",
    # "hedgehog",
    # "javasparrow",
    # "parakeet",
    # "pig",
    # "rabbit"
]


if __name__ == "__main__":
    main(animals=ANIMALS)
