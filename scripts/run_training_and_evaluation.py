from scripts import run_training, run_evaluation


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
    "rabbit",
    "dog",
    "cat"
]

FORCE = False

if __name__ == "__main__":
    run_training.main(animals=ANIMALS, force=FORCE)
    run_evaluation.main(animals=ANIMALS)


# 12839