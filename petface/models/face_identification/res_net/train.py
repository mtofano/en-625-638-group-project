import argparse

from petface.models.face_identification.res_net.v2 import get_reidentification_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--animal", required=True, type=str)
    parser.add_argument("--loss", required=True, type=str)
    parser.add_argument("--classification", required=True, type=str)

    arguments = parser.parse_args()

    model = get_reidentification_model(
        animal=arguments.animal,
        loss_type=arguments.loss,
        classification_type=arguments.classification,
    )

    model.train()
