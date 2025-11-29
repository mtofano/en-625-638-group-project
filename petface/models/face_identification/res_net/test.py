import pathlib
import keras
import argparse
import tensorflow as tf

from petface.const import IMG_HEIGHT, IMG_WIDTH
from petface.models.face_identification.res_net.model import get_model, get_embedding_model, ModelLoss


IMG_DIMENSIONS = (IMG_HEIGHT, IMG_WIDTH)


def evaluate_accuracy(
    dataset: tf.data.Dataset,
    num_classes: int,
    model_loss: ModelLoss,
    weights_file: pathlib.Path
) -> None:
    model = get_model(
        num_classes=num_classes,
        model_loss=model_loss
    )

    model.load_weights(weights_file)

    print(model.summary())

    loss, acc, top5_acc, top50_acc = model.evaluate(dataset)
    
    print(f"Loss={loss}, Accuracy={acc}, Top 5 Accuracy={top5_acc}, Top 50 Accuracy={top50_acc}")


def evaluate_embeddings(
    dataset: tf.data.Dataset,
    num_classes: int,
    model_loss: ModelLoss,
    weights_file: pathlib.Path
) -> None:
    full_model = get_model(
        num_classes=num_classes,
        model_loss=model_loss
    )

    full_model.load_weights(weights_file)

    emb_seq = full_model.get_layer("sequential")

    embedding_output = emb_seq.layers[-1].output

    embedding_model = keras.Model(
        inputs=full_model.inputs[0],
        outputs=embedding_output
    )

    embeddings = embedding_model.predict(dataset)

    x = 5


def main(
    animal: str,
    model_loss: ModelLoss,
    batch_size: int,
    root_dir: pathlib.Path
) -> None:
    train_dir = root_dir / "dataset/out/images/train" / animal
    test_dir = root_dir / "dataset/out/images/test" / animal
    weights_file = root_dir / ".checkpoints" / animal / "res_net" / model_loss / "cp.weights.h5"

    train_ds = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    test_ds = keras.utils.image_dataset_from_directory(
        directory=test_dir,
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    num_classes = len(test_ds.class_names)  # type: ignore

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # type: ignore
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)  # type: ignore

    # evaluate_accuracy(
    #     dataset=test_ds,
    #     num_classes=num_classes,
    #     model_loss=model_loss,
    #     weights_file=weights_file
    # )

    evaluate_embeddings(
        dataset=test_ds,
        num_classes=num_classes,
        model_loss=model_loss,
        weights_file=weights_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--animal", required=True, type=str)
    parser.add_argument("--loss", required=True, type=str)
    parser.add_argument("--batch-size", required=False, default=32, type=int)
    parser.add_argument("--root-dir", required=False, default="/home/mtofano/src/en-625-638-group-project", type=str)

    arguments = parser.parse_args()

    main(
        animal=arguments.animal,
        model_loss=arguments.loss,
        batch_size=arguments.batch_size,
        root_dir=pathlib.Path(arguments.root_dir)
    )
