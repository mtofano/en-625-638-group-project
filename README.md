# JHU EN 625 638 Group Project - PetFace

## Python Environment Setup

To create your Python environment for development, you need [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

Run the following command to create the `petface` environment:

```sh
conda env create -f environment.yml
```

To activate the environment:

```sh
conda activate petface
```

## PetFace Download

To download the PetFace dataset from the Google Drive share follow these steps:

1. Install command line tool [rclone](https://rclone.org/install/)
2. Run `rclone config` to add a new remote connection
3. Call the name of the remote `google-drive` when prompted
4. Enter `22`(or `drive`) to select the Google Drive setup
5. Follow the prompts, most of which you can just press `Enter` and leave blank, but make sure for the `Option scope` selection you choose `2`

Once the steps above are complete, you should now have a now rclone config set up to connect to Google Drive.

To download the dataset, run the following commands in the root of this directory:

```sh
mkdir dataset
rclone sync google-drive:PetFace ./dataset --drive-shared-with-me --progress
```

This command will only work if you have the PetFace dataset in your 'Shared with me' folder of Google Drive.

The image files are all zipped tar files, to extract them you need to run the `tar -xzf` command. For convenience, there is the `extract_petface.sh` bash script in the `scripts` folder you can run to extract all of the images.

🚨 **IMPORTANT** 🚨 The PetFace dataset takes up **69 GB** in total _before_ extracting all the image files, and **138 GB** in total _after_ extraction, so make sure you have enough storage on your machine before downloading. 
