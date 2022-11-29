# ProtoTrees: Neural Prototype Trees for Interpretable Fine-grained Image Recognition
This repository presents the PyTorch code for Neural Prototype Trees (ProtoTrees), published at CVPR 2021: ["Neural Prototype Trees for Interpretable Fine-grained Image Recognition"](https://openaccess.thecvf.com/content/CVPR2021/html/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.html).

A ProtoTree is an intrinsically interpretable deep learning method for fine-grained image recognition. It includes prototypes in an interpretable decision tree to faithfully visualize the entire model. Each node in our binary tree contains a trainable prototypical part. The presence or absence of this prototype in an image determines the routing through a node. Decision making is therefore similar to human reasoning: Does the bird have a red throat? And an elongated beak? Then it's a hummingbird!

![Example of a ProtoTree.](images/prototree_teaser.png "ProtoTree")
Figure shows an example of a ProtoTree. A ProtoTree is a globally interpretable model faithfully explaining its entire behaviour (left, partially shown) and additionally the reasoning process for a single prediction can be followed (right): the presence of a red chest and black wing, and the absence of a black stripe near the eye, identifies a Scarlet Tanager.

## Prerequisites

### General
* Python 3
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5 and <= 1.7!
* Optional: CUDA

### Required Python Packages:
* numpy
* pandas
* opencv
* tqdm
* scipy
* matplotlib
* requests (to download the CARS dataset, or download it manually)
* gdown (to download the CUB dataset, or download it manually)

## Data
The code can be applied to the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset with 200 bird species, or the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset with 196 car types.

The folder `preprocess_data` contains python code to download, extract and preprocess these datasets.

### Preprocessing CUB
1. create a folder ./data/CUB_200_2011
2. download [ResNet50 pretrained on iNaturalist2017](https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU) (Filename on Google Drive: `BBN.iNaturalist2017.res50.180epoch.best_model.pth`) and place it in the folder `features/state_dicts`.
3. from the main ProtoTree folder, run `python preprocess_data/download_birds.py`
4. from the main ProtoTree folder, run `python preprocess_data/cub.py` to create training and test sets

### Preprocessing CARS
1. create a folder ./data/cars
2. from the main ProtoTree folder, run `python preprocess_data/download_cars.py`
3. from the main ProtoTree folder, run `python preprocess_data/cars.py` to create training and test sets

## Training a ProtoTree
1. create a folder ./runs

A ProtoTree can be trained by running `main_tree.py` with arguments. An example for CUB:
```bash
main_tree.py
	--num_features 256 --depth 9 --net resnet50_inat --dataset CUB-200-2011 \
	--epochs 100 --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --device cuda:0 \
	--freeze_epochs 10 --milestones 60,70,80,90,100 --batch_size 64 --random_seed 42 \
	--root_dir ~/runs/prototree  \
	--proj_dir proj_corners_sm --upsample_mode smoothgrads --upsample_threshold 0.3 --projection_mode corners
```
Patch/prototype visualization is controlled by the following options:
* --upsample\_mode: Either 'vanilla' (upsampling w/ cubic interpolation) or 'smoothgrads'
* --upsample\_threshold: The threshold of activation used to extract the patch bounding box. Suggested values are 0.98 for 'vanilla' and '0.3' for 'smoothgrads'.
* --projection\_mode: Which dataset to use during projection (CUB only). Either:
	* 'corners': Use the augmented dataset (5 corners crop) used during training.
	* 'cropped': Use the training set cropped using provided bounding boxes.
	* 'raw': Use the raw training set (closer to the test set).
* --proj\_dir: Since there multiple ways to perform projection, specify target directory inside root\_dir

To speed up the training process, the number of workers of the [DataLoaders](https://github.com/M-Nauta/ProtoTree/blob/main/util/data.py#L39) can be increased by setting `num_workers` to a positive integer value (suitable number depends on your available memory).

Check your `--root_dir` to keep track of the training progress. This directory contains `log_epoch_overview.csv` which prints per epoch the test accuracy, mean training accuracy and the mean loss. File `log.txt` logs additional info.

The resulting visualized prototree (i.e. *global explanation*) is saved as a pdf in your `--root_dir / --proj_dir/treevis.pdf`. NOTE: this pdf can get large which is not supported by Adobe Acrobat Reader. Open it with e.g. Google Chrome or Apple Preview.

NOT TESTED! To train and evaluate an ensemble of ProtoTrees, run `main_ensemble.py` with the same arguments as for `main_tree.py`, but include the `--nr_trees_ensemble` to indicate the number of trees in the ensemble.

## Restart a training sequence from a checkpoint
There are two ways to restart a training sequence.
1. Relaunch the `main_tree.py` with exactly the same options. If the `--root_dir` directory exists, the training process will automatically restart from the checkpoint located in `--root_dir/checkpoints/latest`.
2. Specify explicitely a path using the `--tree_dir` option pointing to the checkpoint directory.

## Perform different projection methods on a pretrained ProtoTree
It is possible to test different projection methods with different projection datasets without retraining the entire ProtoTree.
```bash
finalize_tree.py --tree_dir ./runs/prototree/checkpoints/latest/ --root_dir runs/prototree \
	--dataset CUB-200-2011 --batch_size 16 --device cuda:0
	--proj_dir proj_raw_vanilla --upsample_threshold 0.98 --upsample_mode vanilla --projection_mode raw
```

### Local explanations
A trained ProtoTree is intrinsically interpretable and globally explainable. It can also *locally* explain a prediction. Run e.g. the following command to explain a single test image:


```bash
python main_explain_local.py --tree_dir runs/prototree/proj_corners_sm/model/ --proj_dir proj_corners_sm
	--root_dir ./runs/prototree
	--sample_dir ./data/CUB_200_2011/dataset/test_full/096.Hooded_Oriole/
	--upsample_mode smoothgrads --upsample_threshold 0.3
	--results_dir local_explanations_sm
```

In the folder `--root_dir`/`--proj_dir`/`--results_dir`, the visualized local explanation is saved in `predvis.pdf`.
