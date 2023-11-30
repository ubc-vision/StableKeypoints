# Unsupervised Keypoints from Pretrained Diffusion Models

[Eric Hedlin](https://ehedlin.github.io/), [Gopal Sharma](https://hippogriff.github.io/), [Shweta Mahajan](https://s-mahajan.github.io/), [Xingzhe He](https://xingzhehe.github.io/), [Hossam Isack](http://www.hossamisack.com/), [Abhishek Kar](https://abhishekkar.info/), [Helge Rhodin](https://www.cs.ubc.ca/~rhodin/web/), [Andrea Tagliasacchi](https://taiya.github.io/), [Kwang Moo Yi](https://www.cs.ubc.ca/~kmyi/)

## Project Page

For more detailed information, visit our project page: [StableKeypoints](https://ubc-vision.github.io/StableKeypoints/)

## Requirements

### Set up environment

Create a conda environment using the provided `requirements.yaml`:

```bash
conda env create -f requirements.yaml
conda activate StableKeypoints
```

### Download datasets

The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Taichi](https://github.com/AliaksandrSiarohin/motion-cosegmentation), [Human3.6m](http://vision.imar.ro/human3.6m/description.php), [DeepFashion](https://github.com/theRealSuperMario/unsupervised-disentangling/tree/reproducing_baselines/original_code/custom_datasets/deepfashion), and [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) datasets can be found on their websites.

Preprocessed data for CelebA, and CUB can be found in [Autolink's repository](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/tree/main/datasets/preprocess).

## Usage

To use the code, run:

```bash
python3 -m unsupervised_keypoints.main [arguments]
```

### Main Arguments

- `--dataset_loc`: Path to the dataset.
- `--dataset_name`: Name of the dataset.
- `--num_steps`: Number of steps (default 500, up to 10,000 for non-human datasets).
- `--evaluation_method`: Following baselines, the evaluation method varies by dataset:
  - CelebA: 'inter_eye_distance'
  - CUB: 'visible'
  - Taichi: 'mean_average_error' (renormalized per keypoint)
  - DeepFashion: 'pck'
  - Human3.6M: 'orientation_invariant'
- `--save_folder`: Output save location (default "outputs" inside the repo).

## Example Usage

```bash
python3 -m unsupervised_keypoints.main --dataset_loc /path/to/dataset --dataset_name celeba_wild --evaluation_method inter_eye_distance --save_folder /path/to/save
```

## BibTeX

```bibtex
@article{hedlin2023keypoints,
  title={Unsupervised Keypoints from Pretrained Diffusion Models},
  author={Eric Hedlin and Gopal Sharma and Shweta Mahajan and Xingzhe He and Hossam Isack and Abhishek Kar and Helge Rhodin and Andrea Tagliasacchi and Kwang Moo Yi},
  journal={arXiv},
  year={2023},
}
```
