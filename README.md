# Unsupervised Semantic Correspondence Using Stable Diffusion

[Paper](https://arxiv.org/abs/2305.15581)

[Project Page](https://ubc-vision.github.io/LDM_correspondences/)

This repository contains the implementation of our method for estimating correspondences with Stable Diffusion in an unsupervised manner. Our new method surpasses weakly supervised methods and closes the gap to strongly supervised methods. 

## Getting Started

Here are instructions on how to run the repository:

1. Install dependencies: This project uses a conda environment for managing dependencies. You can create the environment and install all dependencies with the following command:
    ```shell
    conda env create -f environment.yml
    ```
2. Run the evaluation script:
    ```shell
    conda activate LDM_correspondences
    python3 -m eval.eval
    ```
3. More options can be found with 
    ```shell
    python3 -m eval.eval --help
    ```

## Visualizing Attention Maps

The project includes an interactive local website for visualizing attention maps associated with identified correspondences. Follow the steps below to launch the visualization:

1. Activate the conda environment and run the evaluation script with visualization:

    ```shell
    conda activate LDM_correspondences
    python3 -m eval.eval --visualize
    ```

2. Launch the interactive website by running the visualization script:

    ```shell
    python3 -m clickable_lines.app
    ```

This will display correspondences. Click on each to visualize the corresponding attention maps. 

## Method Overview

We supervise the attention maps corresponding to randomly initialized text embedding to activate in a source region. This text embedding can then be applied to any target image where we simply look for the argmax in its attention map.

[![Method Overview](./docs/method.png)](https://youtu.be/br2zX9XkWX0)

We are motivated by the fact that the attention maps for specific words act as pseudo-segmentation for those regions. By inputting an image instead of random noise we can use Stable Diffusion for inference tasks.

![English Word Attention Maps](./docs/english_word_attn_maps.png)

We find that even when our method predicts incorrect correspondences, the regions it predicts still seem reasonable. On the bottom right, of note, even though all points are meant to correspond with the wine bottle, points occluded by the wine glass instead map to the wine glass.

![Qualitative Examples](./docs/qualitative_examples.png)

Our method outperforms weakly supervised methods and in the case of PF-Willow, is on par with strongly supervised methods.

![Qualitative Performance](./docs/qualitative_performance.png)

We also find that when we look for correspondences between different classes, it still estimates plausible correspondences.

![Cross Class Correspondences](./docs/cross_class_correspondences.png)


## Citing
If you find this code useful for your research please consider citing the following paper:

	@article{hedlin2023unsupervised,
      title={Unsupervised Semantic Correspondence Using Stable Diffusion}, 
      author={Eric Hedlin and Gopal Sharma and Shweta Mahajan and Hossam Isack and Abhishek Kar and Andrea Tagliasacchi and Kwang Moo Yi},
      year={2023},
      eprint={2305.15581},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }