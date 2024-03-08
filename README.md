# T-TAME: Trainable Attention Mechanism for Explanations, now for transformers

This repository hosts the code and data lists for our learning-based eXplainable AI (XAI) method called T-TAME, for Convolutional and Transformer-like Deep Neural Network-based (DNN) image classifiers. Our method receives as input an image and a class label and produces as output the image regions that the DNN has focused on in order to infer this class. T-TAME uses an attention mechanism (AM), trained end-to-end along with the original, already-trained (frozen) DNN, to derive class activation maps from feature map sets extracted from selected layers. During training, the generated attention maps of the AM are applied to the inputs. The AM weights are updated by applying backpropagation on a multi-objective loss function to optimize the appearance of the attention maps (minimize high-frequency variation and attention mask area) and minimize the cross-entropy loss. This process forces the AM to learn the image regions responsible for the DNN’s output. Two widely-used evaluation metrics, Increase in Confidence (IC) and Average Drop (AD), are used for evaluation. Additionally, the promising ROAD framework is also used for evaluation. We evaluate T-TAME on the ImageNet dataset, using the VGG16, ResNet-50 and ViT-B-16 DNNs. Our method outperforms the state-of-the-art methods in terms of IC and AD, and achieves competitive results in terms of ROAD. We also provide a detailed ablation study to demonstrate the effectiveness of our method.

- This repository contains the code for training, evaluating and applying T-TAME, using VGG-16, ResNet-50 or ViT-B-16 as the pre-trained backbone network along with the Attention Mechanism and our selected loss function. There is also a guide on applying TAME to any DNN image classifier.
- Instead of training, the user can also use a pretrained attention mechanism for the pretrained VGG-16, ResNet-50 or ViT-B-16 image classifiers in `T-TAME/logs/`.
- In `T-TAME/datalist/ILSVRC`, text files with annotations for 2000 randomly selected images to be used at the validation stage (Validation_2000.txt) and 2000 randomly selected images (exclusive of the previous 2000) for the evaluation stage (Evaluation_2000.txt) of the L-CAM methods.
- The ILSVRC 2012 dataset images should be downloaded by the user manually.

---

- [T-TAME: Trainable Attention Mechanism for Explanations, now for transformers](#t-tame-trainable-attention-mechanism-for-explanations-now-for-transformers)
  - [Initial Setup](#initial-setup)
  - [Available scripts](#available-scripts)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)

## Initial Setup

Make sure that you have a working git, Python 3, cuda and poetry installation before proceeding.

- To install git, follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

- To install python, follow the instructions [here](https://www.python.org/downloads/).

- To install cuda, follow the instructions [here](https://developer.nvidia.com/cuda-downloads).

- To install poetry, follow the instructions [here](https://python-poetry.org/docs/).

1. Clone this repository:

   ```shell
   git clone git@github.com:marios1861/T-TAME.git
   ```

2. Go to the locally saved repository path:

   ```shell
   cd T-TAME
   ```

3. Install:

   ```shell
   poetry install
   ```

> __Note__: You may need to modify the venv activate script in the case that cuda is already installed in your machine. If so, add this line:
> `export LD_LIBRARY_PATH=.../venv/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH`

## Available scripts

You can evaluate the compared methoods, including T-TAME with the `{vgg16, resnet50, vit_b_16}` backbones using the following command:
 using the command:

```shell
python pl_scripts/{vgg16, resnet50, vit_b_16}_comparisons.py
```

You can train the T-TAME method using the following command:

```shell
python pl_scripts/{vgg16, resnet50, vit}_TAME.py
```

You can generate explanation maps for the T-TAME method using the following command:

```shell
python pl_scripts/TAME_print_mask.py
```

You can generate explanation maps for the compared methods using the following command:

```shell
python pl_scripts/othger_methods_print_mask.py
```

## Citation

<div align="justify">

If you find our TAME method, code or pretrained models useful in your work, please cite the following publication:

<span style="color:red">
M. Ntrougkas, N. Gkalelis, V. Mezaris, "TAME: Attention Mechanism Based Feature Fusion for Generating Explanation Maps of Convolutional Neural Networks", 2022, under review.
</span>

</div>

BibTeX:

<span style="color:red">
~~~bibtex
@INPROCEEDINGS{Ntrougkas2022,
    author    = {Ntrougkas, Mariano and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {TAME: Attention Mechanism Based Feature Fusion for Generating Explanation Maps of Convolutional Neural Networks},
    booktitle = {under review},
    month     = {},
    year      = {2022},
    pages     = {}
}
~~~
</span>

<div align="justify">

You may want to also consult and, if you find it also useful, also cite our earlier work on this topic (methods L-CAM-Img, L-CAM-Fm):

I. Gkartzonika, N. Gkalelis, V. Mezaris, "Learning Visual Explanations for DCNN-Based Image Classifiers Using an Attention Mechanism", Proc. ECCV 2022 Workshop on Vision with Biased or Scarce Data (VBSD), Oct. 2022.
</div>

BibTeX:

~~~bibtex
@INPROCEEDINGS{Gkartzonika2022,
    author    = {Gkartzonika, Ioanna and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {Learning Visual Explanations for DCNN-Based Image Classifiers Using an Attention Mechanism},
    booktitle = {Proc. ECCV 2022 Workshop on Vision with Biased or Scarce Data (VBSD)},
    month     = {Oct.},
    year      = {2022},
    pages     = {}
}
~~~

## License

<div align="justify">

Copyright (c) 2024, Mariano Ntrougkas, Nikolaos Gkalelis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement

The T-TAME implementation was built in part on code previously released in the [L-CAM](https://github.com/bmezaris/L-CAM) repository.

The code for the methods that are used for comparison is taken from the [L-CAM](https://github.com/bmezaris/L-CAM) repository for L-CAM-Img, the [RISE](https://github.com/eclique/RISE) repository for RISE, the [IIA](https://github.com/iia-iccv23/iia) repository for IIA, the [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) repository for the Transformer LRP method and the [pytorch-gradcam](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository for all of the remaining methods.

<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-101021866 CRiTERIA. </div>