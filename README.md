# samplernn-pytorch avec conditionnement (Implémentation GRPA)

Une implementation PyTorch de [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837). Incluant une première tentative de conditionnement paramètriques semblable à [High-quality speech coding with SampleRNN](https://arxiv.org/abs/1811.03021)

Le code de base est tiré de [samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch)

## Dependencies

Ce code nécessite Python 3.5+ and PyTorch 0.1.12+. Les instructions d'installation de PyTorch sont disponibles sur leur site Web: http://pytorch.org. Aussi, pour l'entrainement sur GPU il est recommandé d'installer le [cuda toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) et d'avoir les pilots NVIDIA fonctionnels.

Vous pouvez créer l'environnement avec les dépendences nécéssaires avec la commande: `conda env create -f environment.yml`

## Datasets

Les fichiers audio utilisés pour l'entrainement doivent être dans le dossiers `./datasets/YOUR_DATASET`
## Training

To train the model you need to run `train.py`. All model hyperparameters are settable in the command line. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. To train on the `piano` dataset using the best hyperparameters we've found, run:

```
python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset piano
```

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`.
