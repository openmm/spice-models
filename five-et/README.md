# Using the Models

This directory contains the five equivariant transformer models described in (insert reference when available).
They were created with TorchMD-Net 0.2.2.  They might work with later versions as well, but that is not guaranteed.

To use them, first install TorchMD-Net by following the instructions at https://github.com/torchmd/torchmd-net.
That involves checking out the source code, creating a conda environment using the provided environment file,
and running `pip` to install it into the environment.

The files ending in `.ckpt` are checkpoint files containing the trained models.  They can be loaded like this:

```python
from torchmdnet.models.model import load_model
model = load_model('model1.ckpt')
```

The `device` argument to `load_model()` can be used to specify a device to load it on.  For example,

```python
model = load_model('model1.ckpt', device=torch.device('cuda:0'))
```

To compute energy and forces for a molecular conformation, invoke the model's `forward()` method.  It takes two arguments:
a tensor of length `n_atoms` and dtype `long` containing the atom types, and a tensor of shape `(n_atoms, 3)` and dtype
`float32` containing the atom positions in angstroms.  It returns two arguments: the potential energy in kJ/mol, and
the force on each atom in kJ/mol/angstrom.  Atom types are defined by the element and formal charge of each atom.  The
mapping is defined in `createSpiceDataset.py` with this dictionary:

```python
typeDict = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
```

For example, the following computes the energy and forces for a pair of ions (Cl- and Na+) positioned 3 angstroms apart.

```python
types = torch.tensor([6, 19], dtype=torch.long)
pos = torch.tensor([[0, 0, 0], [0, 3, 0]], dtype=torch.float32)
energy, forces = model.forward(types, pos)
```

# Training New Models

If you want to train new models on the same data, follow these steps.

1. Install OpenFF-Toolkit into the conda environment by executing the command

```
conda install -c conda-forge openff-toolkit=0.10.6
```

2. Download the `SPICE.hdf5` file from https://github.com/openmm/spice-dataset/releases/tag/1.1 and place it
   in this directory.
3. Run the `createSpiceDataset.py` script, which converts the dataset to the format used by TorchMD-Net.  It
   generates a new file `SPICE-processed.hdf5` to use for training.
4. Run the `train.py` script provided by TorchMD-Net.  The command will be something like

```
python <path to torchmd-net>/scripts/train.py --conf hparams.yaml
```

The file `hparams.yaml` contains the configuration used for training the models.  All models here used identical settings
except that `seed` was set to a different value for each one (the numbers 1 through 5).  Be sure to use TorchMD-Net 0.2.2,
since later versions made incompatible changes to some of the parameter definitions.  Note that although the file
specifies `num_epochs: 1000`, training was halted after 24 hours (when the training job reached the end of its allocated
time).  This corresponded to 118 epochs.  You can edit the file to try different hyperparameters, or override them with
command line arguments to `train.py`.
