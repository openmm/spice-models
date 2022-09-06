This directory contains the five equivariant transformer models described in (insert reference when available).

The script `createSpiceDataset.py` converts the dataset file SPICE.hdf5 downloaded from https://github.com/openmm/spice-dataset/releases
to the format used by [TorchMD-Net](https://github.com/torchmd/torchmd-net).  It generates a new file SPICE-processed.hdf5
which was used for training.

The file `hparams.yaml` contains the configuration used for training the models.  All models used identical settings
except that `seed` was set to a different value for each one (the numbers 1 through 5).  Note that although the file
specifies `num_epochs: 1000`, training was halted after 24 hours (when the training job reached the end of its allocated
time).  This corresponded to 118 epochs.

The files ending in `.ckpt` are checkpoint files for TorchMD-Net 0.2.4 containing the trained models.  They should
hopefully work with later versions as well, but that may not be guaranteed.  They can be loaded like this:

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