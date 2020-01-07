# Transformer

## Running Code

- Adjust hyperparameters in ```hyperparams.py```, especially 'data_path' which is a directory that you extract files, and the others if necessary.

- ```python3 prepare_data.py```

- ```python3 train_transformer.py``` (text --> mel)

- ```python3 train_postnet.py``` (mel --> linear)

- Generate samples with ```python3 synthesis.py```