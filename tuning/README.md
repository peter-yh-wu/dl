# Simple Hyperparameter Optimization

config file has the following keys:

- name: name of ```exp``` subfolder

- search: GRID, RANDOM (default RANDOM)

- task: runs the main script for the given task

- model: class name of model to train

- maxTrials: max number of trials to perform (default is infinity)

- goal: MAX or MIN

- params: list, each param has the following keys:

    - name: e.g. "data"

    - type: INTEGER, DOUBLE, DISCRETE, or CATEGORICAL

    - min: for INTEGER or DOUBLE

    - max: for INTEGER or DOUBLE

    - step: for INTEGER or DOUBLE (default 1 for INTEGER)

    - values: list; for DISCRETE (number) or CATEGORICAL (str)

    - scale: for INTEGER or DOUBLE (LINEAR or EXP, default LINEAR)
