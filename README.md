<div align="center">
<img src="https://raw.githubusercontent.com/perrin-isir/o3p/main/o3p/assets/o3p_logo.png" alt="o3p logo"></img>
</div>

[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**o3p** is a JAX-based library for offline and online off-policy reinforcement learning.

It is currently in BETA VERSION.

<details><summary> <b>Install from source</b> </summary><p>

    git clone https://github.com/perrin-isir/o3p.git

We recommand to create a python environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html),
but any python package manager can be used instead.

    cd o3p

    micromamba create --name o3penv --file environment.yaml

    micromamba activate o3penv

    pip install -e .
    
* About [JAX](https://docs.jax.dev/en/latest/index.html): JAX is in the dependencies, so the procedure will install it on your system. However, if you encounter specific issues with JAX (e.g. it runs on your CPU instead of your GPU), we recommend to install it separately, following instructions at: [https://docs.jax.dev/en/latest/installation.html#installation](https://docs.jax.dev/en/latest/installation.html#installation).

* About [TFP](https://github.com/tensorflow/probability): Currently, the latest stable version of TFP is not compatible with the latest version of JAX. Therefore, you should upgrade to a nightly build with this command (within your new o3penv environment):

    pip install --upgrade --user tf-nightly tfp-nightly

</details>
<details><summary> <b>How to use it</b> </summary><p>

To test offline RL, run:

    python test/offline_rl.py

TODO

To test online RL, run:

    python test/online_rl.py

</details>
<details><summary> <b>Design choices</b> </summary><p>

TODO

</details>
