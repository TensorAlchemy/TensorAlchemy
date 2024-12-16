<div align="center">

![TensorAlchemy - Splash image](../../TensorAlchemy-splash.png)

# **TESTNET GUIDE FOR MINERS** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This documentation is for starting and running a miner on the TensorAlchemy testnet. Note that we are continuing to push and test updates so you may notice the miner and/or validator encounter issues.

## MINER REQUIREMENTS
- ⚗️ NVIDIA 4090 | A6000 or better
- ⚗️ Ubuntu 20.04 or 22.04
- ⚗️ Python 3.9 or 3.10
- ⚗️ CUDA 12.0 or higher

Python 3.10 is used in the steps below:

### STEP 1. CLONE REPOSITORY
```bash
git clone https://github.com/TensorAlchemy/TensorAlchemy.git ~/TensorAlchemy
```

### STEP 2. INSTALL PREREQUISITES
```bash
sudo apt-get update && sudo apt-get install python3.10-venv
```

### STEP 3. CREATE A VENV
```bash
python3.10 -m venv ~/venvs/TensorAlchemy && source ~/venvs/TensorAlchemy/bin/activate && pip install wheel && pip install --upgrade setuptools
```

### STEP 4. INSTALL REQUIREMENTS INTO VENV
```bash
source ~/venvs/TensorAlchemy/bin/activate && cd ~/TensorAlchemy && pip install -r requirements.txt
```

### STEP 5. LAUNCH MINER
```bash
python ~/TensorAlchemy/neurons/miners/Inpainter/main.py --wallet.name NAME --wallet.hotkey HOTKEY --netuid 25 --subtensor.network test --axon.port 8101 --miner.device cuda:0
```

### NOTES
- ⚗️ TensorAlchemy is netuid 26 on production however it is netuid 25 on the testnet.
