# Threat Intelligence Filtering Pipeline

---

## Installation & Environment Setup

Before running the main pipeline, ensure you have Python 3.10+ installed. Follow the steps below to set up your environment and dependencies.

### Install Core System Dependencies
The language filtration module requires compiling **FastText**, which needs a C++ compiler (`gcc` or `clang`).
* **Ubuntu/Debian:** `sudo apt-get install build-essential`
* **macOS:** `xcode-select --install`
* **Windows:** Install Visual Studio Build Tools with C++ desktop development components.

### Install Required Packages
Install all necessary data processing, probabilistic, and natural language packages:

```bash
pip install pandas numpy wordfreq fasttext-wheel datasketch pybloom-live psutil
```

### The language identification layer relies on Meta's pre-trained FastText subword compressed binary. Download it and place it in your root directory:
```bash
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### Run the testing script:
```bash
python -m unittest test_pipeline_stages.py -v
```

### File Overview

* **`pipeline.py`**: The core system file: manages the multi-core filtering. This file must be run before any other files, to ensure the creation of the needed csv files. Add the unfiltered unfiltered-messages.csv file to the folder before running this file.
* **`test_pipeline.py`**: The testing framework: runs isolated unit and integration tests.
* **`analyze_length.py`**: Analyzes character and word distributions to establish the 30-to-2,500 length thresholds.
* **`check_memory_usage.py`**: Logs system CPU and RAM usage over time to monitor resource constraints.
* **`generate_irr_task.py`**: The sampling file.
