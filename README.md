# Real-ESRGAN-Enhanced-Anime-Diffusion
Generate high resolution and quality anime pictures from texts or existed images.

(Based on [Anything V3](https://huggingface.co/Linaqruf/anything-v3.0) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN))

### Colab demo : [Demo](https://colab.research.google.com/drive/1HpLkNnBfbrLD6t7cGc2i2gVAwiA_V_qp?usp=sharing)

## Installation

This project requires:
  Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

clone this repository

```bash
  git clone https://github.com/dotmet/Real-ESRGAN-Enhanced-Anime-Diffusion.git
```

install depencies

```bash
  cd Real-ESRGAN-Enhanced-Anime-Diffusion
  pip install -r requirements.txt
```

## Run

```bash
  python inference.py
```
  Type ```python inference.py -h``` in command line to see more options.
  
  The text are passed by ```-wd``` or ```--words```,  and this arg should be follow by one sentence or the name of a file which contains text(s). Notice that each text in the file should strictly be 1 line (The text in file will be automatically split by the symbol "\n"). Example usage:
  ```python inference.py -wd "1girl, beautiful eyes, "```, 
  ```python inference.py -wd prompt_keys.txt```
  
## Run Web UI

```
  python app.py
```
