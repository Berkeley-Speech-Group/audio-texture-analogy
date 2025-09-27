# Audio Texture Manipulation by Exemplar-Based Analogy

This is the repository that contains source code for the [paper](https://arxiv.org/abs/2501.12385) and the [project website](https://berkeley-speech-group.github.io/audio-texture-analogy/).
The code implements the method described in our paper with the following difference:
- Instead of using mel-spectrogram, spectrogram VAE, and a 16k Hz HiFi-GAN vocoder, we use a [44.1k Hz waveform VAE](https://arxiv.org/abs/2405.18503) for better reconstruction and higher fidelity generation.
- We replicate [AUDIT](https://arxiv.org/abs/2304.00830) for text-conditioned audio editing also using the above setup.


## Installation
```bash
git clone https://github.com/Berkeley-Speech-Group/audio-texture-analogy.git
conda create -n atm python=3.9
conda activate atm
cd src/audio-diffusion-pytorch
pip install -e .
cd soundctm_dit_iclr
pip install -e .
```


## Citation
If you find our research useful for your work please cite:
```bibtex
@inproceedings{cheng2025audio,
  title={Audio Texture Manipulation by Exemplar-Based Analogy},
  author={Cheng, Kan Jen and Li, Tingle and Anumanchipalli, Gopala},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
