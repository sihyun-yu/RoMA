# RoMA: Robust Model Adaptation for Offline Model-based Optimization

Implementation of [RoMA: Robust Model Adaptation for Offline Model-based Optimization (NeurIPS 2021)](http://arxiv.org/abs/2110.14188).


## Setup
```
conda create -n roma python=3.7
conda activate roma
pip install -r requirement.txt
```

## Run experiments
```
python run.py --task [TASK]
```
where available tasks are `TASKS=[ant, superconductor, dkitty, hopper, gfp, molecule].`


## Citation
```
@inproceedings{
    yu2021roma,
    title={RoMA: Robust Model Adaptation for Offline Model-based Optimization},
    author={Yu, Sihyun and Ahn, Sungsoo and Song, Le and Shin, Jinwoo},
    booktitle={Advances in Neural Information Processing Systems},
    year={2021},
}
```


## References
- [Design-bench](https://github.com/brandontrabucco/design-bench)
- [Design-baselines](https://github.com/brandontrabucco/design-baselines)
- [Adversarial weight perturbation](https://github.com/csdongxian/AWP)
