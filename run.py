import os
import argparse
from design_baselines.safeweight.experiments import ant, dkitty, hopper, superconductor
from design_baselines.safeweight_latent.experiments import gfp, molecule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['ant', 'dkitty', 'hopper', 'superconductor', 'gfp', 'molecule'])    
    args = parser.parse_args()

    task = args.task

    if task == "ant":
        ant()
    elif task == "dkitty":
        dkitty()
    elif task == "hopper":
        hopper()
    elif task == "superconductor":
        superconductor()
    elif task == "gfp":
        gfp()
    elif task == "molecule":
        molecule()


if __name__  == "__main__":
    main()
