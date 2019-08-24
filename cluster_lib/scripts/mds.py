"""
Multidimensional Scaling to visualize embeddings.
"""
import os
import torch
import logging
from sklearn.manifold import MDS

from src.agents.agents import *


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, 
                        default='where trained models are saved')
    args = parser.parse_args()

    logger = logging.getLogger("Visualizer")

    checkpoint_path = os.path.join(args.exp_dir, 'checkpoints', checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    config = checkpoint['config']

    logger.info("Loading trained Agent from filesystem")
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)
    agent.load_checkpoint(checkpoint_name)
    agent._set_models_to_eval()

    bank = agent.get_memory_bank()._bank.cpu().numpy()
    logger.info("Fitting MDS model")
    embedding = MDS(n_components=2)
    bank = embedding.fit_transformbank)

    viz_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(viz_dir, exist_ok=True)

    logger.info("Visualizing embedding")
    plt.figure(figsize=(10,10))
    plt.scatter(bank[:, 0], bank[:, 1])
    plt.savefig(os.path.join(viz_dir, 'mds.png'))
