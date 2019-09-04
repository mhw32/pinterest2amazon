import os
import json
import torch
from PIL import Image
from src.agents.agents import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, 
                        help='path to image to test')
    parser.add_argument('exp_dir', type=str,
                        help='path to trained file')
    parser.add_argument('--k', type=int, default=1,
                        help='number of nearest neighbours [default: 1]')
    args = parser.parse_args()

    checkpoint_name = 'final.pth.tar'
    checkpoint_path = os.path.join(args.exp_dir, 'checkpoints', checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    config = checkpoint['config']

    print("Loading trained Agent from filesystem")
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)
    agent.load_checkpoint(checkpoint_name)
    agent._set_models_to_eval()

    _, test_transforms = agent._load_image_transforms()
    image = Image.open(args.image_path)
    image = test_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(agent.device)

    outputs = agent.model(image)
    all_dps = agent.memory_bank.get_all_dot_products(outputs)
    _, indices = torch.topk(all_dps, k=args.k, sorted=False, dim=1)
    indices = indices.flatten().cpu().numpy()

    paths = []
    for index in indices:
        path, target = agent.train_dataset.dataset.samples[index]
        paths.append(path)

    with open('./neighbors.json', 'w') as fp:
        json.dump(paths, fp)
