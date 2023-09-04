import argparse
import json
import os

def load_config():    
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input", required=True, type=str)
    # parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-i", "--input", type=str, default="/mnt/sda/Data/MOOD/brain_toy/toy")
    parser.add_argument("-region", type=str, default="brain", help="can be either 'brain' or 'abdom'.", required=False)
    # parser.add_argument("-i", "--input", type=str, default="/mnt/sda/Data/MOOD/abdom_toy/toy")
    # parser.add_argument("-region", type=str, default="abdom", help="can be either 'brain' or 'abdom'.", required=False)
    parser.add_argument("-o", "--output", type=str, default="/home/ehuijben/ownCloud2/Code/models/202306_OOD/mood/docker_example/tmp_output")
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-workspace", type=str, default="/home/ehuijben/ownCloud2/Code/models/202306_OOD/mood/docker_example/scripts")

    args = parser.parse_args()
    
    # Load configuration from JSON file
    with open(os.path.join(args.workspace, 'submission_config.json'), 'r') as config_file:
        config = json.load(config_file)
    
    # Create an argparse.Namespace object from the configuration dictionary
    config = argparse.Namespace(**config)
    
    if args.region == 'brain':
        config.model_name = config.model_name_brain
    else:
        config.model_name = config.model_name_abdom
    config.output_dir = os.path.join(args.workspace, config.output_dir)
    config.result_dir = args.output
    config.input_dir = args.input
    config.task = args.region
    config.mode = args.mode
    config.hist_dir = os.path.join(args.workspace, 'histogram') 

    print()
    print('************* TEST GPU ********************')
    import torch
    print('CUDA AVAILABLE:', torch.cuda.is_available())
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', config.device)
    test_gpu = torch.ones((1,)).to(config.device)
    print('TENSOR DEVICE', test_gpu.device)
    
    print('*******************************************')
    print()
    
    return config