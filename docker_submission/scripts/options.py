import argparse
import json
import os

def load_config():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-region", type=str, required=True, help="can be either 'brain' or 'abdom'.")
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-workspace", type=str, required=True)

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
    
    return config