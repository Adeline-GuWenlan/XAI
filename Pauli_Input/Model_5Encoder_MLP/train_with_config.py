"""
Train Encoder-MLP model using YAML configuration file
Usage: python train_with_config.py --config config.yaml
"""

import yaml
import argparse
import sys
import subprocess
import os


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_command_from_config(config):
    """Build training command from configuration"""
    cmd = ["python", "train_encoder_mlp.py"]

    # Model parameters
    cmd.extend(["--d_model", str(config['model']['d_model'])])
    cmd.extend(["--nhead", str(config['model']['nhead'])])
    cmd.extend(["--num_encoder_layers", str(config['model']['num_encoder_layers'])])
    cmd.extend(["--dim_feedforward", str(config['model']['dim_feedforward'])])
    cmd.extend(["--pooling", config['model']['pooling_method']])

    # Training parameters
    cmd.extend(["--batch_size", str(config['training']['batch_size'])])
    cmd.extend(["--learning_rate", str(config['training']['learning_rate'])])
    cmd.extend(["--num_epochs", str(config['training']['num_epochs'])])
    cmd.extend(["--train_split", str(config['data']['train_split'])])

    # Regression head
    head_type = config['regression_head']['type']
    cmd.extend(["--head_type", head_type])

    return cmd


def print_config_summary(config):
    """Print a summary of the configuration"""
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)

    print("\nData:")
    print(f"  Directory: {config['data']['data_dir']}")
    print(f"  X file: {config['data']['X_filename']}")
    print(f"  Y file: {config['data']['Y_filename']}")
    print(f"  Train/Val split: {config['data']['train_split']:.2%}")

    print("\nModel Architecture:")
    print(f"  d_model: {config['model']['d_model']}")
    print(f"  Attention heads: {config['model']['nhead']}")
    print(f"  Encoder layers: {config['model']['num_encoder_layers']}")
    print(f"  Feedforward dim: {config['model']['dim_feedforward']}")
    print(f"  Pooling method: {config['model']['pooling_method']}")

    print("\nTraining:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Epochs: {config['training']['num_epochs']}")

    print("\nRegression Head:")
    head_type = config['regression_head']['type']
    print(f"  Type: {head_type.upper()}")

    if head_type == 'ensemble':
        params = config['regression_head']['ensemble']
        print(f"  Deep weight: {params['deep_weight']}")
        print(f"  Shallow weight: {params['shallow_weight']}")
        print(f"  Medium weight: {params['medium_weight']}")
    elif head_type == 'lasso':
        params = config['regression_head']['lasso']
        print(f"  Lambda: {params['lasso_lambda']}")
    elif head_type == 'gbdt':
        params = config['regression_head']['gbdt']
        print(f"  Estimators: {params['n_estimators']}")
        print(f"  Max depth: {params['max_depth']}")
        print(f"  Learning rate: {params['learning_rate']}")
        print(f"  MLP refinement: {params['use_mlp_refinement']}")

    if 'experiment' in config:
        exp = config['experiment']
        if exp.get('name'):
            print(f"\nExperiment: {exp['name']}")
        if exp.get('notes'):
            print(f"Notes: {exp['notes']}")
        if exp.get('tags'):
            print(f"Tags: {', '.join(exp['tags'])}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train Encoder-MLP model using YAML configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and command without running training'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found!")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Print summary
    print_config_summary(config)

    # Build command
    cmd = build_command_from_config(config)

    # Update data paths in train_encoder_mlp.py if needed
    # Note: This requires modifying the training script to accept data paths as arguments
    # For now, make sure the paths in config.yaml match those in train_encoder_mlp.py

    print("Training command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("Dry run mode - training not started.")
        print("Remove --dry-run flag to start training.")
        return

    # Prompt user to continue
    response = input("Start training? [y/N]: ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
