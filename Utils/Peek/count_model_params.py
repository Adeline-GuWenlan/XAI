#!/usr/bin/env python3
"""
Script to count parameters in PyTorch models with detailed breakdown
"""
import torch
import os
from pathlib import Path
from collections import defaultdict


def count_parameters(model_path):
    """Count parameters in a PyTorch model with component breakdown"""
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Count parameters by component
        component_counts = defaultdict(int)
        total_params = 0
        
        for name, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            
            # Categorize parameters
            if 'encoder' in name.lower():
                component_counts['encoder'] += param_count
            elif 'mlp' in name.lower() or 'fc' in name.lower() or 'linear' in name.lower():
                component_counts['mlp'] += param_count
            elif 'attention' in name.lower() or 'attn' in name.lower():
                component_counts['attention'] += param_count
            elif 'embed' in name.lower():
                component_counts['embedding'] += param_count
            elif 'norm' in name.lower() or 'bn' in name.lower() or 'ln' in name.lower():
                component_counts['normalization'] += param_count
            else:
                component_counts['other'] += param_count
        
        return {
            'total': total_params,
            'components': dict(component_counts),
            'parameter_details': {name: param.numel() for name, param in state_dict.items()}
        }, None
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def format_number(num):
    """Format number with commas and suffixes"""
    if num >= 1_000_000:
        return f"{num:,} ({num/1_000_000:.2f}M)"
    elif num >= 1_000:
        return f"{num:,} ({num/1_000:.1f}K)"
    else:
        return f"{num:,}"


def main():
    # Model paths
    model_paths = [
        "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth",
        "/Users/guwenlan/Desktop/XAI/CONFIGs/128_CLS_attention_enhanced/best_128_CLS_attention_enhanced_20250810_214428.pth",
        "/Users/guwenlan/Desktop/XAI/CONFIGs/64_CLS_asymmetric_ensemble_20250810_024658/best_64_CLS_asymmetric_ensemble_20250810_024658.pth",
        "/Users/guwenlan/Desktop/XAI/CONFIGs/128_Attention_attention_enhanced_20250815_120257/best_128_Attention_attention_enhanced_20250815_120257.pth",
        "/Users/guwenlan/Desktop/XAI/CONFIGs/moe_competitive_00000019.pt"
    ]
    
    print("=" * 80)
    print("PyTorch Model Parameter Analysis")
    print("=" * 80)
    
    total_all_models = 0
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\nModel {i}: {Path(model_path).name}")
        print("-" * 60)
        
        result, error = count_parameters(model_path)
        
        if error:
            print(f"❌ {error}")
            continue
        
        total_params = result['total']
        components = result['components']
        total_all_models += total_params
        
        print(f"📊 Total Parameters: {format_number(total_params)}")
        print()
        print("Component Breakdown:")
        
        # Sort components by parameter count (descending)
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        
        for component, count in sorted_components:
            percentage = (count / total_params) * 100
            print(f"  • {component.title():<15}: {format_number(count)} ({percentage:.1f}%)")
        
        # Show encoder and MLP specifically if they exist
        encoder_params = components.get('encoder', 0)
        mlp_params = components.get('mlp', 0)
        
        if encoder_params > 0 or mlp_params > 0:
            print()
            print("Key Components:")
            if encoder_params > 0:
                print(f"  🔧 Encoder:  {format_number(encoder_params)}")
            if mlp_params > 0:
                print(f"  🧠 MLP:      {format_number(mlp_params)}")
        
        print()
    
    print("=" * 80)
    print(f"📈 Total Parameters Across All Models: {format_number(total_all_models)}")
    print("=" * 80)


if __name__ == "__main__":
    main()