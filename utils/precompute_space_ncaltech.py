import math
import itertools


def calculate_multipliers(num_channels, layer):
    if layer in ['conv2', 'conv3']:
        num_m = math.ceil(num_channels / 5)
    elif layer in ['conv4', 'conv5']:
        num_m = math.ceil(num_channels / 61.5)
    elif layer in ['conv1']:
        num_m = num_channels
    return num_m

def precompute_space_ncaltech(config, depth_start=0, depth_end=1):
    """
    Precompute the space for the given configuration.

    For each convolutional layer, calculate the number of possible pruning patterns
    based on the number of output channels and the number of bits.

    Create a list of dictionaries, where each dictionary contains the following

    keys:
    - 'layer': The layer name (e.g., 'conv1', 'conv2', etc.)
    - 'num_channels': The number of output channels for the layer
    - 'num_bits': The number of bits for the layer
    - 'num_patterns': The number of possible pruning patterns for the layer


    """
    # Initialize the list to store the precomputed space
    precomputed_space = []

    # Iterate through the convolutional layers in the configuration
    
    for i in range(1, 6):
        layer_name = f'conv{i}'
        num_channels = getattr(config, layer_name).out_channels

        pruning_space = []
        bit_space = []
        bram_space = []
        m_space = []

        for num_bits in [6, 8]:
            if layer_name in ['conv1', 'conv2']:
                multiple = 9 if num_bits == 8 else 12
                const = 6 if num_bits == 8 else 9
                max_mulpiple = num_channels // multiple

                for j in range(depth_start, depth_end):
                    new_pruned_channels = (max_mulpiple - j) * multiple + const

                    if new_pruned_channels > num_channels:
                        continue
                    if new_pruned_channels <= 0:
                        break

                    # TODO - calculate URAM utilisation
                    num_uram = math.ceil( (new_pruned_channels * num_bits + 18) / 72 )
                    num_brams = num_uram * 2

                    num_m = calculate_multipliers(new_pruned_channels, layer_name)

                    pruning_space.append(new_pruned_channels)
                    bit_space.append(num_bits)
                    bram_space.append(num_brams)
                    m_space.append(num_m)

            elif layer_name in ['conv3', 'conv4', 'conv5']:
                multiple = 9 if num_bits == 8 else 3
                max_mulpiple = num_channels // multiple

                for j in range(depth_start, depth_end):
                    new_pruned_channels = (max_mulpiple - j) * multiple
                    if new_pruned_channels <= 0:
                        break

                    # Calculate the number of BRAMs where half of the BRAM is equal to 18 bits
                    num_half_bram = math.ceil( (new_pruned_channels * num_bits + 18) / 18 )
                    num_brams = num_half_bram / 2

                    num_m = calculate_multipliers(new_pruned_channels, layer_name)

                    pruning_space.append(new_pruned_channels)
                    bit_space.append(num_bits)
                    bram_space.append(num_brams)
                    m_space.append(num_m)


        # Append the information to the precomputed space list
        precomputed_space.append({
            'layer': layer_name,
            'pruning_space': pruning_space,
            'bit_space': bit_space,
            'bram_space': bram_space,
            'm_space': m_space
        })

    return precomputed_space


def generate_configs_ncaltech(precomputed_space):
    """
    Generate all possible model configurations by taking the Cartesian product
    of each layer's pruning, bit, and BRAM options.
    """
    choices_per_layer = []
    for layer_info in precomputed_space:
        layer = layer_info['layer']
        # Create per-layer option dictionaries
        layer_choices = [
            {
                f"{layer}_pruning": p,
                f"{layer}_bits": b,
                f"{layer}_bram": br,
                f"{layer}_m": m
            }
            for p, b, br, m in zip(layer_info['pruning_space'], layer_info['bit_space'], layer_info['bram_space'], layer_info['m_space'])
        ]
        choices_per_layer.append(layer_choices)
    
    # Compute Cartesian product across layers
    configs = []
    for combo in itertools.product(*choices_per_layer):
        config = {}
        for opt in combo:
            config.update(opt)
        configs.append(config)
    return configs


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/ncaltech.yaml')
    precomputed_space = precompute_space_ncaltech(cfg, depth_start=0, depth_end=2)
    configs = generate_configs_ncaltech(precomputed_space)
    print(precomputed_space)
    # print(configs)
    print(len(configs))