import pandas as pd
from omegaconf import OmegaConf



def select_best_model(df, cfg, acc_dif=0.02):


    baseline_model = df[
        (df['conv1_pruning'] == cfg.conv1.out_channels) &
        (df['conv1_bits'] == cfg.conv1.num_bits) &
        (df['conv2_pruning'] == cfg.conv2.out_channels) &
        (df['conv2_bits'] == cfg.conv2.num_bits) &
        (df['conv3_pruning'] == cfg.conv3.out_channels) &
        (df['conv3_bits'] == cfg.conv3.num_bits) &
        (df['conv4_pruning'] == cfg.conv4.out_channels) &
        (df['conv4_bits'] == cfg.conv4.num_bits) &
        (df['conv5_pruning'] == cfg.conv5.out_channels) &
        (df['conv5_bits'] == cfg.conv5.num_bits) 
    ]

    baseline_model_accuracy = baseline_model['accuracy'].values[0]
    selected_models = df[df['accuracy'] >= baseline_model_accuracy - 0.02]

    # sort selected models by brams
    selected_models = selected_models.sort_values(by='brams', ascending=True)
    best_model = selected_models.iloc[0:1]

    return best_model.to_dict(orient='records')[0], baseline_model.to_dict(orient='records')[0]


if __name__ == "__main__":
    # Load the dataframe
    df = pd.read_csv('results_cifar10-dvs.csv')

    # Load the config
    cfg = OmegaConf.load('configs/cifar.yaml')

    # Select the best model
    best_model = select_best_model(df, cfg)
    print(best_model)
