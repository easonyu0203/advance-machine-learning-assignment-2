import torch
from experiment.config import ExperimentConfig
import pickle

from experiment.experiment import ModelTrainingExperiment

# Define parameters for experiments
dataset_names = ['MNIST5', 'MNIST6', 'CIFAR']
mnist_models = ['mnist', 'mnist_nal']
cifar_models = ['cifar', 'cifar_nal']
loss_fn_names = ['ce', 'gce']
gce_alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
MNIST5_transition_matrix = torch.Tensor([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
MNIST6_transition_matrix = torch.Tensor([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])

try:
    with open('experiment_results.pickle', 'rb') as handle:
        experiment_results = pickle.load(handle)
except FileNotFoundError:
    experiment_results = {}


def run_experiments():
    # Load existing results if available
    try:
        with open('experiment_results.pickle', 'rb') as handle:
            experiment_results = pickle.load(handle)
    except FileNotFoundError:
        experiment_results = {}

    for dataset in dataset_names:
        models = mnist_models if dataset in ['MNIST5', 'MNIST6'] else cifar_models
        for model in models:
            for loss_fn in loss_fn_names:
                alphas = [None]
                if loss_fn == 'gce':
                    alphas = gce_alphas

                for alpha in alphas:
                    use_transition_matrix = False
                    if 'nal' in model and (dataset in ['MNIST5', 'MNIST6']):
                        use_transition_matrix = True

                    # Run trainer for both cases: with and without transition matrix
                    for matrix in [True, False] if use_transition_matrix else [False]:
                        config_key = generate_config_key(dataset, model, loss_fn, alpha, matrix)

                        # Checkpoint: Skip if trainer already run
                        if config_key in experiment_results:
                            print(f"Skipping already run trainer: {config_key}")
                            continue

                        print(f"\n\nRunning trainer for {config_key}\n\n")
                        experiment_results[config_key] = run_single_experiment(dataset, model, loss_fn, alpha, matrix)

    # Save updated results
    with open('experiment_results.pickle', 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_config_key(dataset, model, loss_fn, alpha, use_transition_matrix):
    return f"{dataset}_{model}_{loss_fn}_{'alpha_' + str(alpha) if alpha else 'no_alpha'}_{'tm' if use_transition_matrix else 'no_tm'}"


def run_single_experiment(dataset, model, loss_fn, alpha, use_transition_matrix):
    transition_matrix = None
    if use_transition_matrix:
        transition_matrix = MNIST5_transition_matrix if dataset == 'MNIST5' else MNIST6_transition_matrix

    config = ExperimentConfig(
        model_name=model,
        dataset_name=dataset,
        loss_fn_name=loss_fn,
        learning_rate=0.0001,
        batch_size=512,
        num_epochs=15,
        gce_alpha=alpha,
        nal_transition_matrix=transition_matrix
    )
    experiment = ModelTrainingExperiment(config)
    experiment.run_experiment()
    return experiment.calculate_mean_std_metrics()


def look_up_experiment(dataset, model, loss_fn, alpha, use_transition_matrix):
    """
        Retrieves the results of a specific trainer based on its configuration.

        Parameters:
        - dataset (str): The name of the dataset used in the trainer.
                         Acceptable values are 'MNIST5', 'MNIST6', and 'CIFAR'.
        - model (str): The model type used in the trainer.
                       For 'MNIST5' and 'MNIST6' datasets, acceptable values are 'mnist' and 'mnist_nal'.
                       For the 'CIFAR' dataset, acceptable values are 'cifar' and 'cifar_nal'.
                       'mnist_nal' and 'cifar_nal' refer to models with an additional NAL layer.
        - loss_fn (str): The loss function used in the trainer.
                         Acceptable values are 'ce' for cross-entropy and 'gce' for generalized cross-entropy.
        - alpha (float or None): The alpha value used with generalized cross-entropy ('gce').
                                 It should be a float in the range [0.1, 0.99] or None if 'ce' is used as the loss function.
        - use_transition_matrix (bool): Indicates whether the trainer used a transition matrix with the NAL models.
                                        Acceptable values are True or False. This is relevant only for 'mnist_nal' and 'cifar_nal' models.
                                        For 'mnist' and 'cifar' models, this parameter should always be False.

        Returns:
        dict or None: A dictionary containing the results (mean and standard deviation of metrics) of the trainer if it exists,
                      or None if no trainer matches the provided configuration.

        Example Usage:
        result = look_up_experiment('MNIST5', 'mnist_nal', 'gce', 0.9, True)
        """
    config_key = generate_config_key(dataset, model, loss_fn, alpha, use_transition_matrix)
    return experiment_results.get(config_key, None)


if __name__ == "__main__":

    # Run all experiments
    run_experiments()


