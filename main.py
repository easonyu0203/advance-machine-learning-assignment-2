from experiment import ExperimentConfig, TrainEvalExperiment


experiment_config = ExperimentConfig("mnist", "MNIST6", 0.001, 64, 1)
experiment = TrainEvalExperiment(experiment_config)
experiment.run_experiment()