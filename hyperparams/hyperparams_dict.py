

hp_categorical = {
    "optimisers": {"names": ["SGD", "Adam", "RMSprop", "AdamW"]},
    "lr_schedulers": {
        "names": ["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
        "params": {
            "StepLR": {"step_size": 5, "gamma": 0.5},
            "ExponentialLR": {"gamma": 0.9},
            "ReduceLROnPlateau": {"mode": 'min', "factor": 0.5, "patience": 3},
            "CosineAnnealingLR": {"T_max": 10},
        }
    },
}