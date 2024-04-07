from importlib import resources

import numpy as np

label_names = np.array(
    [
        "Reach",
        "Flip",
        "Touch_object",
        "Push",
        "Pull",
        "Pre_grasp",
        "Grasp",
        "Lift_with_grasp",
        "Transport_forward",
        "Transport_down",
        "Release",
        "Go_back_to_setup",
        "Pre_rotate",
        "Rotate",
        "Shake_a",
        "Shake_b",
        "Twist",
        "Pre_touch",
        "Pour",
        "Side_place",
    ]
)
labels = label_names.reshape(-1, 1)
