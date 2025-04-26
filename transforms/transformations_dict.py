from torchvision.transforms import v2 as transformsv2
from transforms.transforms import cutout, gaussian_noise, hide_and_seek

rotation_degree = 45
translation = (0.2, 0.2) 
shearing = (10, 20, 0, 10)
kernel_size = 5
CUTOUT_SIZE = 16
CUTOUT_PROB = 1


transformations = {
    # Geometric
    "rotation": transformsv2.RandomRotation(rotation_degree),
    "translation": transformsv2.RandomAffine(degrees=0, translate=translation),
    "shearing": transformsv2.RandomAffine(degrees=0, shear=shearing),

    # Non geometric
    "horizontal_flip": transformsv2.RandomHorizontalFlip(),
    "vertical_flip": transformsv2.RandomVerticalFlip(),
    "crop": transformsv2.RandomCrop(32, padding=4),
    "color_jitter": transformsv2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "noise_injection": gaussian_noise,
    "kernel": transformsv2.GaussianBlur(kernel_size=kernel_size),

    # Erasing
    "random_erasing": transformsv2.RandomErasing(),
    "cutout": cutout(mask_size=CUTOUT_SIZE, p=CUTOUT_PROB),
    "hide_and_seek": hide_and_seek,
    
}