#!/usr/bin/env python3
import json
from pathlib import Path
import torch
import torchvision
from torchvision.datasets import StanfordCars


def main():
    root = Path("models/stanford_cars_raw")
    out = Path("models/stanford_cars_imagefolder")
    out.mkdir(parents=True, exist_ok=True)

    # Download datasets
    train = StanfordCars(root=str(root), split='train', download=True)
    test = StanfordCars(root=str(root), split='test', download=True)
    classes = train.classes

    def dump(ds):
        for img, idx in ds:
            cls = classes[idx].replace('/', '_').replace(' ', '_')
            d = out / cls
            d.mkdir(parents=True, exist_ok=True)
            fn = f"{len(list(d.glob('*.jpg'))):06d}.jpg"
            torchvision.utils.save_image(
                torchvision.transforms.ToTensor()(img),
                str(d / fn)
            )

    dump(train)
    dump(test)

    with open(out / 'labels_preview.json', 'w') as f:
        json.dump(classes, f, indent=2)

    print("Prepared ImageFolder at:", out)


if __name__ == "__main__":
    main()
