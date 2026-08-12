from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
)


def create_model(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    weights = (
        EfficientNet_V2_S_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_v2_s(weights=weights)

    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def freeze_all_features(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_last_feature_blocks(
    model: nn.Module,
    num_blocks: int,
) -> None:
    freeze_all_features(model)

    if num_blocks <= 0:
        return

    for block in model.features[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    test_model = create_model(
        num_classes=120,
        pretrained=False,
    )
    sample = torch.randn(2, 3, 384, 384)

    output = test_model(sample)

    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        f"{count_trainable_parameters(test_model):,}",
    )
