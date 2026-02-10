import torch
import numpy as np

def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    target_wo_nan = torch.nan_to_num(target)
    loss = (prediction - target_wo_nan) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.sum(loss)

def mape(prediction: torch.Tensor, target: torch.Tensor, max: float, min: float) -> torch.Tensor:

    mask = ~torch.isnan(target)

    filtered_prediction = (prediction[mask] + 1) / 2 * (max - min) + min
    filtered_target = (target[mask] + 1) / 2 * (max - min) + min
    relative_errors = torch.abs((filtered_prediction - filtered_target) / filtered_target)

    return torch.mean(relative_errors)

def rmse(prediction: torch.Tensor, target: torch.Tensor, max: float, min: float, norm_target: bool = True) -> torch.Tensor:

    mask = ~torch.isnan(target)

    filtered_prediction = (prediction[mask] + 1) / 2 * (max - min) + min
    if norm_target:
        filtered_target = (target[mask] + 1) / 2 * (max - min) + min
    else:
        filtered_target = target[mask]
    mse = torch.nn.functional.mse_loss(filtered_prediction, filtered_target, reduction='mean')

    return torch.sqrt(mse)

def correlation_coefficient(prediction: torch.Tensor, target: torch.Tensor, max: float, min: float) -> torch.Tensor:

    mask = ~torch.isnan(target)

    filtered_prediction = (prediction[mask] + 1) / 2 * (max - min) + min
    filtered_target = (target[mask] + 1) / 2 * (max - min) + min

    mean_prediction = torch.mean(filtered_prediction)
    mean_target = torch.mean(filtered_target)

    covariance = torch.mean((filtered_prediction - mean_prediction) * (filtered_target - mean_target))

    std_prediction = torch.std(filtered_prediction)
    std_target = torch.std(filtered_target)

    return covariance / (std_prediction * std_target)

def nanmean(prediction: torch.Tensor, target: torch.Tensor, max: float, min: float) -> torch.Tensor:

    mask = ~torch.isnan(target)

    filtered_prediction = (prediction[mask] + 1) / 2 * (max - min) + min
    filtered_target = (target[mask] + 1) / 2 * (max - min) + min

    return torch.mean(filtered_prediction), torch.mean(filtered_target)
