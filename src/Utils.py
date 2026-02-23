import torch


def fed_avg_state_dicts(state_dicts, weights=None):
    num = len(state_dicts)
    if num == 0:
        raise ValueError("fed_avg_state_dicts: don't have any state_dict.")

    if weights is None:
        weights = [1.0] * num
    total_w = sum(weights)

    all_keys = set().union(*(sd.keys() for sd in state_dicts))
    avg_dict = {}

    for key in all_keys:
        acc = None
        for sd, w in zip(state_dicts, weights):
            if key not in sd:
                continue
            t = sd[key].float()
            if torch.isnan(t).any():
                t = torch.nan_to_num(t)
            t = t * w
            acc = t if acc is None else acc + t

        avg = acc / total_w

        orig = next(sd[key] for sd in state_dicts if key in sd)
        if orig.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            avg = avg.round().to(orig.dtype)
        else:
            avg = avg.to(orig.dtype)

        avg_dict[key] = avg

    return avg_dict
