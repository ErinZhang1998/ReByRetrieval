import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for reconstruction model.
"""


def build_model(args):
    assert (
        args.num_gpus <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    # import pdb; pdb.set_trace()
    name = args.model_config.model_name
    model = MODEL_REGISTRY.get(name)(args)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    return model
