"""A little utility library to make for easy logging to wandb."""
import math
import secrets
import time

import torch
import wandb

last_time = None
last_step = None
LOG_BACKOFF_POINT = 5000
LOG_BACKOFF_FACTOR = 20

DEFAULT_FREQ = 1

def make_video_grid(
    tensor,
    num_images_per_row: int = 10,
    padding: int = 2,
    pad_value: int = 0,
):
    """
    This is a repurposed implementation of `make_grid` from torchvision to work with videos.
    """
    n_maps, sequence_length, num_channels, height, width = tensor.size()
    x_maps = min(num_images_per_row, n_maps)
    y_maps = int(math.ceil(float(n_maps) / x_maps))
    height, width = int(height + padding), int(width + padding)
    grid = tensor.new_full(
        (sequence_length, num_channels, height * y_maps + padding, width * x_maps + padding), pad_value
    )
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid.narrow(2, y * height + padding, height - padding).narrow(
                3, x * width + padding, width - padding
            ).copy_(tensor[k])
            k += 1
    return grid


def effective_freq(step, freq):
    # Simple logging backoff logic.
    if step > LOG_BACKOFF_POINT and freq != 1:
        freq *= LOG_BACKOFF_FACTOR
    return freq


def check_log_interval(step, freq):
    freq = effective_freq(step, freq)
    return step % freq == 0


def download_file(run_id, project_name, filename=None):
    api = wandb.Api()
    run = api.run(f"sourceress/{project_name}/{run_id}")
    # We save to a random directory to avoid contention issues if there's
    # potentially multiple processes downloading the same file at the same time.
    # TODO: clean these files up
    path = run.file(filename).download(replace=True, root=f"./data/{secrets.token_hex(10)}").name
    return path


def log_histogram(tag, value, step, freq=DEFAULT_FREQ):
    if not check_log_interval(step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: value}, step=step)


def log_video(trainer, tag, batch, freq=DEFAULT_FREQ, normalize=False):
    # Expects b, t, c, h, w
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return

    if normalize:
        min_v = torch.min(batch)
        range_v = torch.max(batch) - min_v
        if range_v > 0:
            batch = (batch - min_v) / range_v
        else:
            batch = torch.zeros(batch.size())

    # batch = preprocess_batch(batch).permute(0, 2, 1, 3, 4) + .5
    frames = make_video_grid(batch, num_images_per_row=4, pad_value=1)

    # This should be in range 0-1
    if type(frames) == torch.Tensor:
        frames = frames.detach()
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.cpu()
    trainer.logger.experiment.log({tag: wandb.Video(frames, fps=4, format="gif")}, step=trainer.global_step)


def log_image(trainer, tag, value, freq=DEFAULT_FREQ):
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: wandb.Image(value)}, step=trainer.global_step)


def log_table(trainer, tag, value, freq=DEFAULT_FREQ):
    # value should be a 1d tensor, in this current implementation. can add more columns in the future.
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    columns = ["test"]
    rows = [[x] for x in value]
    table = wandb.Table(data=rows, columns=columns)
    trainer.logger.experiment.log({tag: table}, step=trainer.global_step)


def log_scalar(tag, value, step, freq=DEFAULT_FREQ):
    if not check_log_interval(step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: value}, step=step)


def log_iteration_time(batch_size, step, freq=DEFAULT_FREQ):
    """Call this once per training iteration."""
    global last_time
    global last_step
    if not check_log_interval(step, freq):
        return

    if last_time is None:
        last_time = time.time()
        last_step = step
    else:
        if step == last_step:
            return
        dt = (time.time() - last_time) / (step - last_step)
        last_time = time.time()
        last_step = step
        log_scalar("timings/iterations-per-sec", 1 / dt, step, freq=1)
        log_scalar("timings/samples-per-sec", batch_size / dt, step, freq=1)


def watch(model, freq=50):
    # wandb.watch(model, log="all")
    wandb.watch(model, "all", log_freq=freq)
