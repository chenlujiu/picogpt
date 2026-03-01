import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    if os.environ.get("MYCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("MYCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir,  ".cache")
        nanochat_dir = os.path.join(cache_dir, "mychat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ["RANK", "LOCAL", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        device_type  = "cuda"
    elif torch.backend.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common"""
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for cuda but device_type is cuda"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is  not configured for mps but device_type is mps"

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    #  Distributed setup: Distributed Data Parallel, optional, and requires cuda
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id = device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size : {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass





