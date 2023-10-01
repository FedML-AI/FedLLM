from typing import Optional, Sequence

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
import sys

import torch.cuda

TORCH_DISTRIBUTED_DEFAULT_PORT = 29500


def parse_args(args: Optional[Sequence[str]] = None, namespace: Optional[Namespace] = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--rank",
        dest="rank",
        type=int,
        help="Distributed rank."
    )
    parser.add_argument(
        "--master_addr",
        dest="master_addr",
        type=str,
        default="localhost",
        help="The IP address of the master node."
    )
    parser.add_argument(
        "--master_port",
        dest="master_port",
        type=str,
        default=None,
        help="The port of the master node."
    )
    parser.add_argument(
        "--num_nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Number of nodes for this client/aggregator."
    )
    parser.add_argument(
        "--num_gpus",
        dest="num_gpus",
        type=int,
        default=None,
        help="Number of GPUs in this node/device. If unspecified, will use all GPUs."
             " This will be overrode by environment variable `CUDA_VISIBLE_DEVICES`."
    )

    # parse args
    output_args, *_ = parser.parse_known_args(args=args, namespace=namespace)

    # update args
    if output_args.master_port is None:
        output_args.master_port = int(os.getenv("TORCH_DISTRIBUTED_DEFAULT_PORT", TORCH_DISTRIBUTED_DEFAULT_PORT))
        output_args.master_port += output_args.rank

    if output_args.num_gpus is None:
        output_args.num_gpus = torch.cuda.device_count()

    return output_args


def main() -> int:
    # go to project root directory
    os.chdir(Path(__file__).parent)

    # update environment variables
    if len(os.getenv("WANDB_MODE", "")) == 0:
        os.environ["WANDB_MODE"] = "disabled"

    # parse args
    args = parse_args()

    print(
        f"master_addr: {args.master_addr},"
        f" master_port: {args.master_port},"
        f" num_nodes: {args.num_nodes},"
        f" num_gpus: {args.num_gpus}"
    )

    """
        process = ClientConstants.exec_console_with_shell_script_list(
            [
                python_program,         # python
                entry_fill_full_path,   # ./run_mlops.py
                "--cf",                 # --cf
                conf_file_full_path,    # $mlops_path/fedml_config.yaml
                "--rank",               # --rank
                str(dynamic_args_config["rank"]), # rank
                "--role",               # --role
                "client",               # client
            ],
        python run_mlops.py --cf fedml_config/fedml_config.yaml --rank 0 --role client
    """
    cmd = []
    if args.num_gpus > 0:
        cmd.extend([
            f"-m",
            f"deepspeed.launcher.runner",
            f"--master_addr", f"{args.master_addr}",
            f"--master_port", f"{args.master_port}",
        ])

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # when `CUDA_VISIBLE_DEVICES` is not specified, use all GPUs by setting `--num_nodes`
            cmd.extend([
                f"--num_nodes", f"{args.num_nodes}",
                f"--num_gpus", f"{args.num_gpus}",
            ])
        else:
            # see https://github.com/microsoft/DeepSpeed/issues/662
            # use `--include` to select GPUs and unset `CUDA_VISIBLE_DEVICES`
            cmd.extend([
                f"--include", f"{args.master_addr}:{os.getenv('CUDA_VISIBLE_DEVICES')}",
            ])
            os.environ.pop("CUDA_VISIBLE_DEVICES")

    cmd.extend([
        "run_fedllm.py",  # main program
    ])

    print(f"cmd = {cmd}")
    print(f"sys.argv = {sys.argv}")

    proc = subprocess.run(
        [
            sys.executable,
            *cmd,
            *sys.argv[1:],
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ
    )

    print(f"{__file__} done.")
    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, proc.args)

    return proc.returncode


if __name__ == '__main__':
    exit(main())
