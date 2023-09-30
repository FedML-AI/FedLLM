from typing import AnyStr, IO, Optional, Sequence, TextIO

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from queue import Queue
from subprocess import CalledProcessError, PIPE, Popen
import sys
from threading import Thread

import torch.cuda

TORCH_DISTRIBUTED_DEFAULT_PORT = 29500


def read_stream(input_stream: IO[AnyStr], message_queue: Queue, output_stream: TextIO) -> None:
    try:
        for line in input_stream:
            message_queue.put_nowait((line, output_stream))
    except ValueError as e:
        if not input_stream.closed:
            # raise exception if the stream is not closed
            raise e
    finally:
        # send closing message
        message_queue.put_nowait((None, output_stream))


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

    # see https://stackoverflow.com/a/28319191
    message_queue = Queue()

    with Popen(
            [
                sys.executable,
                *cmd,
                *sys.argv[1:],
            ],
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            universal_newlines=True,
            env=os.environ
    ) as proc:
        thread_stdout = Thread(target=read_stream, args=(proc.stdout, message_queue, sys.stdout))
        thread_stderr = Thread(target=read_stream, args=(proc.stderr, message_queue, sys.stderr))
        thread_stdout.start()
        thread_stderr.start()

        is_stdout_done = False
        is_stderr_done = False

        # Adapted from https://stackoverflow.com/a/4896288 and
        # and https://stackoverflow.com/a/51668895
        # and https://stackoverflow.com/a/18422264
        while not is_stdout_done or not is_stderr_done:
            line, output_file = message_queue.get()
            message_queue.task_done()

            if line is not None:
                print(line, end="", file=output_file, flush=True)
            elif output_file is sys.stdout:
                is_stdout_done = True
            elif output_file is sys.stderr:
                is_stderr_done = True

    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, proc.args)

    print("Closing IO threads...", flush=True)
    thread_stdout.join()
    thread_stderr.join()
    # see https://superfastpython.com/thread-queue-task-done-join/
    print("Closing message queue...", flush=True)
    message_queue.join()

    print(f"{__file__} done.")
    return proc.returncode


if __name__ == '__main__':
    exit(main())
