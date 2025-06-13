# Copyright (c) EEEM071, University of Surrey

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--SRM-path",
        type=str,
        default="~/SRM17149.ckpt",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="LSG-Train-run",
        help="name of the experiment, used for logging and saving results",
    )


    parser.add_argument(
        "--root-dir",
        type=str,
        default="/scratch/ks02450",
        help="root directory for saving results and models",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful when restart)",
    )

    parser.add_argument(
        "--train-batch-size", default=2, type=int, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", default=1, type=int, help="test batch size"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        metavar="PATH",
        help="resume from a checkpoint",
    )

    return parser
