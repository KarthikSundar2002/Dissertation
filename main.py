from options.args import argument_parser
import os

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()

    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    