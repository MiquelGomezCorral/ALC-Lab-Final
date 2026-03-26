"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from scripts import get_transcriptions


def cmd_get_transcriptions(args: argparse.Namespace):
    """Call get_transcriptions with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    get_transcriptions(CONFIG)

def cmd_test(args):
    """Call test functions."""
    ...

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       get_transcriptions
    # ======================================================================================
    p_transcriptions = subparsers.add_parser("get-transcriptions", help="Get transcriptions for videos")
    
    p_transcriptions.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for transcription (default: 16)")

    p_transcriptions.set_defaults(func=cmd_get_transcriptions)


    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)