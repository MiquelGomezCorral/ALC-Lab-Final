"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from scripts import get_transcriptions, get_ocr_transcriptions, get_extra_sounds


def cmd_get_transcriptions(args: argparse.Namespace):
    """Call get_transcriptions with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    get_transcriptions(CONFIG)

def cmd_get_ocr(args):
    """Call get_ocr_transcriptions with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    get_ocr_transcriptions(CONFIG)

def cmd_get_extra_sounds(args):
    """Call get_extra_sounds with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    get_extra_sounds(CONFIG)

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
    #                                       get_ocr
    # ======================================================================================
    p_ocr = subparsers.add_parser("get-ocr", help="Get OCR results for videos")
    
    p_ocr.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for OCR (default: 16)")

    p_ocr.set_defaults(func=cmd_get_ocr)

    # ======================================================================================
    #                                       get_extra_sounds
    # ======================================================================================
    p_extra_sounds = subparsers.add_parser("get-extra-sounds", help="Get extra sound features for videos")
    p_extra_sounds.set_defaults(func=cmd_get_extra_sounds)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)