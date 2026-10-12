"""Postlaunch entrypoint for the clean reviewer's fixed, bounded reproduction."""
import argparse
from .independent_clean_review import reproduce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    reproduce(args.deadline)


if __name__ == '__main__':
    main()
