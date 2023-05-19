import os
import logging

from run_intro import run_intro

def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                        format='%(asctime)s %(levelname)s:\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("start main")

    sx_df = run_intro()

    logging.info('end main')


if __name__ == "__main__":
    main()
