import os
import logging

from run_intro import RunIntroOutput, run_intro
from run_part1 import run_part1, RunPart1Input


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                        format='%(asctime)s %(levelname)s:\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("start main")

    runIntroOutput: RunIntroOutput = run_intro()

    run_part1(RunPart1Input(runIntroOutput.sx_df, runIntroOutput.t_min, runIntroOutput.t_max, runIntroOutput.DT, runIntroOutput.dt))

    logging.info('end main')


if __name__ == "__main__":
    main()
