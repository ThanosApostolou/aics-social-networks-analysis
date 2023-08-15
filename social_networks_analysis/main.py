import os
import logging

from run_intro import RunIntroOutput, run_intro
from run_part1 import run_part1, RunPart1Input
from run_part2 import RunPart2Output, run_part2, RunPart2Input
from social_networks_analysis.run_part3 import RunPart3Input, run_part3


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                        format='%(asctime)s %(levelname)s:\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("start main")
    # run intro
    run_intro_output: RunIntroOutput = run_intro()
    # run part1
    run_part1(RunPart1Input(run_intro_output.sx_df, run_intro_output.t_min, run_intro_output.t_max,
              run_intro_output.DT, run_intro_output.dt, run_intro_output.time_spans))
    # run part2
    run_part2(RunPart2Input(run_intro_output.sx_df, run_intro_output.t_min, run_intro_output.t_max,
                            run_intro_output.DT, run_intro_output.dt, run_intro_output.time_spans))
    # run part3
    run_part3(RunPart3Input(run_intro_output.sx_df, run_intro_output.t_min, run_intro_output.t_max,
                            run_intro_output.DT, run_intro_output.dt, run_intro_output.time_spans))

    logging.info('end main')


if __name__ == "__main__":
    main()
