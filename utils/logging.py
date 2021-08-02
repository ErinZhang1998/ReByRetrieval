import os 
import builtins
import decimal
import logging
import sys
import utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(log_to_file =  False, experiment_dir = "./"):
    print(log_to_file, experiment_dir)
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    _FORMAT = "%(levelname)s - {%(filename)s:%(funcName)s:%(lineno)d} - %(message)s"

    if du.is_master_proc():
        # Enable logging for the master process.
        if not log_to_file:
            logging.basicConfig(
                level=logging.INFO, format=_FORMAT, stream=sys.stdout
            )
        else:
            fname = os.path.join(experiment_dir, 'log.txt')
            
            # logging.root.handlers = [
            #     logging.StreamHandler(sys.stdout),
            #     logging.FileHandler(fname),
            # ]
            
            # logging.basicConfig(
            #     level=logging.INFO, 
            #     format=_FORMAT,
            # )

            logging.getLogger().setLevel(logging.INFO)
            fh = logging.FileHandler(fname, 'w+')
            # fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(_FORMAT)
            fh.setFormatter(formatter)
            logging.getLogger().addHandler(fh)
            stdh = logging.StreamHandler(sys.stdout)
            stdh.setFormatter(formatter)
            logging.getLogger().addHandler(stdh)
            
    else:
        # Suppress logging for non-master processes.
        _suppress_print()


def get_logger(name):
    return logging.getLogger(name)