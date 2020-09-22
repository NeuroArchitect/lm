
from . import train_model, train_tokenizer
from absl import app

SUBCOMMANDS = {
    'model': train_model,
    'tokenizer': train_tokenizer,
}

def parse_args(args, parser=None):
    
    subparsers = parser.add_subparsers(help="Available train commands", dest="training_type")
    for name, cmd in SUBCOMMANDS.items():
        cmd_parser = subparsers.add_parser(
            name,
            help=cmd.__doc__,
            # Do not make absl FLAGS available after the subcommand
            inherited_absl_flags=False,
        )
        cmd.parse_args(args, cmd_parser)


def main(args):
    cmd = SUBCOMMANDS.get(args.training_type, None)
    if cmd is None:
        app.usage(shorthelp=True, exitcode=-1)
        return
    return cmd.main(args)
    
