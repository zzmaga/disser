"""One command dispatcher; heavyweight dependencies load only for the selected task."""
import argparse
import runpy
import sys

COMMANDS = {
    'serve': ('kazstyle.api.server', 'Start the local testing website'),
    'predict': ('kazstyle.inference.cli', 'Classify text with the same models as the website'),
    'build-data': ('kazstyle.data.build', 'Build a new balanced dataset version'),
    'audit-data': ('kazstyle.data.audit', 'Audit CSV and dataset structure'),
    'diagnose-data': ('kazstyle.data.diagnostics', 'Check overlaps and source confounding'),
    'import-hf': ('kazstyle.data.import_hf', 'Collect unlabelled candidates for review'),
    'train-classical': ('kazstyle.training.classical', 'Train TF-IDF baselines'),
    'train-transformer': ('kazstyle.training.transformer', 'Fine-tune Kaz-RoBERTa'),
    'compare': ('kazstyle.evaluation.compare', 'Compare compatible saved runs'),
    'verify-model': ('kazstyle.evaluation.verify', 'Verify offline checkpoint restoration'),
}


def main():
    parser = argparse.ArgumentParser(description='Kazakh text style project',
        epilog='Use COMMAND --help for options.')
    parser.add_argument('command', choices=COMMANDS, help='; '.join(f'{k}: {v[1]}' for k, v in COMMANDS.items()))
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f'{sys.argv[0]} {args.command}', *sys.argv[2:]]
    runpy.run_module(COMMANDS[args.command][0], run_name='__main__')


if __name__ == '__main__':
    main()
