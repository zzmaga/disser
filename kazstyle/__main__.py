"""One command dispatcher; heavyweight dependencies load only for the selected task."""
import argparse
import runpy
import sys

COMMANDS = {
    'build-training-views': ('kazstyle.data.training_views', 'Controlled train-length ablation with byte-identical parent evaluation CSVs'),
    'build-source-holdout': ('kazstyle.data.source_holdout', 'Freeze a group-disjoint publisher-holdout development diagnostic'),
    'merge-candidates': ('kazstyle.data.pool', 'Compose a provisional development pool from checksummed sources and exclusions'),
    'collect-university-forms': ('kazstyle.data.university_forms', 'Save a reserved university form source with validated DOCX extraction'),
    'prepare-views': ('kazstyle.data.views', 'Freeze exact text views for annotation within every registered tokenizer limit'),
    'apply-prose-metadata': ('kazstyle.data.provenance', 'Separate explicit author credits from prose and preserve grouping provenance'),
    'collect-business': ('kazstyle.data.business', 'Collect Kazakh document templates for review from explicit genre sections'),
    'audit-overlap': ('kazstyle.data.overlap', 'Find exact repeats and shared passages before adding new texts'),
    'annotate-morphology': ('kazstyle.data.morphology', 'Annotate fixed inputs with local Kazakh Stanza models'),
    'evaluate-morphology': ('kazstyle.evaluation.morphology', 'Compare raw words, tokenizer output and lemmas on fixed splits'),
    'audit-experiment-code': ('kazstyle.evaluation.code_audit', 'Verify computational source snapshots across a completed suite'),
    'gpu-check': ('kazstyle.training.gpu_check', 'Check CUDA training memory using discarded train-only steps'),
    'export-figures': ('kazstyle.evaluation.figures', 'Export figures from verified repeated experiment results'),
    'build-manuscript': ('kazstyle.evaluation.manuscript', 'Build the working manuscript from recorded evidence'),
    'project-status': ('kazstyle.evaluation.project_status', 'Build the local project report navigation page'),
    'collect-prose': ('kazstyle.data.prose', 'Collect bounded prose/story candidates from verified category listings'),
    'prepare-deployment': ('kazstyle.inference.deployment', 'Prepare a profile from complete verified runs and validation selection'),
    'audit-linear': ('kazstyle.evaluation.linear_audit', 'Inspect saved linear features and exact score contributions'),
    'evaluate-diagnostics': ('kazstyle.evaluation.diagnostics', 'Check known user examples and fixed name/date variants'),
    'build-shared': ('kazstyle.data.shared', 'Preserve document splits and fit shared text to multiple tokenizers'),
    'train-suite': ('kazstyle.training.suite', 'Run a frozen multi-seed experiment plan sequentially'),
    'summarize-seeds': ('kazstyle.evaluation.repeated', 'Summarize all planned seeds and group bootstrap intervals'),
    'collect-forum': ('kazstyle.data.forum', 'Collect or re-screen public question bodies for independent review'),
    'annotate': ('kazstyle.data.annotation', 'Build blind review packets or compare two independent annotations'),
    'collect-journals': ('kazstyle.data.journals', 'Collect Kazakh research abstracts as unreviewed candidates'),
    'screen-long-chat': ('kazstyle.data.chat_review', 'Screen longer messages for review, without training models'),
    'evaluate-external': ('kazstyle.evaluation.external', 'Evaluate frozen external texts without changing weights'),
    'apply-review': ('kazstyle.data.review', 'Apply explicit decisions to a hash-matched candidate snapshot'),
    'evaluate-cohorts': ('kazstyle.evaluation.cohorts', 'Measure fixed predictions by length and publisher'),
    'build-clean': ('kazstyle.data.build_v3', 'Build balanced text-only splits with separate provenance'),
    'collect-data': ('kazstyle.data.collect', 'Collect article bodies from explicit genre sections'),
    'import-sources': ('kazstyle.data.import_sources', 'Screen downloaded news and informal messages'),
    'download-data': ('kazstyle.data.download', 'Download pinned public source files with a byte limit'),
    'prepare-data': ('kazstyle.data.prepare', 'Clean and quarantine raw records before dataset construction'),
    'serve': ('kazstyle.api.server', 'Start the local testing website'),
    'predict': ('kazstyle.inference.cli', 'Classify text with the same models as the website'),
    'build-data': ('kazstyle.data.build', 'Build a new balanced dataset version'),
    'audit-data': ('kazstyle.data.audit', 'Audit CSV and dataset structure'),
    'diagnose-data': ('kazstyle.data.diagnostics', 'Check overlaps and source confounding'),
    'import-hf': ('kazstyle.data.import_hf', 'Collect unlabelled candidates for review'),
    'train-classical': ('kazstyle.training.classical', 'Train TF-IDF baselines'),
    'train-transformer': ('kazstyle.training.transformer', 'Fine-tune a registered local or multilingual encoder'),
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
