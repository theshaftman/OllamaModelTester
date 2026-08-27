import OllamaModelTester as omt

i_models = ['tinyllama', 'gemma2:2b', 'llama3.2:1b', 'smollm2:135m']
i_prompt_text = """Summarize this text

Intel Corp. raised $20 billion in an upsized share sale, a third more than it was targeting when it announced the deal Monday morning.

The chipmaker priced the offering at $95 per share, according to a company statement. That represents a discount of 6.5% to Friday’s closing price, according to Bloomberg calculations. The share sale drew more than $100 billion in demand, people familiar with the matter said.
"""

i_generated_text = """Intel Corp. raised $20 billion in an upsized share sale, a third more than it was targeting when it announced the deal Monday morning"""
i_human_label = 'faithful' # faithful or hallucinated

i_metrics = [{
    'data': 'model_results',
    'columns': ['elapsed_time', 'ttft', 'tokens_per_second']
}, {
    'data': 'validation_results',
    'columns': ['overall_f1_score']
}, {
    'data': 'sql',
    'columns': ['faithfulness_score'],
    'project_id': 'llm-practical-experiment',
    'dataset_id': 'llm_model_evaluation',
    'sql': """
WITH models AS (
SELECT mr.model AS `model`
    , mr.faithfulness_score AS `faithfulness_score`
FROM `llm-practical-experiment.llm_model_evaluation.model_results` AS mr
LIMIT 1000
),
validations AS (
SELECT 'human' AS `model`
    , vr.faithfulness_score AS `faithfulness_score`
FROM `llm-practical-experiment.llm_model_evaluation.validation_results` AS vr
LIMIT 1000
)
SELECT m.`model`
    , m.`faithfulness_score`
FROM models AS m
UNION ALL
SELECT v.`model`
    , v.`faithfulness_score`
FROM validations AS v
    """
}]
i_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'white']

with omt.OllamaModelTester(host='127.0.0.1', port=11434, models=i_models, install_packages=False) as om_tester:
    om_tester.import_results_from_csv()
    """
    Create a IAM service account, export the Key as JSON, upload it in folder "credentials" and
    create a Google BigQuery Dataset to use the import export module of Google BigQuery.
    """
    om_tester.import_results_from_gbq('llm-practical-experiment', 'llm_model_evaluation')

    var_validate_elevator = om_tester.validate_evaluator(
        prompt_text=i_prompt_text,
        generated_text=i_generated_text,
        human_label=i_human_label
    )
    # om_tester.print_results(var_validate_elevator)

    om_tester.pull_models(models=None)
    om_tester.compare_models(prompt_text=i_prompt_text, models=None)
    # om_tester.print_results()

    om_tester.visualize_results(plot_type='bar', metrics=i_metrics, colors=i_colors, savefig_path=f'charts/bar_chart.png', max_cols_per_row=2)
    om_tester.visualize_results(plot_type='plot', metrics=i_metrics, colors=i_colors, savefig_path=f'charts/plot_chart.png', max_cols_per_row=2)
    om_tester.visualize_results(plot_type='scatter', metrics=i_metrics, colors=i_colors, savefig_path=f'charts/scatter_chart.png', max_cols_per_row=2)
    om_tester.visualize_results(plot_type='pie', metrics=i_metrics, colors=i_colors, savefig_path=f'charts/pie_chart.png', max_cols_per_row=2)

    om_tester.export_results_to_csv()
    om_tester.export_results_to_gqb('llm-practical-experiment', 'llm_model_evaluation', 'replace')
