# OllamaModelTester

A comprehensive Python framework for testing, evaluating, and visualizing Ollama language models with built-in hallucination detection, performance metrics, and Google BigQuery integration.

---

## Features

- **Multi-Model Testing**: Test multiple Ollama models simultaneously (tinyllama, gemma2:2b, llama3.2:1b, smollm2:135m)
- **Performance Metrics**: Track TTFT (Time to First Token), tokens/second, latency (p50, p95)
- **Hallucination Detection**: Using NLI (Natural Language Inference) models
- **Data Export**: Export results to CSV and Google BigQuery
- **Visualization**: Built-in chart generation (bar, line, scatter, pie)
- **Cross-Platform**: Works on Windows, Linux, and macOS

---

## Prerequisites

- **Python**: 3.13 or higher
- **Ollama**: Installed and running ([Download](https://ollama.com/download))
- **Hardware**: 8GB+ RAM recommended (4GB minimum)
- **Disk Space**: 10GB+ for models

---

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install OllamaModelTester
```

## Complete Example

```bash
import OllamaModelTester as omt

# Define models and test data
i_models = ['tinyllama', 'gemma2:2b', 'llama3.2:1b', 'smollm2:135m']
i_prompt_text = """Original text"""

i_generated_text = """Generated text from human"""
i_human_label = 'faithful'  # faithful or hallucinated

i_metrics = [{
    'data': 'model_results',
    'columns': ['elapsed_time', 'ttft', 'tokens_per_second']
}, {
    'data': 'validation_results',
    'columns': ['overall_f1_score']
}]
i_colors = ['red', 'green', 'blue', 'purple']

with omt.OllamaModelTester(host='127.0.0.1', port=11434, models=i_models, install_packages=False) as om_tester:
    # Validate evaluator
    var_validate_elevator = om_tester.validate_evaluator(
        prompt_text=i_prompt_text,
        generated_text=i_generated_text,
        human_label=i_human_label
    )

    # Pull and test models
    om_tester.pull_models(models=None)
    # Compare all models
    om_tester.compare_models(prompt_text=i_prompt_text, models=None)

    # Generate visualizations
    om_tester.visualize_results(plot_type='bar', metrics=i_metrics, colors=i_colors, savefig_path='charts/bar_chart.png', max_cols_per_row=2)
    om_tester.visualize_results(plot_type='plot', metrics=i_metrics, colors=i_colors, savefig_path='charts/plot_chart.png', max_cols_per_row=2)

    # Export results
    om_tester.export_results_to_csv()
```

## Project Structure
```
OllamaModelTester/
├── src/
│   └── OllamaModelTester/
│       ├── __init__.py          # Package entry point
│       ├── main.py              # Main OllamaModelTester class
│       └── model_visualizer.py  # Visualization utilities
├── tests/
│   └── test_ollama.py          # Unit tests
├── credentials/
│   └── service-account-key.json # Google Cloud credentials
├── documents/
│   ├── model_results.csv       # Model test results
│   └── validation_results.csv  # Validation results
├── charts/
│   ├── bar_chart.png           # Generated visualizations
│   ├── plot_chart.png
│   ├── scatter_chart.png
│   └── pie_chart.png
├── pyproject.toml              # Build configuration
├── README.md                   # Documentation
├── LICENSE                     # MIT License
└── requirements.txt            # Dependencies
```

## Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository
- Create a feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request