import importlib
import os
from typing import Optional, List, Dict, Any


class ModelVisualizer:
    """
    Meeting 4
    Public class ModelVisualizer to build static charts by given data
    """
    DEFAULT_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
                      '#BC4B51', '#5D576B', '#F4A261', '#264653', '#E76F51']

    @staticmethod
    def __ensure_1d(arr) -> Any:
        """Ensure array is 1D, flatten if multi-dimensional."""
        if hasattr(arr, 'ndim') and arr.ndim > 1:
            return arr.flatten()
        return arr

    @staticmethod
    def _group_by_model(data, cols, metric) -> Dict[str, Any]:
        """Group columns by model name and aggregate values."""
        model_data = {}
        for c in cols:
            model_name = c.replace(f'_{metric}', '')
            vals = ModelVisualizer.__ensure_1d(data[c].dropna().values)
            if model_name not in model_data:
                model_data[model_name] = []
            model_data[model_name].append(vals)
        return model_data

    @staticmethod
    def visualize_chart(
            plot_type: str = 'bar',
            metrics: List[str] = None,
            colors: Optional[List[str]] = None,
            data: Any = None,
            savefig_path: str = None,
            max_cols_per_row: int = 3,
            show_figure: bool = False,
            imported_modules: Dict[str, Any] = None,
            os_path: str = os.path.dirname(os.path.abspath(__file__)),
            **options
    ) -> Any:
        """
        Public method to visualize and save charts.

        Args:
            plot_type (str): Select plot type ('bar', 'plot', 'scatter', 'pie')
            metrics (List[Dict[str, Any]]): Metrics to extract and compare the data
            colors: (Optional[List[Dict[str, Any]]]): List of predefined colors to color the AI model or Human-labelled chart with the data
            data (pd.DataFrame): DataFrame with the data to process
            savefig_path (str): Path to save the figures
            max_cols_per_row (int): Define maximum number of charts per row
            show_figure (bool): Show figure
            imported_modules (Dict[str, Any]): A dictionary with imported modules
            os_path (str): Standard folder to create the charts in
            **options: Keyword arguments to accept any number of named (keyword) inputs, packed into a standard dictionary

        Returns:
            plt: Matplotlib Pyplot of the built chart
        """
        
        matplotlib = imported_modules['matplotlib']
        plt = importlib.import_module('matplotlib.pyplot')
        numpy = imported_modules['numpy']
        np = numpy
        pyarrow = imported_modules['pyarrow']


        if colors is None:
            colors = ModelVisualizer.DEFAULT_COLORS

        # Get unique columns while preserving order
        var_unique_columns = []
        seen = set()
        for item in metrics:
            for col in item['columns']:
                if col not in seen:
                    seen.add(col)
                    var_unique_columns.append(col)

        # Setup subplots
        n_cols = min(max_cols_per_row, len(var_unique_columns))
        n_rows = (len(var_unique_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, metric in enumerate(var_unique_columns):
            ax = axes[idx]

            # Get columns for this metric
            cols = [c for c in data.columns if metric in c]

            if not cols:
                ax.set_title(f'{metric.upper()} (No Data)')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.grid(True, alpha=0.3)
                continue

            model_data = {}
            for c in cols:
                model_name = c.replace(f'_{metric}', '')
                vals = ModelVisualizer.__ensure_1d(data[c].dropna().values)
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append(vals)

            # Create unique names list
            names_list = list(model_data.keys())

            # Aggregate values (take first occurrence)
            values = []
            for model_name in names_list:
                model_vals = model_data[model_name]
                # Take first occurrence (prefer model_results over validation)
                combined_vals = model_vals[0]
                values.append(combined_vals)

            if plot_type == 'bar':
                if values:
                    max_len = max([len(v) for v in values]) if values else 0
                    x = np.arange(len(names_list))
                    width = 0.8 / max_len if max_len > 0 else 0.8

                    for i in range(max_len):
                        vals = [v[i] if i < len(v) else 0 for v in values]
                        if len(vals) == len(x):
                            bar_colors = [colors[j % len(colors)] for j in range(len(vals))]
                            ax.bar(x + i * width, vals, width, color=bar_colors,
                                   alpha=0.7, edgecolor='white', linewidth=1)

                    ax.set_xticks(x + width * (max_len - 1) / 2)
                    ax.set_xticklabels(names_list, rotation=45, ha='right')

            elif plot_type == 'plot':
                for i, name in enumerate(names_list):
                    # Get values for this model
                    model_vals = model_data[name]
                    vals = model_vals[0]  # Take first
                    ax.plot(range(len(vals)), vals, marker='o',
                        label=name, color=colors[i % len(colors)])
                ax.legend()

            elif plot_type == 'scatter':
                for i, name in enumerate(names_list):
                    model_vals = model_data[name]
                    vals = model_vals[0]  # Take first
                    ax.scatter(range(len(vals)), vals,
                             color=colors[i % len(colors)], label=name, s=50)
                ax.legend()

            elif plot_type == 'pie':
                values = []
                for name in names_list:
                    model_vals = model_data[name]
                    val = model_vals[0].sum() if len(model_vals[0]) > 0 else 0
                    values.append(val)

                if values and any(v > 0 for v in values):
                    pie_colors = [colors[i % len(colors)] for i in range(len(names_list))]
                    ax.pie(values, labels=names_list, colors=pie_colors, autopct='%1.1f%%')

            ax.set_title(f'{metric.upper()} {plot_type}')
            ax.grid(True, alpha=0.3)

        # Cleanup figures on the row
        for idx in range(len(var_unique_columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if savefig_path:
            save_dir = os.path.dirname(os.path.join(os_path, savefig_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            var_dpi = options.get('dpi', 300)
            plt.savefig(os.path.join(save_dir, os.path.basename(savefig_path)), dpi=var_dpi, bbox_inches='tight')

        if (show_figure):
            plt.show()

        return plt
