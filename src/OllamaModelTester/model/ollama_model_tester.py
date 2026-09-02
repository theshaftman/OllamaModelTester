import importlib
import time
import os
import re
from typing import Optional, List, Dict, Any
import multiprocessing
from datetime import datetime

from .model_visualizer import ModelVisualizer
from OllamaModelTester.services.import_module import ImportModule as im
from OllamaModelTester.services.package_install import PackageInstall as packi
from OllamaModelTester.services.execute_cmd import ExecuteCommand as ec


class OllamaModelTester:
    """
    OllamaModelTester
    A comprehensive Python framework for testing, evaluating, and visualizing Ollama language models with built-in hallucination detection, performance metrics, and Google BigQuery integration.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 11434,
        models: List[str] = [],
        install_packages: bool = True,
        show_figure: bool = True,
        is_libraries_exec_requested: bool = True,
        install_requirements_txt: bool = False,
        cmd_timeout: int = 120,
        os_path: str = os.path.dirname(os.path.abspath(__file__))
    ):
        """
        OllamaModelTester initialization

        Args:
            host (str): Ollama server host
            port (int): Ollama server port
            models (List[str]): List of selected models
            install_packages (bool): Install packages for OS
            show_figure (bool): Show figure when method ".visualize_results" is executed
            is_libraries_exec_requested (bool): Install internal packages
            install_requirements_txt (bool): Install packages from "requirements.txt" file
            cmd_timeout (int): Standard command timeout
            os_path (str): Source path to use folders 'credentials', 'documents' and charts
        """
        self.host = host
        self.port = port
        self.models = models
        self.process = None
        self.model_results = []
        self.validation_results = []
        self.dft_sleep_sec = 10
        self.nli_model = None
        self.key_path = None
        self.credentials = None
        self.install_packages = install_packages
        self.show_figure = show_figure
        self.is_libraries_exec_requested = is_libraries_exec_requested
        self.install_requirements_txt = install_requirements_txt
        self.cmd_timeout = cmd_timeout
        self.os_path = os_path
        self.imported_modules = None

    def get_data(self, key) -> Any:
        return getattr(self, key)

    def set_data(self, key, val) -> None:
        setattr(self, key, val)

    def pull_model(self, model_name: str = None) -> None:
        """
        Public method: Pull a selected method

        Args:
            model_name (str): Model name to pull

        Returns:
            None
        """
        result = ec.run_command(command=f'ollama pull {model_name}', check=True, timeout=self.cmd_timeout)
        if (result.returncode == 0):
            print(f'"{model_name}" model is pulled successfully!')
        else:
            print(f'"{model_name}" model pull threw an error')

    def pull_models(self, models: List[str] = []) -> None:
        """
        Public method: Pull a selected list of methods

        Args:
            model_name (List[str]): List of model names to pull

        Returns:
            None
        """
        models = models if models else self.models
        for model_name in models:
            result = ec.run_command(f'ollama pull {model_name}', check=True, timeout=self.cmd_timeout)
            if (result.returncode == 0):
                print(f'"{model_name}" model is pulled successfully!')
            else:
                print(f'"{model_name}" model pull threw an error')

    def compare_models(
        self,
        prompt_text: str = None,
        models: List[str] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Public method: Compare selected or internal added models

        Args:
            prompt_text (str): Prompt original text
            models (List[str]): Selected models to compare
            **options: Kwargs additional options

        Returns:
            List[Dict[str, Any]]: Compared models with scores
        """
        var_models = models if models else self.models
        var_options = {
            'temperature': options.get('temperature', 0.1),
            'num_ctx': options.get('num_ctx', 512)
        }
        for model_name in var_models:
            try:
                print(f'Testing model "{model_name}"')

                first_token_received = False
                tokens_received = 0
                token_times = []
                response_text = ''

                ttft_start = time.time()
                start=time.time()

                response_generator = self.ollama.generate(
                    model=model_name,
                    prompt=prompt_text,
                    options=var_options,

                    stream=True
                )

                for chunk in response_generator:
                    current_time = time.time()
                    tokens_received += 1

                    if (not first_token_received):
                        ttft = current_time - ttft_start
                        first_token_received = True
                        token_times.append(ttft)
                    else:
                        token_times.append(current_time - start)

                    response_text += chunk['response']

                elapsed=time.time() - start

                tokens_per_second = tokens_received / elapsed if elapsed > 0 else 0
                sorted_times = sorted(token_times)
                p50 = sorted_times[len(sorted_times) // 2] if len(sorted_times) > 0 else 0
                p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0

                model_comparison: Dict[str, Any] = {
                    'model': model_name,
                    'response': response_text,
                    'tokens': len(response_text.split()),
                    'elapsed_time': round(elapsed, 2),
                    'ttft': round(ttft, 3),
                    'tokens_per_second': round(tokens_per_second, 3),
                    'p50_latency': round(p50, 3),
                    'p95_latency': round(p95, 3)
                }

                # Extract hardware and model information
                try:
                    model_info = self.__get_model_info(model_name)
                except:
                    model_info = {
                        'hardware': 'unknown',
                        'quantization': 'unknown',
                        'param_size': 'unknown',
                        'files_size': 'unknown',
                        'context_length': 'unknown'
                    }
                model_comparison.update(model_info)

                calculate_scores = self.__calculate_scores(prompt_text, response_text)
                model_comparison.update(calculate_scores)

                self.model_results.append(model_comparison)
                print(f'Successfully completed testing model "{model_name}"')
            except Exception as e:
                self.model_results.append({
                    'model': model_name,
                    'response': f'"{model_name}" error: {str(e)}',
                    'tokens': 0,
                    'elapsed_time': 0,
                    'ttft': 0,
                    'tokens_per_second': 0,
                    'p50_latency': 0,
                    'p95_latency': 0,
                    'hardware': 'unknown',
                    'quantization': 'unknown',
                    'param_size': 'unknown',
                    'files_size': 'unknown',
                    'context_length': 'unknown'
                })
                print(f'Testing model "{model_name}" threw an error')

        return self.model_results

    def validate_evaluator(
        self,
        prompt_text: str = None,
        generated_text: str = None,
        human_label: str = None
    ) -> List[Dict[str, Any]]:
        """
        Meeting 2 (Extended)
        Public method: Validate human generated text and label

        Args:
            prompt_text (str): Original text
            generated_text (str): Human generated text
            human_label (str): Text defining that generated_text is faithful or hallucinated
        Returns:
            Dict[str, Any]
        """
        result: Dict[str, Any] = {}
        try:
            calculate_scores = self.__calculate_scores(prompt_text, generated_text)
            predicted = 'hallucinated' if (calculate_scores['is_hallucinated'] == 1.0) else 'faithful'

            human_labels = [human_label]
            predictions = [predicted]

            overall_accuracy = self.accuracy_score(human_labels, predictions)
            overall_f1_score = self.f1_score(human_labels, predictions, pos_label='hallucinated', average='binary')

            result.update({
                'human_label': human_label,
                'predicted': predicted,
                'overall_accuracy': overall_accuracy,
                'overall_f1_score': overall_f1_score
            })

            # Extract hardware and model information
            try:
                model_info = self.__get_model_info()
            except:
                model_info = {
                    'hardware': 'unknown',
                    'quantization': 'unknown',
                    'param_size': 'unknown',
                    'files_size': 'unknown',
                    'context_length': 'unknown'
                }
            result.update(model_info)
            result.update(calculate_scores)
            self.validation_results.append(result)
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return self.validation_results

    def export_results_to_csv(self) -> None:
        """
        Meeting 3
        Public method to export "model_results" and "validation_results" to CSV in a folder "documents"
        """
        folder_path = os.path.join(self.os_path, 'documents')
        if (not os.path.exists(folder_path)):
            os.makedirs(folder_path)
        files_to_export = ['model_results', 'validation_results']
        for file_name in files_to_export:
            filepath = os.path.join(folder_path, f'{file_name}.csv')
            df = self.pd.DataFrame(self.get_data(file_name))
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now()
            df.to_csv(filepath, index=False)
            print(f'Exported file to "{filepath}"')

    def import_results_from_csv(self) -> None:
        """
        Meeting 3
        Public method to import "model_results" and "validation_results" from CSV in a folder "documents"
        """
        folder_path = os.path.join(self.os_path, 'documents')
        files_to_import = ['model_results', 'validation_results']
        for file_name in files_to_import:
            filepath = os.path.join(folder_path, f'{file_name}.csv')
            if (os.path.exists(filepath)):
                df = self.pd.read_csv(filepath)
                model_data = df.to_dict('records')
                self.set_data(file_name, model_data)
                print(f'"{filepath}" imported successfully')
            else:
                print(f'"{filepath}" NOT imported')

    def export_results_to_gqb(
        self,
        project_id: str = None,
        dataset_id: str = None,
        if_exists: str = 'append'
    ) -> bool:
        """
        Meeting 3
        Public method to export "model_results" and "validation_results" to Google BigQuery Dataset

        Args:
            project_id (str): Set project ID in Google Cloud
            dataset_id (str): Set dataset ID in Google BigQuery Dataset
            if_exists (str): Select if you want to "append" or "replace" the new data in the export

        Returns:
            bool: Validation if process is completed successful
        """
        is_exported = False
        if (project_id and dataset_id):
            tables_to_export = ['model_results', 'validation_results']
            try:
                for table_name in tables_to_export:
                    df = self.pd.DataFrame(self.get_data(table_name))
                    if ('timestamp' not in df.columns):
                        df['timestamp'] = datetime.now()
                    self.pandas_gbq.to_gbq(
                        df,
                        f'{project_id}.{dataset_id}.{table_name}',
                        project_id = project_id,
                        if_exists = if_exists,
                        credentials = self.credentials,
                        progress_bar=True
                    )
                    is_exported = True
                    print(f'Table "{table_name}" export is completed')
            except Exception as e:
                is_exported = False
                print(f'Exception appeared (export_results_to_gqb): {str(e)}')

        return is_exported

    def import_results_from_gbq(self, project_id: str = None, dataset_id: str = None, limits: int = 1000) -> bool:
        """
        Meeting 3
        Public method to import "model_results" and "validation_results" from Google BigQuery Dataset

        Args:
            project_id (str): Set project ID in Google Cloud
            dataset_id (str): Set dataset ID in Google BigQuery Dataset
            limits (int): Select the maximum limits of results that will be selected from the Google BigQuery Dataset tables

        Returns:
            bool: Validation if process is completed successful
        """
        is_imported = False
        if (project_id and dataset_id):
            tables_to_export = ['model_results', 'validation_results']
            try:
                for table_name in tables_to_export:
                    query = f"""
                    SELECT *
                      FROM `{project_id}.{dataset_id}.{table_name}`
                     LIMIT {limits}
                    """
                    df = self.pandas_gbq.read_gbq(
                        query,
                        project_id = project_id,
                        credentials = self.credentials
                    )

                    model_data = df.to_dict('records')
                    self.set_data(table_name, model_data)
                    is_imported = True
                    print(f'Table "{table_name}" import is successful')
            except self.NotFound as e:
                is_imported = False
                print(f'Table "{table_name}" is not found')
            except Exception as e:
                is_imported = False
                print(f'Exception appeared (import_results_from_gbq): {str(e)}')

        return is_imported

    def print_results(self, i_results: List[Dict[str, Any]] = []) -> None:
        """
        Public method: Print a list of models

        Args:
            i_results (List[Dict[str, Any]]): List of models
        
        Returns:
            None
        """
        var_results = i_results if i_results else self.model_results
        for row in var_results:
            print('')
            [print(f'{k}: {v}') for k, v in row.items()]
            print('')

    def visualize_results(
        self,
        plot_type: str = None,
        metrics: List[Dict[str, Any]] = None,
        colors: Optional[List[Dict[str, Any]]] = None,
        savefig_path: str = None,
        max_cols_per_row: int = 3,
        **options
    ) -> Any:
        """
        Meeting 4
        Public method to visualize internal resutls from self.model_results, self.validation_results and query SQL statements to Google BigQuery Dataset.

        Args:
            plot_type (str): Select plot type ('bar', 'plot', 'scatter', 'pie')
            metrics (List[Dict[str, Any]]): Metrics to extract and compare the data
            colors: (Optional[List[Dict[str, Any]]]): List of predefined colors to color the AI model or Human-labelled chart with the data
            savefig_path (str): Path to save the figures
            max_cols_per_row (int): Define maximum number of charts per row
            **options: Keyword arguments to accept any number of named (keyword) inputs, packed into a standard dictionary

        Returns:
            plt: Matplotlib Pyplot of the chart
        """
        all_dfs = []
        VALID_DATA_SOURCES = ['model_results', 'validation_results']
        for metric in metrics:
            var_data = metric['data']
            var_columns = metric['columns']
            var_self_data = self.get_data(var_data) if var_data in VALID_DATA_SOURCES else []
            columns_to_select = ['model', *var_columns] if var_data == 'model_results' else var_columns
            var_df = self.pd.DataFrame(var_self_data)[columns_to_select] if var_data in VALID_DATA_SOURCES else self.pd.DataFrame()

            if (var_data == 'model_results'):
                var_df['row_num'] = var_df.groupby('model').cumcount()
                pivoted = var_df.pivot(index='row_num', columns='model', values=var_columns)
                pivoted.columns = [f'{model}_{col}' for col, model in pivoted.columns]
                var_df = pivoted.fillna(0).reset_index(drop=True)
                all_dfs.append(var_df)
            elif (var_data == 'validation_results'):
                var_df.columns = [f'human_{col}' for col in var_df.columns]
                all_dfs.append(var_df)
            elif (var_data == 'sql'):
                try:
                    var_project_id = metric['project_id']
                    query = metric['sql']
                    var_df = self.pandas_gbq.read_gbq(
                        query,
                        project_id = var_project_id,
                        credentials = self.credentials
                    )
                    var_df['row_num'] = var_df.groupby('model').cumcount()
                    pivoted = var_df.pivot(index='row_num', columns='model', values=var_columns)
                    pivoted.columns = [f'{model}_{col}' for col, model in pivoted.columns]
                    var_df = pivoted.fillna(0).reset_index(drop=True)

                    # print(var_df)
                    all_dfs.append(var_df)
                except self.NotFound as e:
                    print(f'Exception NotFound: {str(e)}')
                except Exception as e:
                    print(f'Exception: {str(e)}')

        final_df = self.pd.concat(all_dfs, axis=1)

        fig = ModelVisualizer.visualize_chart(
            plot_type = plot_type,
            metrics = metrics,
            colors = colors,
            data = final_df,
            savefig_path = savefig_path,
            max_cols_per_row = max_cols_per_row,
            show_figure = self.show_figure,
            imported_modules = self.imported_modules,
            os_path = self.os_path,
            options = options
        )
        return fig

    def __initialization(self) -> None:
        os.environ['OLLAMA_HOST'] = f'{self.host}:{self.port}'

        try:
            self.nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            self.nltk.download('punkt_tab')
            self.nltk.download('punkt')
            self.nltk.download('averaged_perceptron_tagger_eng')

        self.nli_model = self.pipeline('text-classification',
            model='cross-encoder/nli-deberta-v3-base',
            device=0
        )

        folder_path = 'credentials'
        if (os.path.exists(folder_path)):
            files = os.listdir(folder_path)
            json_files = [f for f in files if f.endswith('.json')]
            self.key_path = f'{folder_path}/{json_files[0]}'
        try:
            self.credentials = self.service_account.Credentials.from_service_account_file(
                self.key_path,
                scopes = ['https://www.googleapis.com/auth/cloud-platform']
            )
        except Exception as e:
            print(f'Credentials to Google Oauth2 are not set: {str(e)}')
        print('Class OllamaModelTester is initialized')

    def __start_server(self) -> 'OllamaModelTester':
        """
        Privatete method: Actions to start the server
        """
        self.process = ec.run_popen(command='ollama serve', text=True)
        time.sleep(self.dft_sleep_sec)
        result = ec.run_command(
            command=f'curl -s http://{self.host}:{self.port}/api/tags',
            check=True,
            timeout=self.cmd_timeout
        )
        if (result.returncode == 0):
            print('Ollama server is started')
        else:
            print('Ollama server is NOT started')

    def __stop_server(self) -> None:
        """
        Privatete method: Actions to stop the server
        """
        self.process.terminate()
        self.process.wait()
        print('Ollama server is terminated')


    def __clean_text(self, text: str = None) -> str:
        """
        Meeting 2
        Private method: Preprocess text for tokenization

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def __calculate_scores(self, reference_text: str = None, generated_text: str = None) -> Dict[str, Any]:
        """
        Meeting 2
        Private method: Calculate multiple BLUE score variants

        Args:
            reference_text (str): Text passed from main
            generated_text (str): Text generated from AI agent

        Returns:
            Dict[str, float]
        """
        dict_scores = {
            'total': 0.0,
            'factual': 0.0,
            'contradictions': 0.0,
            'neutral': 0.0,
            'confidence': 0.0,
            'is_hallucinated': 0.0,
            'hallucination_severity': 0.0,
            'faithfulness_score': 0.0,
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'bleu_avg': 0.0,
            'bleu_smoothing_method0': 0.0,
            'bleu_smoothing_method1': 0.0,
            'bleu_smoothing_method2': 0.0,
            'bleu_smoothing_method3': 0.0,
            'bleu_smoothing_method4': 0.0,
            'bleu_smoothing_method5': 0.0,
            'bleu_smoothing_method6': 0.0,
            'bleu_smoothing_method7': 0.0
        }
        try:
            if (not generated_text or len(generated_text.strip()) == 0):
                return dict_scores

            cleaned_ref = self.__clean_text(reference_text)
            cleaned_gen = self.__clean_text(generated_text)

            reference_tokens = self.word_tokenize(cleaned_ref)
            generated_tokens = self.word_tokenize(cleaned_gen)

            if (not reference_tokens or not generated_tokens):
                return dict_scores

            sentences = self.nltk.sent_tokenize(generated_text)
            if not sentences:
                return dict_scores

            result_scores = []
            for sentence in sentences:
                pred = self.nli_model(f'{reference_text} </s> {sentence}')
                label = (pred[0]['label']).upper()
                score = pred[0]['score']

                result_scores.append({
                    'sentence': sentence,
                    'label': label,
                    'confidence': score
                })

            dict_scores['total'] = len(result_scores)
            dict_scores['factual'] = sum(1 for r in result_scores if r['label'] == 'ENTAILMENT') / len(result_scores)
            dict_scores['contradictions'] = sum(1 for r in result_scores if r['label'] == 'CONTRADICTION') / len(result_scores)
            dict_scores['neutral'] = sum(1 for r in result_scores if r['label'] == 'NEUTRAL') / len(result_scores)
            dict_scores['confidence'] = sum(r['confidence'] for r in result_scores) / len(result_scores)

            dict_scores['is_hallucinated'] = 1.0 if (dict_scores['contradictions'] > 0 or dict_scores['neutral'] > 0.3) else 0
            dict_scores['hallucination_severity'] = (dict_scores['contradictions'] * 1.0) + (dict_scores['neutral'] * 0.5)
            dict_scores['faithfulness_score'] = dict_scores['factual'] - (dict_scores['contradictions'] * 0.5)

            # Calculating BLEU scores
            smoothing = self.SmoothingFunction()
            var_weights = (0.25, 0.25, 0.25, 0.25)

            dict_scores['bleu_1'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing.method1)
            dict_scores['bleu_2'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smoothing.method1)
            dict_scores['bleu_3'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0.0), smoothing_function=smoothing.method1)
            dict_scores['bleu_4'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=(var_weights), smoothing_function=smoothing.method1)
            dict_scores['bleu_avg'] = sum([
                dict_scores['bleu_1'],
                dict_scores['bleu_2'],
                dict_scores['bleu_3'],
                dict_scores['bleu_4']
            ]) / 4.0

            # Calculating smoothing methods
            dict_scores['bleu_smoothing_method0'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method0)
            dict_scores['bleu_smoothing_method1'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method1)
            dict_scores['bleu_smoothing_method2'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method2)
            dict_scores['bleu_smoothing_method3'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method3)
            dict_scores['bleu_smoothing_method4'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method4)
            dict_scores['bleu_smoothing_method5'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method5)
            dict_scores['bleu_smoothing_method6'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method6)
            dict_scores['bleu_smoothing_method7'] = self.sentence_bleu([reference_tokens], generated_tokens, weights=var_weights, smoothing_function=smoothing.method7)

        except Exception as e:
            print(f'Error appeared: {str(e)}')
        finally:
            return dict_scores

    def __get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Meeting 2 (Extended)
        Private method to extract hardware and model information

        Args:
            model_name (str): Model name

        Returns:
            Dict[str, Any]
        """
        info = {
            'hardware': 'unknown',
            'quantization': 'unknown',
            'param_size': 'unknown',
            'files_size': 'unknown',
            'context_length': 'unknown'
        }
        try:
            if (self.torch.cuda.is_available()):
                gpu_name = self.torch.cuda.get_device_name(0)
                info['hardware'] = f'GPU: {gpu_name}'
            else:
                cpu_count = multiprocessing.cpu_count()
                info['hardware'] = f'CPU: {cpu_count} cores'

            if (model_name):
                response = self.requests.get(f'http://{self.host}:{self.port}/api/tags')
                if (response.status_code == 200):
                    data = response.json()
                    models = data.get('models', [])
                    model = [model for model in models if model_name in model['name']]
                    if model:
                        model = model[0]

                        info['quantization'] = model['details']['quantization_level']
                        info['param_size'] = model['details']['parameter_size']
                        info['files_size'] = f'{round(model['size'] / (1024 ** 3), 3)} GB'
                        info['context_length'] = model['details']['context_length']
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return info

    def __package_installation(self) -> bool:
        """
        Private method: Installation of packages
        """
        return packi.package_installation(
            install_packages=self.install_packages,
            install_requirements_txt=self.install_requirements_txt,
            timeout=self.cmd_timeout,
            os_path=self.os_path
        )

    def _library_imports(self) -> None:
        """
        Private method: Import of libraries
        """
        imported_modules = im.import_all(
            is_libraries_exec_requested=self.is_libraries_exec_requested,
            timeout=self.cmd_timeout
        )
        self.imported_modules = imported_modules

        self.ollama = imported_modules['ollama']
        self.nest_asyncio = imported_modules['nest_asyncio']
        self.requests = imported_modules['requests']
        self.transformers = imported_modules['transformers']
        self.pipeline = getattr(self.transformers, 'pipeline')
        self.nltk = self.imported_modules['nltk']
        self.bleu_score = importlib.import_module('nltk.translate.bleu_score')
        self.sentence_bleu = getattr(self.bleu_score, 'sentence_bleu')
        self.SmoothingFunction = getattr(self.bleu_score, 'SmoothingFunction')
        self.tokenize = importlib.import_module('nltk.tokenize')
        self.word_tokenize = getattr(self.tokenize, 'word_tokenize')
        self.torch = self.imported_modules['torch']
        self.sklearn = self.imported_modules['sklearn']
        self.metrics = importlib.import_module('sklearn.metrics')
        self.accuracy_score = getattr(self.metrics, 'accuracy_score')
        self.f1_score = getattr(self.metrics, 'f1_score')
        self.pandas = self.imported_modules['pandas']
        self.pd = self.pandas
        self.google = self.imported_modules['google']
        self.service_account = importlib.import_module('google.oauth2.service_account')
        self.exceptions = importlib.import_module('google.api_core.exceptions')
        self.NotFound = self.exceptions.NotFound
        self.pandas_gbq = self.imported_modules['pandas_gbq']
        self.matplotlib = self.imported_modules['matplotlib']
        self.plt = importlib.import_module('matplotlib.pyplot')
        self.numpy = self.imported_modules['numpy']
        self.np = self.numpy
    
    def __enter__(self) -> 'OllamaModelTester':
        self.__package_installation()
        self._library_imports()
        self.__initialization()
        self.__start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__stop_server()
