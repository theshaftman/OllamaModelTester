
import importlib
from typing import Dict, Any

from .execute_cmd import ExecuteCommand as ec


class ImportModule():

    REQUIRED_VERSIONS = {
        'numpy': { 'import': 'numpy', 'version': '2.1.3' },
        'torch': { 'import': 'torch', 'version': '2.11.0' },
        'transformers': { 'import': 'transformers', 'version': '5.15.0' },
        'ollama': { 'import': 'ollama', 'version': '0.6.2' },
        'nltk': { 'import': 'nltk', 'version': '3.9.1' },
        'scikit-learn': { 'import': 'sklearn', 'version': '1.6.1' },
        'pandas': { 'import': 'pandas', 'version': '2.2.3' },
        'matplotlib': { 'import': 'matplotlib', 'version': '3.10.0' },
        'google-auth': { 'import': 'google', 'version': '2.49.0' },
        'google-cloud-bigquery': { 'import': 'google.cloud', 'version': '3.43.0' },
        'pandas-gbq': { 'import': 'pandas_gbq', 'version': '0.30.0' },
        'google-api-core': { 'import': 'google.api_core', 'version': '2.30.3' },
        'requests': { 'import': 'requests', 'version': '2.32.4' },
        'nest-asyncio': { 'import': 'nest_asyncio', 'version': '1.6.0' },
        'typing': { 'import': 'typing', 'version': '3.7.4.3' },
        'pyarrow': {'import': 'pyarrow', 'version': '25.0.1'}
    }

    @staticmethod
    def import_all(is_libraries_exec_requested: bool = True, timeout: int = 120) -> Dict[str, Any]:
        loaded_modules = {}
        try:
            for package_name, package_data in ImportModule.REQUIRED_VERSIONS.items():
                required_import = package_data['import']
                required_version = package_data['version']

                if (is_libraries_exec_requested):
                    timeout = 600 if package_name == 'torch' and timeout < 600 else timeout
                    var_command = f'pip install {package_name}=={required_version}'
                    try:
                        ec.run_command(command=var_command, check=True, timeout=timeout)
                    except Exception as e:
                        print(f'Execption thrown: {str(e)}')

                module = importlib.import_module(required_import)
                loaded_modules[required_import] = module
        except Exception as e:
            print(f'Execption thrown: {str(e)}')
        finally:
            return loaded_modules
