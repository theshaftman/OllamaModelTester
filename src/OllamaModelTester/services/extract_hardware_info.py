import multiprocessing
from typing import Any, Dict
import platform

from .execute_cmd import ExecuteCommand as ec
from OllamaModelTester.services.hardware_gpu import HardwareGPU as hgpu


class ExtractHardwareInfo():
    @staticmethod
    def get_hardware_info(
        host: int = None,
        port: int = None,
        model_name: str = None,
        imported_modules: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Meeting 2 (Extended)
        Private method to extract hardware and model information

        Args:
            model_name (str): Model name

        Returns:
            Dict[str, Any]
        """
        info = {
            'os': platform.system(),
            'os_release': platform.release(),
            'os_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'cpu_model': 'unknown',
            'cpu_cores': 'unknown',
            'ram_gb': None,
            'version': 'unknown',
            'hardware': 'unknown',
            'quantization': 'unknown',
            'param_size': 'unknown',
            'files_size': 'unknown',
            'context_length': 'unknown',
            'family': 'unknown',
            'quantization_level': 'unknown'
        }
        try:
            gpu_name = hgpu.get_gpu_names()
            requests = imported_modules['requests']
            if (gpu_name):
                info['hardware'] = f'GPU: {gpu_name}'
            else:
                cpu_count = multiprocessing.cpu_count()
                info['hardware'] = f'CPU: {cpu_count} cores'

            if (model_name):
                version = requests.get(f'http://{host}:{port}/api/version').json()['version']
                info['version'] = version
                response = requests.get(f'http://{host}:{port}/api/tags')
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

                        # Get details
                        details = model['details'] if model['details'] else {}
                        info['family'] = details.get('family')
                        info['quantization_level'] = details.get('quantization_level')

            # CPU info
            system = platform.system()
            if system == 'Linux':
                try:
                    with open('/proc/cpuinfo') as f:
                        for line in f:
                            if line.startswith('model name'):
                                info['cpu_model'] = line.split(":", 1)[1].strip()
                                break
                    r = ec.run_command('nproc', check=False)
                    if r.returncode == 0:
                        info['cpu_cores'] = int(r.stdout.strip())
                except Exception:
                    pass
                try:
                    with open('/proc/meminfo') as f:
                        for line in f:
                            if line.startswith('MemTotal'):
                                kb = int(line.split()[1])
                                info['ram_gb'] = round(kb / 1024 / 1024, 2)
                                break
                except Exception:
                    pass

            elif system == 'Windows':
                try:
                    r = ec.run_command(
                        'powershell -Command "(Get-CimInstance Win32_Processor).Name"',
                        check=False,
                    )
                    if r.returncode == 0:
                        info['cpu_model'] = r.stdout.strip().split('\n')[0]
                    r = ec.run_command(
                        'powershell -Command "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"',
                        check=False,
                    )
                    if r.returncode == 0:
                        info['cpu_cores'] = int(r.stdout.strip())
                    r = ec.run_command(
                        'powershell -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)"',
                        check=False,
                    )
                    if r.returncode == 0:
                        info['ram_gb'] = float(r.stdout.strip())
                except Exception:
                    pass

        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return info
