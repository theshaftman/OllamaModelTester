import shutil
import re
import platform
import glob
from typing import Any, Dict, List
from .execute_cmd import ExecuteCommand as ec


class HardwareGPU():

    @staticmethod
    def get_all_gpu_names() -> List[Any]:
        """
        Detect all GPUs including internal/integrated graphics.

        Returns:
            List[Any]: A list with hardware GPU names
        """
        gpus = []
        system = platform.system()
        
        # Linux detection
        if system == 'Linux':
            if shutil.which('lspci'):
                try:
                    result = ec.run_command(command='lspci -vnn', check=False)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'VGA' in line or 'Display' in line or '3D' in line:
                                # Extract GPU info
                                gpu_info = line.strip()
                                
                                # Detect vendor and name
                                vendor = None
                                name = None
                                
                                if 'NVIDIA' in gpu_info:
                                    vendor = 'NVIDIA'
                                elif 'AMD' in gpu_info or 'ATI' in gpu_info:
                                    vendor = 'AMD'
                                elif 'Intel' in gpu_info:
                                    vendor = 'Intel'
                                elif 'Red Hat' in gpu_info:
                                    vendor = 'Red Hat (Virtual)'
                                else:
                                    vendor = 'Unknown'
                                
                                # Extract name
                                parts = gpu_info.split(':')
                                if len(parts) >= 3:
                                    name = parts[2].strip()
                                    # Clean up name
                                    name = re.sub(r'\(.*?\)', '', name).strip()
                                else:
                                    name = gpu_info

                                if not name or name == gpu_info:
                                    match = re.search(r'\[(.*?)\]', gpu_info)
                                    if match:
                                        name = match.group(1)

                                if name and name not in [g[1] for g in gpus]:
                                    gpus.append((vendor, name))
                except Exception:
                    pass

            try:
                for path in glob.glob('/sys/class/drm/card*/device/vendor'):
                    with open(path, 'r') as f:
                        vendor_id = f.read().strip()
                        if vendor_id == '0x10de':
                            vendor = 'NVIDIA'
                        elif vendor_id == '0x1002':
                            vendor = 'AMD'
                        elif vendor_id == '0x8086':
                            vendor = 'Intel'
                        else:
                            vendor = 'Unknown'
                        
                        # Try to get device name
                        device_path = path.replace('vendor', 'device')
                        try:
                            with open(device_path, 'r') as df:
                                device_id = df.read().strip()
                                name = f"{vendor} GPU (device {device_id})"
                                if (vendor, name) not in gpus:
                                    gpus.append((vendor, name))
                        except:
                            pass
            except Exception:
                pass
        
        # Windows detection
        elif system == 'Windows':
            try:
                result = ec.run_command(
                    command='powershell -Command "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"',
                    check=False
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('Name') and not line.startswith('---'):
                            name = line
                            vendor = 'Unknown'
                            if 'NVIDIA' in name:
                                vendor = 'NVIDIA'
                            elif 'AMD' in name or 'Radeon' in name:
                                vendor = 'AMD'
                            elif 'Intel' in name:
                                vendor = 'Intel'
                            
                            if name and (vendor, name) not in gpus:
                                gpus.append((vendor, name))
            except Exception:
                pass
            
            # Fallback: PowerShell
            try:
                result = ec.run_command(
                    command='powershell -Command "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"',
                    check=False
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('Name') and not line.startswith('---'):
                            # Detect vendor
                            vendor = 'Unknown'
                            if 'NVIDIA' in line:
                                vendor = 'NVIDIA'
                            elif 'AMD' in line or 'Radeon' in line:
                                vendor = 'AMD'
                            elif 'Intel' in line:
                                vendor = 'Intel'
                            
                            if line and (vendor, line) not in gpus:
                                gpus.append((vendor, line))
            except Exception:
                pass

        if not gpus:
            if shutil.which('nvidia-smi'):
                try:
                    result = ec.run_command(command='nvidia-smi --query-gpu=name --format=csv,noheader', check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        for name in result.stdout.strip().split('\n'):
                            if name:
                                gpus.append(('NVIDIA', name))
                except Exception:
                    pass
        
        return gpus

    @staticmethod
    def get_gpu_names() -> str:
        """
        Detect all GPUs including internal/integrated graphics.

        Returns:
            str: Hardware GPU names
        """
        gpus = [gpu[1] for gpu in HardwareGPU.get_all_gpu_names()]
        return ', '.join(gpus)