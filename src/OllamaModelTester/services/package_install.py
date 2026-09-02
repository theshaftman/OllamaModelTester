import os
import platform
from .execute_cmd import ExecuteCommand as ec


class PackageInstall():

    @staticmethod
    def _windows_install(
        install_requirements_txt: bool = False,
        timeout: int = 120,
        os_path: str = os.path.dirname(os.path.abspath(__file__))
    ) -> bool:
        is_installed = False
        try:
            if (install_requirements_txt):
                rcommands = [
                    f'pip install --no-cache-dir -r {os.path.join(os_path, 'requirements.txt')}'
                ]
                for cmd in rcommands:
                    ec.run_command(command=cmd, check=True, timeout=timeout)

            pcommands = [
                'winget install Ollama.Ollama'
            ]
            for cmd in pcommands:
                ec.run_popen(command=cmd, text=True)
            is_installed = True
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return is_installed

    @staticmethod
    def _linux_install(
        install_requirements_txt: bool = False,
        timeout: int = 120,
        os_path: str = os.path.dirname(os.path.abspath(__file__))
    ) -> bool:
        print('Installing packages on Linux')
        is_installed = False

        # Confirm sudo privileges
        if (os.geteuid() != 0):
            commands = [
                'sudo apt update',
                'sudo apt install -y pciutils',
                'sudo apt install -y zstd',
                'curl -fsSL https://ollama.com/install.sh | sudo sh'                
            ]

            if (install_requirements_txt):
                commands.append(f'pip install --no-cache-dir -r {os.path.join(os_path, 'requirements.txt')}')

            for cmd in commands:
                try:
                    ec.run_command(command=cmd, check=True, timeout=timeout)
                except Exception as e:
                    print(f'Exception thrown: {str(e)}')
            is_installed = True

        return is_installed

    @staticmethod
    def package_installation(
        install_packages: bool = True,
        install_requirements_txt: bool = False,
        timeout: int = 120,
        os_path: str = os.path.dirname(os.path.abspath(__file__))
    ) -> bool:
        is_installed = False
        if (install_packages):
            print('Installing packages')
            osystem = platform.system()
            print(f'Detected OS: {osystem}')

            if (osystem == 'Windows'):
                is_installed = PackageInstall._windows_install(
                    install_requirements_txt=install_requirements_txt,
                    timeout=timeout,
                    os_path=os_path
                )
            elif (osystem == 'Linux'):
                is_installed = PackageInstall._linux_install(
                    install_requirements_txt=install_requirements_txt,
                    timeout=timeout,
                    os_path=os_path
                )
            else:
                print(f'Unsupported OS: {osystem}')

        return is_installed
