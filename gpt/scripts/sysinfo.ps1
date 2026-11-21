# Диагностика машины под обучение minigpt.
#
# Запуск из cmd (из папки gpt):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sysinfo.ps1
#
# Или одной строкой из любого места (см. README): через -EncodedCommand.
# Скрипт ничего не меняет и никуда не пишет — только читает данные системы.

$ErrorActionPreference = 'SilentlyContinue'
$OutputEncoding = [System.Text.Encoding]::UTF8

"=== SYSTEM ==="
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$os.Caption + ' build ' + $os.BuildNumber
'RAM total GB: ' + [math]::Round($cs.TotalPhysicalMemory / 1GB, 1) + '   free GB: ' + [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List

"=== GPU (nvidia-smi) ==="
nvidia-smi
if (-not $?) { 'nvidia-smi не найден или не запускается' }

"=== GPU (WMI) ==="
Get-CimInstance Win32_VideoController |
    Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor | Format-List

"=== DISK ==="
Get-PSDrive -PSProvider FileSystem |
    Select-Object Name,
        @{n = 'FreeGB'; e = { [math]::Round($_.Free / 1GB, 1) } },
        @{n = 'UsedGB'; e = { [math]::Round($_.Used / 1GB, 1) } } | Format-Table

"=== PYTHON ==="
python --version 2>&1
where.exe python 2>&1
git --version 2>&1

"=== TORCH ==="
python -c 'import torch,sys;print("python",sys.version_info[:3]);print("torch",torch.__version__);print("torch_cuda",torch.version.cuda);print("cuda_available",torch.cuda.is_available());print("device_count",torch.cuda.device_count());print("device_name",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a");print("capability",torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a");print("threads",torch.get_num_threads())' 2>&1
if (-not $?) { 'Python или torch не установлены (это нормально, поставим после диагностики)' }

"=== NETWORK ==="
foreach ($u in 'https://huggingface.co', 'https://pypi.org', 'https://github.com') {
    try {
        $r = Invoke-WebRequest -Uri $u -Method Head -UseBasicParsing -TimeoutSec 10
        $u + ' -> HTTP ' + $r.StatusCode
    } catch {
        $u + ' -> ОШИБКА: ' + $_.Exception.Message
    }
}

"=== END ==="
