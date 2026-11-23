import torch


def get_device(device_type):
    """device_type("cuda" 또는 "cpu")에 맞는 장치를 돌려줍니다.

    학습은 config.TRAIN_DEVICE_TYPE, 추론은 config.INFER_DEVICE_TYPE을 넘깁니다.
    NVIDIA CUDA와 AMD ROCm 모두 "cuda"로 씁니다. ROCm용 PyTorch에서도 torch.cuda.is_available()이 True라 같은 방식으로 동작합니다.
    """
    if device_type == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "\n[경고] DEVICE_TYPE = 'cuda' 이지만 GPU를 찾을 수 없어 CPU로 실행합니다.\n"
            "원인 및 해결 방법:\n"
            "  1. ROCm PyTorch 미설치\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/rocm6.2\n"
            "  2. CUDA PyTorch 미설치\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  3. WSL2에서 GPU 패스스루 미설정\n"
            "     → /proc/driver/nvidia 또는 /dev/dri 디바이스 존재 여부 확인\n"
            "  설치 상태 확인: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n"
        )
        return torch.device("cpu")
    return torch.device("cpu")


def is_rocm() -> bool:
    """지금 PyTorch가 ROCm(HIP)용인지 돌려줍니다. ROCm용이면 torch.version.hip에 버전 문자열이 있습니다."""
    return torch.version.hip is not None


def predict_batch_size(device) -> int:
    """서비스 추론 배치 크기입니다. GPU면 config.PREDICT_INFER_BATCH_SIZE_GPU, CPU면 config.PREDICT_INFER_BATCH_SIZE_CPU를 씁니다."""
    import config
    size = config.PREDICT_INFER_BATCH_SIZE_GPU if is_cuda_like(device) else config.PREDICT_INFER_BATCH_SIZE_CPU
    return max(1, int(size))


def is_cuda_like(device) -> bool:
    """AMP, cuDNN, pin_memory 같은 CUDA 계열 기능을 쓸 수 있는 장치인지 돌려줍니다.
    NVIDIA CUDA와 ROCm 모두 장치 이름이 "cuda"이므로 True이고, CPU는 False입니다.
    """
    return str(device).startswith("cuda")


def is_channels_last_3d_supported(device) -> bool:
    """channels_last_3d 메모리 형식을 안전하게 쓸 수 있는지 돌려줍니다.
    ROCm은 channels_last_3d를 완전히 지원하지 않아 오류가 날 수 있으므로 NVIDIA CUDA에서만 씁니다.
    """
    if not is_cuda_like(device):
        return False
    if is_rocm():
        return False
    return True
