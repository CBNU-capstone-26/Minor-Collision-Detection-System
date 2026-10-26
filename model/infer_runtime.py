"""추론 실행기 — 서버 컴퓨터에서 쓸 수 있는 가장 빠른 방식으로 3D-CNN 을 돌린다.

서비스와 평가는 모델을 직접 부르지 않고 get_runner(model) 이 돌려주는 실행기로 추론한다.
실행기는 logits 를 돌려주고, runner.with_cam(clips) 는 logits 와 칸별 클래스 지도(CAM)를 함께
돌려준다. 변환 모델(ONNX·OpenVINO)이 두 출력을 한 번의 추론에서 함께 내므로 CAM 을 켜도 빠른 방식을
그대로 쓴다(예전엔 logits 만 내보내 CAM 을 켜면 전체 탐색이 PyTorch 로 돌아 CPU 에서 약 4배 느렸다).

방식 (config.INFER_BACKEND):
  auto        : 쓸 수 있는 방식을 처음 한 번 자가 검사하고, 통과한 것 중 가장 빠른 것(기본)
  torch       : PyTorch 그대로(예전 방식). 모델이 GPU 에 있으면 GPU.
  openvino    : OpenVINO (CPU) — Intel CPU 에서 빠르다
  onnxruntime : ONNX Runtime (CPU)
모델이 GPU(CUDA)에 있으면 auto 는 PyTorch GPU 를 쓴다(CPU 방식들보다 빠르다). FP16 은
config.INFER_FP16 을 켰을 때만, Tensor Core 가 있는 NVIDIA GPU 에서 자가 검사를 거쳐 쓴다.

자가 검사: 같은 입력으로 PyTorch(FP32) 와 P(A)·logits·CAM 을 비교한다. 설치 안 됨·변환 실패·
검사 불통과면 그 방식은 버리고 다음 방식(최후엔 PyTorch)으로 넘어간다. 그래서 어느 OS·
하드웨어에서든 기능은 동작하고, 결과는 PyTorch 와 같다(정밀도는 FP32 그대로).

검사 결과와 변환 파일(ONNX)은 config.INFER_CACHE_DIR 에 저장한다. 가중치·패키지 버전·
CPU/GPU·OS 가 같으면 다음부터는 속도를 다시 재지 않고, 서버(워커)를 띄울 때마다 창 1개로
결과만 PyTorch 와 빠르게 비교한다(수 초). 처음 한 번의 전체 검사는 수십 초 걸릴 수 있다.
설치 직후 미리 해 두려면:  python -m model.main --mode check-backend
"""
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import predict_batch_size

# 검사·변환 방식이 바뀌면 올린다 → 저장된 결과·변환 파일을 새로 만든다
# (2: 변환 모델이 logits 와 함께 칸별 클래스 지도(CAM)도 내보낸다)
RUNTIME_VERSION = 2


class Runner:
    """추론 실행기. runner(clips (B, 3, T, H, W)) → logits (B, C) float32.
    runner.with_cam(clips) → (logits, cam (B, C, 7, 7)) — 같은 한 번의 추론에서 함께 나온다."""

    def __init__(self, name, fn, detail='', cam_fn=None):
        self.name = name        # 'torch' | 'torch-fp16' | 'openvino' | 'onnxruntime'
        self.detail = detail
        self._fn = fn
        self._cam_fn = cam_fn

    def __call__(self, clips):
        return self._fn(clips)

    def with_cam(self, clips):
        return self._cam_fn(clips)

    def __repr__(self):
        return f'Runner({self.name}{", " + self.detail if self.detail else ""})'


# ── 공통 ────────────────────────────────────────────────────────────

def _log(msg):
    print(f'[추론 실행기] {msg}', flush=True)


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_brand():
    """CPU 이름 — OS마다 읽는 곳이 다르다(Linux /proc/cpuinfo, macOS sysctl, Windows 레지스트리)."""
    system = platform.system()
    try:
        if system == 'Linux':
            with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        elif system == 'Darwin':
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        elif system == 'Windows':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
    except Exception:  # noqa: BLE001 — 이름을 못 읽어도 동작에는 지장 없다
        pass
    return platform.processor() or platform.machine()


def cpu_vendor():
    s = f'{cpu_brand()} {platform.processor()}'.lower()
    if 'intel' in s or 'genuineintel' in s:
        return 'intel'
    if 'amd' in s or 'authenticamd' in s:
        return 'amd'
    if 'apple' in s or (platform.system() == 'Darwin' and platform.machine() == 'arm64'):
        return 'apple'
    return 'other'


def _state_hash(model):
    """가중치 내용의 지문 — 같은 가중치면 변환 파일·검사 결과를 재사용한다."""
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _check_input(clip_shape, n):
    g = torch.Generator().manual_seed(1234)
    return torch.randn((n, *clip_shape), generator=g)


def _compare(ref, out):
    """기준 logits vs 비교 logits → (P(A) 최대 차이, logits 최대 차이)"""
    ref, out = ref.float().cpu(), out.float().cpu()
    d_prob = float((F.softmax(ref, 1)[:, 1] - F.softmax(out, 1)[:, 1]).abs().max())
    return d_prob, float((ref - out).abs().max())


def _check_against(ref_runner, runner, x):
    """PyTorch 기준과 결과 비교 → (P(A) 최대 차이, logits·CAM 중 큰 최대 차이). 통과 여부는 _passes."""
    ref, ref_cam = ref_runner.with_cam(x)
    out, cam = runner.with_cam(x)
    d_prob, d_logit = _compare(ref, out)
    d_cam = float((ref_cam.float().cpu() - cam.float().cpu()).abs().max())
    return d_prob, max(d_logit, d_cam)


def _passes(d_prob, d_logit):
    return d_prob <= config.INFER_CHECK_MAX_PROB_DIFF and d_logit <= 1e-2


def _time_per_clip(runner, x, reps=2):
    runner(x)                                   # 첫 호출(초기화) 제외
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        runner(x)
    sync()
    return (time.perf_counter() - t0) / (reps * x.shape[0])


# ── 방식별 실행기 ───────────────────────────────────────────────────

def _torch_runner(model, fp16=False):
    dev = next(model.parameters()).device

    def run(x):
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            return model(x).float()

    def run_cam(x):
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            logits, cam = model.forward_with_cam(x)
            return logits.float(), cam.float()
    return Runner('torch-fp16' if fp16 else 'torch', run, str(dev), run_cam)


class _TimeMean(torch.nn.Module):
    """AdaptiveAvgPool3d((1, None, None)) 와 같은 연산(시간축 평균). ONNX 가 앞의 것을 못 바꾼다."""

    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


class _WithCam(torch.nn.Module):
    """변환용 — forward_with_cam 을 forward 로 내보낸다(출력 두 개: logits, cam)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_with_cam(x)


def _export_copy(model):
    """변환용 CPU 복사본 — 모델 객체를 복제하지 않고 가중치만 옮겨 새로 만든다(출력 logits, cam)."""
    from hitandrun_model import HitAndRun3DCNN
    m = HitAndRun3DCNN(num_classes=model.head_conv.out_channels)
    m.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    m.eval()
    m.head_pool.pool = _TimeMean()
    return _WithCam(m).eval()


def _onnx_path(state_hash):
    return Path(config.INFER_CACHE_DIR) / f'x3d_{state_hash[:16]}_v{RUNTIME_VERSION}.onnx'


def _ensure_onnx(model, state_hash, clip_shape):
    """ONNX 변환 파일 — 있으면 그대로, 없으면 만든다(같은 가중치면 한 번만)."""
    path = _onnx_path(state_hash)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp{os.getpid()}.onnx')
    sample = torch.zeros((1, *clip_shape))
    kw = dict(input_names=['x'], output_names=['logits', 'cam'], opset_version=17,
              dynamic_axes={'x': {0: 'b'}, 'logits': {0: 'b'}, 'cam': {0: 'b'}})
    with warnings.catch_warnings(), torch.inference_mode():
        warnings.simplefilter('ignore')
        try:
            torch.onnx.export(_export_copy(model), (sample,), str(tmp), dynamo=False, **kw)
        except TypeError:          # dynamo 인자가 없는 옛 torch
            torch.onnx.export(_export_copy(model), (sample,), str(tmp), **kw)
    os.replace(tmp, path)          # 다른 프로세스가 반쯤 쓴 파일을 읽지 않게 다 쓴 뒤 바꿔 넣는다
    return path


def _openvino_runner(model, state_hash, clip_shape):
    import openvino as ov
    core = ov.Core()
    try:
        om = core.read_model(str(_ensure_onnx(model, state_hash, clip_shape)))
    except Exception:  # noqa: BLE001 — ONNX 를 못 만들면 PyTorch 모델에서 바로 바꾼다
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            om = ov.convert_model(_export_copy(model), example_input=torch.zeros((1, *clip_shape)))
    om.reshape({om.inputs[0]: ov.PartialShape([-1, *clip_shape])})
    # FP32 고정 — AMX 등이 있는 CPU 에서 OpenVINO 기본값은 BF16 이라 결과가 달라질 수 있다
    cm = core.compile_model(om, 'CPU', {'PERFORMANCE_HINT': 'LATENCY',
                                        'INFERENCE_PRECISION_HINT': 'f32'})
    req = cm.create_infer_request()
    out, out_cam = cm.outputs[0], cm.outputs[1]     # 변환 순서대로 logits, cam

    def run_cam(x):
        r = req.infer({0: x.detach().cpu().float().contiguous().numpy()})
        # 결과 버퍼는 다음 호출에서 덮어쓰이니 복사
        return torch.from_numpy(np.array(r[out])), torch.from_numpy(np.array(r[out_cam]))

    def run(x):
        return run_cam(x)[0]    # CAM 도 같은 추론에서 이미 계산된다(1×1 곱셈 49칸이라 비용은 거의 없음)
    return Runner('openvino', run, 'CPU f32', run_cam)


def _onnxruntime_runner(model, state_hash, clip_shape):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(_ensure_onnx(model, state_hash, clip_shape)), so,
                                providers=['CPUExecutionProvider'])

    def run(x):
        return torch.from_numpy(sess.run(['logits'], {'x': x.detach().cpu().float().contiguous().numpy()})[0])

    def run_cam(x):
        logits, cam = sess.run(['logits', 'cam'], {'x': x.detach().cpu().float().contiguous().numpy()})
        return torch.from_numpy(logits), torch.from_numpy(cam)
    return Runner('onnxruntime', run, 'CPU', run_cam)


_CPU_BUILDERS = {'openvino': _openvino_runner, 'onnxruntime': _onnxruntime_runner}


def _cpu_candidates():
    """CPU 후보 순서(같은 속도면 앞의 것). auto 는 어차피 전부 재서 가장 빠른 걸 고른다."""
    if cpu_vendor() == 'intel':
        return ['openvino', 'onnxruntime']
    return ['onnxruntime', 'openvino']


# ── 검사 결과 저장 ──────────────────────────────────────────────────

def _cache_file():
    return Path(config.INFER_CACHE_DIR) / 'backend_choice.json'


def _load_cache():
    try:
        return json.loads(_cache_file().read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(key, entry):
    try:
        data = _load_cache()
        data[key] = entry
        f = _cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f'{f.name}.tmp{os.getpid()}')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, f)
    except Exception as exc:  # noqa: BLE001 — 저장 못 해도 이번 실행은 그대로 동작
        _log(f'검사 결과를 저장하지 못했습니다(다음 실행 때 다시 검사): {exc}')


def _fingerprint(state_hash, clip_shape, batch):
    return {
        'runtime_version': RUNTIME_VERSION, 'weights': state_hash[:16],
        'clip_shape': list(clip_shape), 'batch': batch,
        'backend_cfg': (config.INFER_BACKEND or 'auto').lower(),
        'tol': config.INFER_CHECK_MAX_PROB_DIFF,
        'os': platform.system(), 'os_release': platform.release(), 'machine': platform.machine(),
        'cpu': cpu_brand(), 'torch': torch.__version__,
        'openvino': _pkg_version('openvino'), 'onnxruntime': _pkg_version('onnxruntime'),
    }


# ── 고르기 ──────────────────────────────────────────────────────────

def _fp16_supported(dev):
    """Tensor Core 가 있는 NVIDIA GPU(연산능력 7.0 이상)만 FP16 이 빠르다. AMD(ROCm)는 제외."""
    if dev.type != 'cuda' or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(dev) >= (7, 0)


def _select_gpu(model, clip_shape, batch):
    dev = next(model.parameters()).device
    base = _torch_runner(model)
    if not config.INFER_FP16:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP32')
        return base
    if not _fp16_supported(dev):
        _log(f'FP16 을 켰지만 이 GPU({torch.cuda.get_device_name(dev)})는 FP16 이 빠르지 않아 FP32 로 합니다')
        return base
    x = _check_input(clip_shape, batch).to(dev)
    half = _torch_runner(model, fp16=True)
    d_prob, _ = _compare(base(x), half(x))
    if d_prob <= config.INFER_FP16_MAX_PROB_DIFF:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP16 (자가 검사 통과: P(A) 차이 {d_prob:.1e})')
        return half
    _log(f'FP16 자가 검사 불통과(P(A) 차이 {d_prob:.1e} > {config.INFER_FP16_MAX_PROB_DIFF}) → FP32 로 합니다')
    return base


def _select_cpu(model, clip_shape, batch, force=False):
    cfg = (config.INFER_BACKEND or 'auto').lower()
    if cfg == 'torch':
        _log('PyTorch CPU (config.INFER_BACKEND=torch)')
        return _torch_runner(model)
    if cfg not in ('auto', *_CPU_BUILDERS):
        _log(f"알 수 없는 INFER_BACKEND '{cfg}' → auto 로 고릅니다")
        cfg = 'auto'

    state_hash = _state_hash(model)
    fp = _fingerprint(state_hash, clip_shape, batch)
    key = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()

    # ① 저장된 검사 결과가 있으면 속도는 다시 재지 않고, 창 1개로 결과만 빠르게 확인한다.
    #    저장 정보로는 안 잡히는 변화(설치가 깨짐 등)도 서버를 띄울 때마다 걸러낸다.
    saved = None if force else _load_cache().get(key)
    if saved:
        name = saved['choice']
        try:
            if name == 'torch':
                runner, note = _torch_runner(model), ''
            else:
                runner = _CPU_BUILDERS[name](model, state_hash, clip_shape)
                x1 = _check_input(clip_shape, 1)
                d_prob, d_logit = _check_against(_torch_runner(model), runner, x1)
                if not _passes(d_prob, d_logit):
                    raise RuntimeError(f'빠른 확인 불통과(P(A) 차이 {d_prob:.1e}, logits·CAM 차이 {d_logit:.1e})')
                note = f', 빠른 확인 통과(P(A) 차이 {d_prob:.1e})'
            _log(f"{name} — 저장된 자가 검사 결과 사용({saved.get('checked_at', '?')}, "
                 f"{saved.get('summary', '')}){note}")
            return runner
        except Exception as exc:  # noqa: BLE001
            _log(f'저장된 방식 {name} 이(가) 지금은 맞지 않아 다시 검사합니다: {exc}')

    # ② 자가 검사: PyTorch 결과를 기준으로 후보마다 결과 비교 + 속도
    _log(f'이 컴퓨터에서 처음 쓰는 조합이라 자가 검사를 합니다(한 번만) — CPU: {cpu_brand()}')
    x = _check_input(clip_shape, batch)
    ref_runner = _torch_runner(model)
    results = {'torch': {'ok': True, 'sec_per_clip': _time_per_clip(ref_runner, x)}}
    names = _cpu_candidates() if cfg == 'auto' else [cfg]
    built = {'torch': ref_runner}
    for name in names:
        try:
            runner = _CPU_BUILDERS[name](model, state_hash, clip_shape)
            d_prob, d_logit = _check_against(ref_runner, runner, x)
            ok = _passes(d_prob, d_logit)
            res = {'ok': ok, 'prob_diff': d_prob, 'logit_diff': d_logit}
            if ok:
                res['sec_per_clip'] = _time_per_clip(runner, x)
                built[name] = runner
            else:
                del runner
            results[name] = res
        except ModuleNotFoundError as exc:
            results[name] = {'ok': False, 'error': f'설치 안 됨 ({exc.name})'}
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 그 방식만 버린다
            results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        gc.collect()

    passed = {n: r for n, r in results.items() if r['ok']}
    if cfg == 'auto':
        choice = min(passed, key=lambda n: passed[n]['sec_per_clip'])
    else:
        choice = cfg if cfg in passed else 'torch'
    for n in list(built):
        if n != choice:
            del built[n]
    gc.collect()

    for n, r in results.items():
        if r['ok']:
            extra = (f", P(A) 차이 {r['prob_diff']:.1e}" if 'prob_diff' in r else ' (기준)')
            _log(f"  {n:12s} 통과  창 1개 {r['sec_per_clip']:.3f}초{extra}")
        else:
            why = r.get('error') or f"결과가 다름(P(A) 차이 {r['prob_diff']:.1e}, logits·CAM 차이 {r['logit_diff']:.1e})"
            _log(f'  {n:12s} 제외  {why}')
    t = results[choice]['sec_per_clip']
    summary = f"창 1개 {t:.3f}초, PyTorch 대비 ×{results['torch']['sec_per_clip'] / t:.2f}"
    _log(f'→ {choice} 선택 ({summary})')
    _save_cache(key, {'choice': choice, 'summary': summary,
                      'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                      'fingerprint': fp, 'results': results})
    return built[choice]


def get_runner(model, clip_shape=None, force_check=False):
    """model 에 맞는 실행기. 처음 한 번만 고르고 model 에 붙여 두어 재사용한다.

    force_check=True 면 저장된 검사 결과를 무시하고 다시 검사한다(check-backend 명령).
    """
    clip_shape = tuple(clip_shape or (3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0]))
    key = ((config.INFER_BACKEND or 'auto').lower(), bool(config.INFER_FP16), clip_shape)
    cached = model.__dict__.get('_infer_runner')
    if cached is not None and cached[0] == key and not force_check:
        return cached[1]
    model.eval()
    dev = next(model.parameters()).device
    batch = predict_batch_size(dev)     # 서비스가 실제로 쓰는 묶음 크기로 고른다(CPU 1 · GPU 8)
    if dev.type == 'cpu':
        runner = _select_cpu(model, clip_shape, batch, force=force_check)
    else:
        runner = _select_gpu(model, clip_shape, batch)
    model.__dict__['_infer_runner'] = (key, runner)
    return runner
