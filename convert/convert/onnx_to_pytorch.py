#!/usr/bin/env python3
# =============================================================================
# onnx_to_pytorch.py — SSD-MobileNet V2 raw ONNX → PyTorch TorchScript
# =============================================================================
# 역할:
#   SSD-MV2 ONNX를 TorchScript로 변환한다.
#
# 핵심 안정화:
#   1) raw ONNX의 class_scores shape가 비정상([1,1,N])이면 TFOD ONNX에서
#      raw_detection_boxes/raw_detection_scores를 다시 추출해 재생성한다.
#   2) onnx2torch 1.5.x에서 발생하는 "(tensor,) tuple 입력" 버그를 pre-hook으로
#      우회한다.
#   3) 입력은 벤치마크 파이프라인 규격(NCHW float32)이므로 내부에서 NHWC float32로
#      변환하는 어댑터를 씌워 trace 한다.
#
# 출력:
#   models/ssd_mobilenet_v2.torchscript
# =============================================================================

import argparse
import os
import sys


def _output_shape_from_value_info(v):
    dims = []
    try:
        for d in v.type.tensor_type.shape.dim:
            dims.append(int(d.dim_value) if getattr(d, "dim_value", 0) > 0 else None)
    except Exception:
        return []
    return dims


def _is_suspicious_raw_onnx(onnx_path):
    """class_scores가 [1,1,N] 형태면 비정상 raw로 간주."""
    try:
        import onnx
        m = onnx.load(onnx_path)
    except Exception:
        return False

    outputs = {o.name: o for o in m.graph.output}
    cls = outputs.get("class_scores")
    if cls is None:
        return True
    dims = _output_shape_from_value_info(cls)
    if len(dims) != 3:
        return True
    # 기대는 대개 [1, N, 91]
    c1 = dims[1] if dims[1] is not None else -1
    c2 = dims[2] if dims[2] is not None else -1
    # [1,1,1917] 같은 케이스 탐지
    if min(c1, c2) in (0, 1, 2) and max(c1, c2) >= 100:
        return True
    return False


def _build_raw_from_tfod(tfod_onnx_path, output_path):
    """TFOD ONNX에서 raw_detection_*을 끌어와 box/class 2-output ONNX로 재작성."""
    try:
        import onnx
        import onnx_graphsurgeon as gs
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"missing dependency for raw rebuild: {e}")

    g = gs.import_onnx(onnx.load(tfod_onnx_path))
    t = g.tensors()
    rb = t.get("raw_detection_boxes")
    rs = t.get("raw_detection_scores")
    if rb is None or rs is None:
        raise RuntimeError("raw_detection_boxes/raw_detection_scores not found")

    rb.name = "box_encodings"
    rs.name = "class_scores"
    if getattr(rb, "dtype", None) is None:
        rb.dtype = np.float32
    if getattr(rs, "dtype", None) is None:
        rs.dtype = np.float32
    g.outputs = [rb, rs]
    g.cleanup().toposort()
    onnx.save(gs.export_onnx(g), output_path)
    return output_path


def _select_effective_onnx(onnx_path):
    """변환에 사용할 ONNX 경로를 선택/재생성한다."""
    if not os.path.exists(onnx_path):
        return onnx_path

    if not _is_suspicious_raw_onnx(onnx_path):
        return onnx_path

    model_dir = os.path.dirname(onnx_path) or "."
    candidates = [
        os.path.join(model_dir, "ssd_mobilenet_v2_tfod_fpinput.onnx"),
        os.path.join(model_dir, "ssd_mobilenet_v2_tfod.onnx"),
    ]
    out_path = os.path.join(model_dir, "ssd_mobilenet_v2_raw_rebuilt.onnx")

    for cand in candidates:
        if not os.path.exists(cand):
            continue
        try:
            _build_raw_from_tfod(cand, out_path)
            print(f"[prep] rebuilt raw ONNX from {cand} -> {out_path}")
            return out_path
        except Exception as e:
            print(f"[WARN] raw rebuild failed from {cand}: {e}")
            continue

    return onnx_path


def _register_tuple_unwrap_hooks(model):
    """onnx2torch가 (tensor,)를 전달하는 버그 우회."""

    def _hook(_module, args):
        if not args:
            return None
        changed = False
        new_args = []
        for a in args:
            if isinstance(a, (tuple, list)) and len(a) == 1:
                new_args.append(a[0])
                changed = True
            else:
                new_args.append(a)
        return tuple(new_args) if changed else None

    for mod in model.modules():
        mod.register_forward_pre_hook(_hook)


def _trace_with_adapter(torch, core_model, input_shape):
    """NCHW float32 입력을 NHWC float32로 맞춘 뒤 trace."""

    class _InputAdapter(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            # benchmark 입력 규격: NCHW float32
            if x.ndim == 4 and x.shape[1] in (1, 3):
                x = x.permute(0, 2, 3, 1).contiguous()
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            return self.core(x)

    wrapped = _InputAdapter(core_model).eval()
    dummy = torch.randn(*input_shape)

    # dry-run으로 출력 구조 점검
    with torch.inference_mode():
        out = wrapped(dummy)
    if not (isinstance(out, (tuple, list)) and len(out) >= 2):
        raise RuntimeError(f"unexpected output structure: {type(out)}")

    with torch.inference_mode():
        scripted = torch.jit.trace(
            wrapped, dummy, strict=False, check_trace=False
        )
    return scripted


def convert_onnx_to_torchscript(onnx_path, output_path, input_shape=(1, 3, 320, 320)):
    try:
        import torch
    except ImportError:
        print("[ERROR] torch 필요: pip install torch==2.1.0 torchvision==0.16.0")
        sys.exit(1)

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX 파일이 없습니다: {onnx_path}")
        print("  convert/tf2_od_to_onnx.py 를 먼저 실행하세요.")
        sys.exit(1)

    effective_onnx = _select_effective_onnx(onnx_path)
    print(f"[spec] effective ONNX: {effective_onnx}")

    scripted = None
    converter_used = None

    # =====================================================================
    # 1) onnx2torch 우선
    # =====================================================================
    print(f"[1/3] Trying onnx2torch on {effective_onnx}...")
    try:
        from onnx2torch import convert as o2t_convert
        model = o2t_convert(effective_onnx).eval()
        _register_tuple_unwrap_hooks(model)
        scripted = _trace_with_adapter(torch, model, input_shape)
        converter_used = "onnx2torch"
        print("  [OK] onnx2torch + adapter trace 성공")
    except ImportError as e:
        print(f"  [SKIP] onnx2torch 미설치: {e}")
    except Exception as e:
        print(f"  [WARN] onnx2torch 경로 실패: {e}")

    # =====================================================================
    # 2) onnx2pytorch fallback
    # =====================================================================
    if scripted is None:
        print("[2/3] Trying onnx2pytorch fallback...")
        try:
            import onnx
            from onnx2pytorch import ConvertModel
            model = ConvertModel(onnx.load(effective_onnx)).eval()
            scripted = _trace_with_adapter(torch, model, input_shape)
            converter_used = "onnx2pytorch"
            print("  [OK] onnx2pytorch + adapter trace 성공")
        except ImportError as e:
            print(f"  [SKIP] onnx2pytorch 미설치: {e}")
        except Exception as e:
            print(f"  [ERROR] onnx2pytorch 경로도 실패: {e}")

    if scripted is None:
        print("\n[FAIL] SSD-MV2 TorchScript 변환 실패.")
        print("  pytorch_cpu / pytorch_cuda 조합은 N/A 처리 필요.")
        sys.exit(1)

    # =====================================================================
    # 3) 저장
    # =====================================================================
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n[DONE] TorchScript 저장 완료")
    print(f"  path:      {output_path} ({size_mb:.1f} MB)")
    print(f"  converter: {converter_used}")
    print(f"  input:     shape={input_shape}, dtype=float32 (NCHW)")


def main():
    parser = argparse.ArgumentParser(
        description="SSD-MV2 raw ONNX → PyTorch TorchScript"
    )
    parser.add_argument(
        "--onnx",
        default="models/ssd_mobilenet_v2_raw.onnx",
        help="Input raw ONNX path (default: models/ssd_mobilenet_v2_raw.onnx)",
    )
    parser.add_argument(
        "--output",
        default="models/ssd_mobilenet_v2.torchscript",
        help="Output TorchScript path",
    )
    parser.add_argument("--height", type=int, default=320, help="Input H (default 320)")
    parser.add_argument("--width", type=int, default=320, help="Input W (default 320)")
    args = parser.parse_args()

    convert_onnx_to_torchscript(
        args.onnx, args.output, input_shape=(1, 3, args.height, args.width)
    )


if __name__ == "__main__":
    main()
