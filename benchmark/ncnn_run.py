#!/usr/bin/env python3
# 독립 실행 ncnn inference — stdin: JSON {param,bin,input_name,use_vulkan,shape}, stdout: JSON {ok,lat_ms,shape} or {error}
import sys, os, json, time
try:
    data = json.loads(sys.stdin.read())
    import numpy as np
    import ncnn
    net = ncnn.Net()
    if data['use_vulkan']:
        net.opt.use_vulkan_compute = True
    net.load_param(data['param'])
    net.load_model(data['bin'])
    C, H, W = data['shape']
    arr = np.random.randn(C, H, W).astype(np.float32)
    output_names = ['out0','output','prob','1000','softmax','cls_score','detection_out','boxes','scores','output0']

    # warmup
    ex = net.create_extractor()
    mat_in = ncnn.Mat(arr)
    ex.input(data['input_name'], mat_in)
    out_arr = None
    for oname in output_names:
        try:
            ret, mat_out = ex.extract(oname)
            if ret == 0 and mat_out is not None:
                out_arr = np.array(mat_out).flatten()
                break
        except Exception:
            continue
    if out_arr is None:
        print(json.dumps({'error':'no output extracted'}))
        sys.exit(0)

    # measure 3 iters
    lats = []
    for _ in range(3):
        ex = net.create_extractor()
        mat_in = ncnn.Mat(arr)
        t0 = time.perf_counter()
        ex.input(data['input_name'], mat_in)
        for oname in output_names:
            try:
                ret, mat_out = ex.extract(oname)
                if ret == 0 and mat_out is not None:
                    _ = np.array(mat_out).flatten()
                    break
            except Exception:
                continue
        lats.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({'ok':True, 'lat_ms':float(sum(lats)/len(lats)), 'shape':list(out_arr.shape)}))
except Exception as e:
    print(json.dumps({'error':str(e)}))
