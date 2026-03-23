import pyhip
hip = pyhip.module("depthwise_conv3d.cpp", "-g -O3")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import argparse

torch.cuda.set_device(3)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=} {torch.cuda.device_count()}")


def benchmark_op(op_func, op_name, iters, gflops, device):
    print(f"\n正在进行 {op_name} 预热/Tune...")
    
    # 统计预热/Tune 耗时
    os.putenv("PYTORCH_TUNABLEOP_TUNING", "1")
    warmup_start = time.time()
    for _ in range(10):
        _ = op_func()
    os.unsetenv("PYTORCH_TUNABLEOP_TUNING")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_end = time.time()
    warmup_time_ms = (warmup_end - warmup_start) * 1000
    print(f"预热/Tune 完成，耗时: {warmup_time_ms:.2f} ms")

    print(f"开始 {op_name} 性能测试 ({iters} 次迭代)...")
    start_time = time.time()
    
    for _ in range(iters):
        _ = op_func()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / iters * 1000
    tflops = (gflops / 1000.0) / (avg_time_ms / 1000.0) if avg_time_ms > 0 else 0
    
    return avg_time_ms, tflops, warmup_time_ms

def test_conv3d_benchmark(args):
    # 1. 准备数据
    if args.shape == "case1":
        # Case 1: [1, 64, 63, 45, 80] x [512, 64, 3, 3, 3]
        B, C_in, C_out, D, H, W = 1, 64, 512, 63, 45, 80
        kernel_size = (3, 3, 3)
        padding = (1, 1, 1)
        groups = 5
    elif args.shape == "case2":
        # Case 2: [1, 512, 61, 45, 80] x [2048, 512, 1, 1, 1]
        B, C_in, C_out, D, H, W = 1, 512, 2048, 61, 45, 80
        kernel_size = (1, 1, 1)
        padding = (0, 0, 0)
        groups = 1
    elif args.shape == "case3":
        # Case 3: [1, 512, 61, 45, 80] x [512, 1, 3, 5, 5], groups=512
        B, C_in, C_out, D, H, W = 1, 512, 512, 61, 45, 80
        kernel_size = (3, 5, 5)
        padding = (0, 2, 2)
        groups = 512
    elif args.shape == "case3_32x32":
        # Case 3: [1, 32, 61, 45, 80] x [32, 1, 3, 5, 5], groups=32
        B, C_in, C_out, D, H, W = 1, 32, 32, 61, 45, 80
        kernel_size = (3, 5, 5)
        padding = (0, 2, 2)
        groups = 32
    elif args.shape == "case4":
        # Case 4: [1, 2048, 61, 45, 80] x [2048, 512, 1, 1, 1], groups=4
        B, C_in, C_out, D, H, W = 1, 2048, 2048, 61, 45, 80
        kernel_size = (1, 1, 1)
        padding = (0, 0, 0)
        groups = 512
    else:
        raise ValueError(f"不支持的 shape 选项: {args.shape}")

    stride = (1, 1, 1)
    dilation = (1, 1, 1)

    #input_dtype = torch.bfloat16
    input_dtype = torch.float16
    #input_dtype = torch.float32
    #input_dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- 正在初始化 Conv3d 数据 ({args.shape}) ---")
    print(f"Input Shape: [{B}, {C_in}, {D}, {H}, {W}]")
    print(f"Weight Shape: [{C_out}, {C_in // groups}, {kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}]")
    print(f"Groups: {groups}")
    print(f"Device: {device}, Dtype: {input_dtype}")
    
    # 初始化输入、权重和偏置
    input_tensor = torch.randn(B, C_in, D, H, W).to(dtype=input_dtype).to(device)
    weight_tensor = torch.randn(C_out, C_in // groups, *kernel_size).to(dtype=input_dtype).to(device)
    bias_tensor = torch.randn(C_out).to(dtype=input_dtype).to(device)
  
    # 内存格式处理
    # try:
    #     input_tensor = input_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     weight_tensor = weight_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     bias_tensor = bias_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     print("使用 channels_last_3d 内存格式")
    # except:
    #     input_tensor = input_tensor.contiguous()
    #     print("使用默认内存格式 (NCHW)")
        
    # 计算输出尺寸
    D_out = (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
    print(f"输出尺寸: [{B}, {C_out}, {D_out}, {H_out}, {W_out}]")

    # 计算量 (GFLOPs)
    gflops = (2.0 * B * C_out * D_out * H_out * W_out * (C_in // groups) * kernel_size[0] * kernel_size[1] * kernel_size[2]) / 1e9
    print(f"理论计算量: {gflops:.4f} GFLOPs")

    # 定义测试操作
    def run_torch_conv3d():
        return F.conv3d(input_tensor, weight_tensor, 
                        bias=bias_tensor,
                        stride=stride, 
                        padding=padding, 
                        dilation=dilation,
                        groups=groups)

    def run_custom_conv3d_reference():
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            num_output = B * C_out * D_out * H_out * W_out
            grid_dim = (num_output + 255) // 256
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_reference(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor
    
    def run_custom_conv3d_reference_bf16():
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            num_output = B * C_out * D_out * H_out * W_out
            grid_dim = (num_output + 255) // 256
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_reference_bf16(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor

    def run_custom_conv3d_opt1():
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            # Suggestion B (Option 2): block over (b, oc, ot), 256 threads over (oh, ow)
            plane_size = H_out * W_out
            num_tiles = (plane_size + 255) // 256
            grid_dim = B * C_out * D_out * num_tiles
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_opt1(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor

    def run_custom_conv3d_opt2():
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            # A+B+C: block over (b, oc, ot), 16x16 tile (match cpp TILE_H, TILE_W)
            TILE_H, TILE_W = 16, 16
            num_tiles_h = (H_out + TILE_H - 1) // TILE_H
            num_tiles_w = (W_out + TILE_W - 1) // TILE_W
            num_tiles = num_tiles_h * num_tiles_w
            grid_dim = B * C_out * D_out * num_tiles
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_opt2(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor

    def run_custom_conv3d_opt3():
        """opt3 specialized for case3 (shape3); B * C_out * D_out, block 256."""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_opt3(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor

    def run_custom_conv3d_opt3_bf16():
        """opt3 specialized for case3 (shape3); B * C_out * D_out, block 256."""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            hip.conv_depthwise3d_cuda_kernel_opt3_bf16(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
            return output_tensor

    def _smem_bytes_opt3_general():
        """Dynamic shared memory for opt3_general / opt3_bf16_general kernels."""
        kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
        weight_size = kT * kH * kW
        in_tile_h = (H_out - 1) * stride[1] + (kH - 1) * dilation[1] + 1
        in_tile_w = (W_out - 1) * stride[2] + (kW - 1) * dilation[2] + 1
        input_patch_size = kT * in_tile_h * in_tile_w
        sizeof_bf16 = 2
        return (weight_size + input_patch_size) * sizeof_bf16

    def run_custom_conv3d_opt3_bf16_general():
        """Generalized opt3 bf16: any (oT, oH, oW); dynamic shared memory."""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            smem = _smem_bytes_opt3_general()
            hip.conv_depthwise3d_cuda_kernel_opt3_bf16_general(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
                sharedMemBytes=smem,
            )
            return output_tensor
    
    def run_custom_conv3d_opt3_bf16_general_vec_dot():
        """Generalized opt3 bf16: any (oT, oH, oW); dynamic shared memory."""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=input_dtype)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            smem = _smem_bytes_opt3_general()
            hip.conv_depthwise3d_cuda_kernel_opt3_bf16_general_vec_dot(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
                sharedMemBytes=smem,
            )
            return output_tensor

    def run_custom_conv3d_opt3_fp16_general_vec_dot():
        """Generalized opt3 fp16 + fdot2 vec: 与 opt3_general 相同 launch，热路径用 amdgcn fdot2。"""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=torch.float16)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            smem = _smem_bytes_opt3_general()
            hip.conv_depthwise3d_cuda_kernel_opt3_fp16_general_vec_dot(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
                sharedMemBytes=smem,
            )
            return output_tensor

    def run_custom_conv3d_opt3_general():
        """Generalized opt3 fp16: any (oT, oH, oW); dynamic shared memory. Use only when input_dtype is torch.float16."""
        with torch.no_grad():
            output_tensor = torch.zeros(B, C_out, D_out, H_out, W_out, device=device, dtype=torch.float16)
            grid_dim = B * C_out * D_out
            kT, kH, kW = kernel_size[0], kernel_size[1], kernel_size[2]
            smem = _smem_bytes_opt3_general()
            hip.conv_depthwise3d_cuda_kernel_opt3_general(
                [grid_dim], [256],
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                bias_tensor.data_ptr(),
                B, C_in, C_out, D, H, W, D_out, H_out, W_out,
                kT, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
                sharedMemBytes=smem,
            )
            return output_tensor

    # 2. 运行 Benchmark
    torch_ms, torch_tflops, torch_warmup_ms = benchmark_op(run_torch_conv3d, f"Standard PyTorch Conv3d ({args.shape})", args.iters, gflops, device)
    # custom_ref_ms, custom_ref_tflops, custom_ref_warmup_ms = benchmark_op(run_custom_conv3d_reference, f"Custom Conv3d Reference ({args.shape})", args.iters, gflops, device)
    custom_ref_bf16_ms, custom_ref_bf16_tflops, custom_ref_bf16_warmup_ms = benchmark_op(run_custom_conv3d_reference_bf16, f"Custom Conv3d Reference BF16 ({args.shape})", args.iters, gflops, device)
    # custom_opt1_ms, custom_opt1_tflops, custom_opt1_warmup_ms = benchmark_op(run_custom_conv3d_opt1, f"Custom Conv3d Opt1 ({args.shape})", args.iters, gflops, device)
    # custom_opt2_ms, custom_opt2_tflops, custom_opt2_warmup_ms = benchmark_op(run_custom_conv3d_opt2, f"Custom Conv3d Opt2 ({args.shape})", args.iters, gflops, device)
    custom_opt3_general_ms, custom_opt3_general_tflops, custom_opt3_general_warmup_ms = benchmark_op(run_custom_conv3d_opt3_general, f"Custom Conv3d Opt3 general ({args.shape})", args.iters, gflops, device)
    custom_opt3_bf16_ms, custom_opt3_bf16_tflops, custom_opt3_bf16_warmup_ms = benchmark_op(run_custom_conv3d_opt3_bf16, f"Custom Conv3d Opt3 BF16 ({args.shape})", args.iters, gflops, device)
    custom_opt3_bf16_general_ms, custom_opt3_bf16_general_tflops, custom_opt3_bf16_general_warmup_ms = benchmark_op(run_custom_conv3d_opt3_bf16_general, f"Custom Conv3d Opt3 BF16 general ({args.shape})", args.iters, gflops, device)
    custom_opt3_bf16_general_vec_dot_ms, custom_opt3_bf16_general_vec_dot_tflops, custom_opt3_bf16_general_vec_dot_warmup_ms = benchmark_op(run_custom_conv3d_opt3_bf16_general_vec_dot, f"Custom Conv3d Opt3 BF16 general vec dot ({args.shape})", args.iters, gflops, device)
    custom_opt3_fp16_general_vec_dot_ms, custom_opt3_fp16_general_vec_dot_tflops, custom_opt3_fp16_general_vec_dot_warmup_ms = benchmark_op(run_custom_conv3d_opt3_fp16_general_vec_dot, f"Custom Conv3d Opt3 FP16 general vec dot ({args.shape})", args.iters, gflops, device)
    # 3. 汇总对比
    print(f"\n--- 性能对比汇总 ({args.shape}) ---")
    print(f"{'方法':<35} | {'平均耗时 (ms)':<15} | {'吞吐量 (TFLOPS)':<15} | {'预热/Tune (ms)':<15}")
    print("-" * 90)
    print(f"{'Standard PyTorch Conv3d':<35} | {torch_ms:>15.4f} | {torch_tflops:>15.2f} | {torch_warmup_ms:>15.2f}")
    # print(f"{'Custom Conv3d Reference':<35} | {custom_ref_ms:>15.4f} | {custom_ref_tflops:>15.2f} | {custom_ref_warmup_ms:>15.2f}")
    print(f"{'Custom Conv3d Reference BF16':<35} | {custom_ref_bf16_ms:>15.4f} | {custom_ref_bf16_tflops:>15.2f} | {custom_ref_bf16_warmup_ms:>15.2f}")
    # print(f"{'Custom Conv3d Opt1':<35} | {custom_opt1_ms:>15.4f} | {custom_opt1_tflops:>15.2f} | {custom_opt1_warmup_ms:>15.2f}")
    # print(f"{'Custom Conv3d Opt2':<35} | {custom_opt2_ms:>15.4f} | {custom_opt2_tflops:>15.2f} | {custom_opt2_warmup_ms:>15.2f}")
    print(f"{'Custom Conv3d Opt3 general (F16)':<35} | {custom_opt3_general_ms:>15.4f} | {custom_opt3_general_tflops:>15.2f} | {custom_opt3_general_warmup_ms:>15.2f}")
    #print(f"{'Custom Conv3d Opt3 BF16':<35} | {custom_opt3_bf16_ms:>15.4f} | {custom_opt3_bf16_tflops:>15.2f} | {custom_opt3_bf16_warmup_ms:>15.2f}")
    print(f"{'Custom Conv3d Opt3 BF16 general':<35} | {custom_opt3_bf16_general_ms:>15.4f} | {custom_opt3_bf16_general_tflops:>15.2f} | {custom_opt3_bf16_general_warmup_ms:>15.2f}")
    print(f"{'Custom Conv3d Opt3 BF16 general vec dot':<35} | {custom_opt3_bf16_general_vec_dot_ms:>15.4f} | {custom_opt3_bf16_general_vec_dot_tflops:>15.2f} | {custom_opt3_bf16_general_vec_dot_warmup_ms:>15.2f}")
    print(f"{'Custom Conv3d Opt3 FP16 general vec dot':<35} | {custom_opt3_fp16_general_vec_dot_ms:>15.4f} | {custom_opt3_fp16_general_vec_dot_tflops:>15.2f} | {custom_opt3_fp16_general_vec_dot_warmup_ms:>15.2f}")

    print(f"Run Accuracy check for {args.shape}...")
    ref = run_torch_conv3d()
    print(ref.shape, ref.dtype)
    ret = run_custom_conv3d_reference_bf16()
    # ret = run_custom_conv3d_opt1()
    # ret = run_custom_conv3d_opt2()
    # ret = run_custom_conv3d_opt3()
    # ret = run_custom_conv3d_opt3_bf16_general()
    #ret = run_custom_conv3d_opt3_bf16_general_vec_dot()
    ret = run_custom_conv3d_opt3_fp16_general_vec_dot()
    print(ret.shape, ret.dtype)
    all_diff = pyhip.calc_diff(ref, ret)
    if all_diff > 0.001:
        for iib in range(B):
            for iic in range(C_out):
                for iid in range(D_out):
                    iiref = ref[iib, iic, iid, ...]
                    iiret = ret[iib, iic, iid, ...]
                    passed = torch.allclose(iiref, iiret, atol=0.01, rtol=0.01)
                    diff = pyhip.calc_diff(iiref, iiret)
                    if not passed and diff > 0.001:
                        print(f"================ {B},{C_out},{D_out} : {iib}, {iic}, {iid}    {diff=}", )
                        print(ref.shape)
                        print(ret.shape)
                        print(iiref)
                        print(iiret)
                        print(iiret.view(-1)[:32].view(4,8))
                        assert 0
    print(f"Accuracy check passed for {args.shape}")

    # 4. Profile (可选)
    if args.profile:
        print(f"\n开始 Torch Profile (Standard PyTorch Conv3d - {args.shape})...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/conv3d_{args.shape}_profile'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for _ in range(5):
                run_torch_conv3d()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        print(f"\nProfile 跟踪已保存到 ./log/conv3d_{args.shape}_profile")

if __name__ == "__main__":
    os.putenv("PYTORCH_TUNABLEOP_ENABLED", "1")

    parser = argparse.ArgumentParser(description="Conv3d Benchmark 脚本")
    parser.add_argument("--iters", type=int, default=10, help="迭代次数")
    parser.add_argument("--profile", action="store_true", help="是否启用 profile")
    parser.add_argument("--shape", type=str, default="case3", choices=["case1", "case2", "case3", "case3_32x32", "case4"], 
                        help="选择测试的 shape 选项: case1 ([1, 64, 63, 45, 80]), case2 ([1, 512, 61, 45, 80]), case3 ([1, 512, 61, 45, 80], groups=512), case3_32x32 ([1, 32, 61, 45, 80], groups=32), case4 ([1, 2048, 61, 45, 80], groups=4)")
    args = parser.parse_args()
    
    test_conv3d_benchmark(args)

