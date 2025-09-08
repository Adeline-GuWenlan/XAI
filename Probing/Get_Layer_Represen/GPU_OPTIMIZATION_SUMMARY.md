# GPU Optimization Summary

## Changes Made

The following files have been optimized for GPU utilization based on the recommendations in `claude.md`:

### 1. `hooks_regist_only.py` - LayerwiseCLSProbeExact

**Key Optimizations:**
- ✅ Added proper device management in `__init__()` 
- ✅ Model placement verification with device printing
- ✅ Enabled `torch.backends.cudnn.benchmark = True` for CUDA
- ✅ Replaced `@torch.no_grad()` with `torch.inference_mode()` for better GPU performance
- ✅ Added mixed precision (`torch.cuda.amp.autocast`) for CUDA devices
- ✅ Added tensor device verification with one-time printing
- ✅ Added `time_single_batch()` method for performance monitoring

**Before:**
```python
@torch.no_grad()
def run_once(self, rho_real, rho_imag):
    self.register_hooks()
    _ = self.model(rho_real, rho_imag)
    # ...
```

**After:**
```python
def run_once(self, rho_real, rho_imag):
    with torch.inference_mode():
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.model(rho_real, rho_imag)
        else:
            _ = self.model(rho_real, rho_imag)
    # ...
```

### 2. `new_prober.py` - extract_layer_csv_exact

**Key Optimizations:**
- ✅ Increased default batch size from 512 → 1024 (adjustable)
- ✅ Increased default num_workers from 4 → 8
- ✅ Added device setup with GPU count verification
- ✅ Enabled `torch.backends.cudnn.benchmark = True`
- ✅ Added model device placement verification
- ✅ Optimized DataLoader with proper `pin_memory`, `persistent_workers`, `prefetch_factor`
- ✅ Wrapped main processing loop with `torch.inference_mode()`
- ✅ Added batch timing with CUDA synchronization
- ✅ Added performance monitoring every 10 batches
- ✅ Added final performance summary with throughput calculation

**Enhanced DataLoader:**
```python
loader = DataLoader(
    ds, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=min(num_workers, os.cpu_count()) if num_workers > 0 else 0,
    pin_memory=(device_type == 'cuda'),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None
)
```

**Performance Monitoring:**
```python
with torch.inference_mode():
    for batch_i, (rho_real, rho_imag, _) in enumerate(loader):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        batch_start = time.time()
        
        # Processing...
        
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        batch_time = time.time() - batch_start
        
        if batch_i % 10 == 0:
            avg_time = np.mean(batch_times[-10:])
            print(f"Batch {batch_i}/{len(loader)}: {avg_time:.3f}s/batch")
```

### 3. `RUN_new_prober.py` - Configuration Script

**Key Optimizations:**
- ✅ Added `CUDA_VISIBLE_DEVICES=0` environment variable
- ✅ Increased batch size from 512 → 1024
- ✅ Increased num_workers from 4 → 8

**Before:**
```python
"batch_size": 512,
# ...
num_workers = 4,
```

**After:**
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 as recommended
# ...
"batch_size": 1024,  # Increased from 512 to better utilize GPU
# ...
num_workers = 8,  # Increased from 4 to min(8, cpu_count())
```

### 4. `test_gpu_optimization.py` - Validation Script (NEW)

**Features:**
- ✅ Device setup and model placement verification
- ✅ Input tensor device placement testing
- ✅ Inference mode forward pass validation
- ✅ GPU timing and throughput measurement
- ✅ Comprehensive test suite with pass/fail reporting

## Expected Performance Improvements

### Before Optimization:
- GPU Utilization: 0% (CPU-bound)
- VRAM Usage: ~424 MiB (underutilized)
- Batch Size: 512
- Workers: 4

### After Optimization:
- ✅ GPU Utilization: Should see significant increase during processing
- ✅ Better VRAM utilization with larger batches
- ✅ Batch Size: 1024 (2x increase)
- ✅ Workers: 8 (2x increase for better I/O)
- ✅ Mixed precision (FP16) for 2x memory efficiency on compatible GPUs
- ✅ `torch.inference_mode()` for better GPU kernel scheduling
- ✅ CuDNN benchmark enabled for consistent workloads

## How to Test

1. Run the validation script:
   ```bash
   cd /Users/guwenlan/Desktop/XAI/Probing/Get_Layer_Represen
   python test_gpu_optimization.py
   ```

2. Run the actual probe with GPU monitoring:
   ```bash
   # In terminal 1:
   watch -n 1 nvidia-smi
   
   # In terminal 2:
   python RUN_new_prober.py
   ```

3. Look for these indicators of successful optimization:
   - GPU-Util > 0% (should see spikes during processing)
   - Batch timing output showing reasonable throughput
   - Console output confirming device placement
   - Performance summary at the end

## Troubleshooting

If GPU utilization is still low:

1. **Check tensor placement**: Look for console output showing both model and input tensors on CUDA
2. **Verify batch size**: Increase further if VRAM allows (monitor with `nvidia-smi`)
3. **Check data loading**: If batches process fast but overall slow, it's I/O bound
4. **Monitor workers**: Reduce num_workers if CPU is bottleneck

The optimizations follow all recommendations from `claude.md` and should resolve the GPU underutilization issue.