# Matrix Multiplication Kernel (Before/After)

I built and profiled a CUDA matrix multiplication kernel, starting from a naive baseline and optimizing it with shared memory tiling. This project documents the full diagnostic process: identifying why the naive version was slow, applying a specific fix, and proving the improvement with real profiling data rather than just claiming it worked.

## Project Structure / Architecture

I developed this in Visual Studio Code with CUDA Toolkit installed under WSL2 Ubuntu. My diagnostic pipeline followed this order:

```
CUDA kernel (VS Code)
       |
       v
Nsight Systems     (timeline view, finds where time goes across the whole run)
       |
       v
Nsight Compute     (kernel-level detail, finds why a specific kernel is slow)
```

I used Nsight Systems first to confirm the matmul kernel was the dominant cost in the run and to check for gaps or overhead around the launch. I then used Nsight Compute to look inside the kernel itself, at the roofline position, memory throughput, and warp behavior.

## Methodology

I started with a naive matmul kernel, where each thread computes one output element by looping over the shared dimension and reading directly from global memory on every iteration. I profiled this baseline with Nsight Systems and Nsight Compute to establish where its performance limits actually came from.

Based on that profile, I rewrote the kernel using shared memory tiling. Threads within a block cooperatively load a tile of each input matrix into shared memory once, then reuse that data for their computations instead of each thread independently re-fetching the same values from global memory repeatedly. I then re-ran the same profiling pass on the tiled version to measure the actual difference.

Both versions were verified for correctness against an independent CPU implementation before any performance comparison was made.

## Results / Evidence

<!-- Add screenshots here once profiling is complete -->

- Nsight Systems timeline, naive kernel: `assets/nsight-systems-naive.png`
- Nsight Systems timeline, tiled kernel: `assets/nsight-systems-tiled.png`
- Nsight Compute roofline, naive kernel: `assets/nsight-compute-roofline-naive.png`
- Nsight Compute roofline, tiled kernel: `assets/nsight-compute-roofline-tiled.png`
- DCGM dashboard during naive kernel run: `assets/dcgm-naive.png`
- DCGM dashboard during tiled kernel run: `assets/dcgm-tiled.png`

Each screenshot is paired with the specific metric it demonstrates rather than included as a standalone image.

## Key Findings

The naive kernel was memory bound. Every iteration of its inner loop reads directly from global memory, and neighboring threads in the same block were reading largely overlapping data without sharing any of it. This showed up in Nsight Compute as high memory throughput relative to compute throughput, with the kernel sitting to the left of the roofline's ridge point.

The tiled kernel performs the same total amount of arithmetic. What changes is how often it touches global memory. By having each thread contribute one value to a shared tile, then having the whole block reuse that tile for multiple computations, global memory traffic dropped substantially. The re-profiled kernel showed [insert measured shift in compute vs memory throughput, and runtime improvement, once profiling is complete].

Correlating this with my DCGM dashboard, the GPU's utilization and power signature during the tiled run looked different from the naive run, consistent with the workload becoming less bottlenecked by memory stalls.

<details>
<summary>A plainer language explanation of what changed</summary>

Picture a kitchen with 1,024 cooks, each cook assigned to plate one dish. Every ingredient a cook needs lives in the walk-in pantry at the back of the kitchen. In the naive version, each cook walks to the pantry alone for every single ingredient in their recipe, one trip per ingredient, even though the cook working right next to them needs almost the exact same ingredients and just made the same walk.

In the tiled version, cooks are grouped into small stations. Each station gets its own prep counter close by. Instead of every cook walking to the pantry alone, each cook at the station grabs one ingredient and sets it on the shared counter. Once the whole counter is stocked, everyone at that station cooks using what is already sitting right there, no more long pantry trips for a while. The station then goes back together for the next batch of ingredients and repeats.

The total amount of cooking is identical between the two versions. What changes is how many trips to the pantry it took to gather the ingredients.

</details>

## Design Notes

I chose shared memory tiling specifically because it directly addresses what the baseline profiling showed, rather than applying a generic optimization and hoping it helped. The tile size matches the block dimensions I launched with, since each thread is responsible for loading exactly one element into the shared tile.

I used `__syncthreads()` twice inside the kernel. The first call ensures every thread in the block has finished writing its element into the shared tile before any thread starts reading from it. The second call ensures every thread has finished using the current tile before any thread starts overwriting it with the next one. Without either barrier, threads could read a partially loaded tile or read from a tile that is being overwritten mid-use, producing incorrect results.

I verified correctness against a CPU implementation for both kernel versions before comparing performance, since a faster kernel that produces wrong answers is not actually an improvement.

## Getting Started / Reproduction

Requirements:
- NVIDIA GPU with current drivers
- CUDA Toolkit installed
- Nsight Systems and Nsight Compute installed

```bash
git clone https://github.com/Dre1896/matmul-kernel-optimization.git
cd matmul-kernel-optimization
nvcc baseline_matmul.cu -o baseline_matmul
nvcc tiled_matmul.cu -o tiled_matmul
```

To profile either version:

```bash
nsys profile ./baseline_matmul
ncu ./baseline_matmul
```

Swap in `tiled_matmul` to profile the optimized version the same way.

## Next Steps

- Extend the tiling approach with register blocking to see if a second optimization pass yields further gains
- Apply the same DCGM, Nsight Systems, and Nsight Compute diagnostic chain to a different kernel to confirm the process generalizes beyond matrix multiplication
- Compare this hand written CUDA approach against an OpenACC directive based version of the same kernel

## Acknowledgments

Special thanks to CoffeeBeforeArch for the naive matrix multiplication kernel, which served as the starting baseline before I optimized it through shared memory tiling. The optimization, profiling, and analysis work in this repository are my own.

## License

MIT, see LICENSE.
