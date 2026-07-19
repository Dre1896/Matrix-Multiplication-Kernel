// Tiled matrix multiplication — optimized version of the naive matmul kernel
// The math and loop structure are conceptually the same as the naive version.
// What changes is WHERE the data for the inner loop comes from: shared memory
// instead of global memory, which is the actual optimization.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// SHMEM_SIZE = the width/height of the square tile each block loads into shared memory.
// This must match the block dimensions you launch with (THREADS below) — each thread
// in the block is responsible for loading exactly one element into the shared tile.
const int SHMEM_SIZE = 1 << 5; // 32, matching THREADS = 32 in main()

__global__ void tiledMatrixMul(const int *a, const int *b, int *c, int N) {
  // __shared__ memory is on-chip, fast, and shared by every thread in the SAME block.
  // Unlike global memory (a and b), shared memory has to be manually loaded by the
  // threads themselves — the hardware doesn't do it automatically. I declare two
  // tiles here: one to hold a piece of matrix a, one to hold a piece of matrix b.
  __shared__ int s_a[SHMEM_SIZE][SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE][SHMEM_SIZE];

  // Same global row/col computation as the naive version — this thread still
  // "owns" exactly one output element, c[row][col].
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // This will accumulate the final dot-product result for c[row][col], same role
  // as the running total in the naive kernel's global-memory accumulation.
  int tmp = 0;

  // Instead of one loop over the full width N (like the naive version), I now loop
  // over TILES. Each iteration processes one SHMEM_SIZE-wide slice of the shared
  // dimension, rather than the whole thing in one pass. N / SHMEM_SIZE tells us how
  // many tiles are needed to cover the full row/column.
  for (int i = 0; i < N; i += SHMEM_SIZE) {

    // Every thread in the block cooperates to fill the shared tiles — each thread
    // loads exactly ONE element into s_a and ONE element into s_b, from global memory.
    // This is the key idea: instead of every thread independently re-reading the same
    // row of `a` or column of `b` from slow global memory, the block reads each needed
    // value ONCE, cooperatively, and stores it somewhere fast that everyone can reuse.
    s_a[threadIdx.y][threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y][threadIdx.x] = b[i * N + threadIdx.y * N + col];

    // __syncthreads() is a barrier: every thread in the block must reach this line
    // before any thread is allowed to proceed past it. This is essential here —
    // without it, some threads could start reading s_a/s_b before other threads have
    // finished writing their piece of the tile, producing a race condition and wrong
    // results. This is the "cooperative" part of cooperative loading.
    __syncthreads();

    // Now every thread in the block computes its partial dot-product contribution
    // for THIS tile, reading from fast shared memory instead of slow global memory.
    // Same accumulation logic as the naive kernel's inner loop, just operating on a
    // SHMEM_SIZE-sized chunk at a time instead of the whole row/column at once.
    for (int j = 0; j < SHMEM_SIZE; j++) {
      tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
    }

    // A second __syncthreads() barrier: this ensures every thread has finished
    // reading from the CURRENT shared tiles before any thread starts overwriting
    // them with the NEXT tile's data at the top of the next loop iteration.
    // Without this, a fast thread could start loading new data into s_a/s_b while
    // a slower thread is still reading the old values — another race condition.
    __syncthreads();
  }

  // After looping through all tiles, tmp holds the complete dot-product result —
  // same final value the naive kernel would compute, just built up more efficiently.
  c[row * N + col] = tmp;
}

// Naive version kept for side-by-side profiling comparison
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

// Check result on the CPU — unchanged from the naive version, since correctness
// verification doesn't care HOW the GPU computed the answer, only whether it's right.
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  int N = 1 << 10; // 1024 x 1024

  size_t bytes = N * N * sizeof(int);

  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // THREADS must equal SHMEM_SIZE — each thread loads exactly one shared-memory
  // element, so the block dimensions and the tile dimensions have to match.
  int THREADS = 32;
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Swap this line to matrixMul<<<...>>> to run the naive version for comparison —
  // same launch configuration works for both, since the thread/block layout is
  // identical between the two kernels.
  tiledMatrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
