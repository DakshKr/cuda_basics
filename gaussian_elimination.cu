#include <iostream>


std::pair<float*, float*> take_input(int n, int m);
void allocate_and_copy(const float* h_A, const float* h_B, int n, int m, 
                        float** d_A, size_t* pitch_A, float** d_B);
void execute_kernel(float* d_A, size_t pitch_A, float* d_B, int n, int m);

__global__ void forward_elimination(float* d_A, size_t pitch_A, float* d_B, int n, int m, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = k + 1 + idx;

    if (i < n) {
        float* row_k = (float*)((char*)d_A + k * pitch_A);
        float* row_i = (float*)((char*)d_A + i * pitch_A);

        float factor = row_i[k] / row_k[k];
        for (int j = k; j < m; ++j) {
            row_i[j] -= factor * row_k[j];
        }
        d_B[i] -= factor * d_B[k];
    }
}

int main() {
    int n, m;
    std::cout << "Enter the dimensions of the matrix (n m): ";
    std::cin >> n >> m;

    auto [h_A, h_B] = take_input(n, m);
    float* d_A = nullptr;
    float* d_B = nullptr;
    size_t pitch_A = 0;

    allocate_and_copy(h_A, h_B, n, m, &d_A, &pitch_A, &d_B);
    execute_kernel(d_A, pitch_A, d_B, n, m);

    // Copy triangular matrix and vector back to host
    cudaMemcpy2D(h_A, m * sizeof(float), d_A, pitch_A, m * sizeof(float), n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Back-substitution on host
    float* x = new float[n];
    for (int i = n - 1; i >= 0; --i) {
        float sum = 0.0f;
        for (int j = i + 1; j < m; ++j) {
            sum += h_A[i * m + j] * x[j];
        }
        x[i] = (h_B[i] - sum) / h_A[i * m + i];
    }

    // Print solution
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";

    // Free resources
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
    delete[] x;

    return 0;
}

void allocate_and_copy(const float* h_A, const float* h_B, int n, int m, 
                        float** d_A, size_t* pitch_A, float** d_B){

    cudaMallocPitch((void**)d_A, pitch_A, m * sizeof(float), n);

    cudaMalloc((void**)d_B, n * sizeof(float));

    cudaMemcpy2D(*d_A, *pitch_A, 
                 h_A, m * sizeof(float), 
                 m * sizeof(float), n, 
                 cudaMemcpyHostToDevice);

    cudaMemcpy(*d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);
}

void execute_kernel(float* d_A, size_t pitch_A, float* d_B, int n, int m) {
    for (int k = 0; k < n - 1; ++k) {
        int below_size = n - (k + 1);
        int threads = 256;
        int blocks = (below_size + threads - 1) / threads;
        forward_elimination<<<blocks, threads>>>(d_A, pitch_A, d_B, n, m, k);
    }
    cudaDeviceSynchronize();
}

// take 2d matrix A and vecor B for Ax = B
std::pair<float*, float*> take_input(int n, int m){

  int t_elements = n*m;
  float* A = new float[t_elements];
  std::cout << "Enter the elements of the matrix:" << std::endl;
  for (int i = 0; i<t_elements; i++){
      std::cin >> A[i];
  }
  

  float* B = new float[n];
  std::cout << "Enter the elements of the vector B:" << std::endl;
  for (int i = 0; i<n; i++){
    std::cin >> B[i];
  }

  return std::make_pair(A, B);
}



