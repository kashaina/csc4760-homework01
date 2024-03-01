#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
// Create a program that does matrix addition between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions

// Function that performs matrix-vector multiplication and checks for correct shape and dimensions
template <typename MatrixViewType, typename VectorViewType, typename ResultViewType>
void matrixMultiplication(const MatrixViewType& matrix, const VectorViewType& vector, ResultViewType& result) {
  
  // Check if each view has the correct number of dimensions
  // This function works in theory but does not actually due to compiler checks
  if (matrix.rank() != 2) {
    std::cerr << "\nError: Matrix must be a 2D view." << std::endl << std::endl;
    return;
  }
  if (vector.rank() != 1) {
    std::cerr << "\nError: Vector must be a 1D view." << std::endl << std::endl;
    return;
  }
  if (result.rank() != 1) {
    std::cerr << "\nError: Result must be a 1D view." << std::endl << std::endl;
    return;
  }

  // Extract number of indices of each dimension of each View
  const size_t matrixRows = matrix.extent(0);
  const size_t matrixCols = matrix.extent(1);
  const size_t vectorSize = vector.extent(0);
  const size_t resultSize = result.extent(0);
  
  // Check if the views' sizes are compatible
  if (matrixCols != vectorSize) {
    std::cerr << "\nError: Matrix columns (" << matrixCols << ") must be equal to vector size (" << vectorSize << ")." << std::endl << std::endl;
    return;
  }
  if (matrixRows != resultSize) {
    std::cerr << "\nError: Result matrix dimensions (" << matrixRows << " x 1 are not compatible." << std::endl << std::endl;
    return;
  }

  // Loop to perform matrix-vector multiplication
  Kokkos::parallel_for("total matrix multiplication", matrixRows, KOKKOS_LAMBDA(const int i) {
    int row_sum = 0;
    for (size_t j = 0; j < matrixCols; j++) {
      assert(matrix.extent(1) == vector.extent(0));
      row_sum += matrix(i, j) * vector(j);
    }
    result(i) = row_sum;
  });
  Kokkos::fence();
}


// Main to define Views and call function
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  int n = 3;
  int m = 3;
  
  // Make View and add values
  Kokkos::View<int**> matrix("matrix", n, m);
  Kokkos::View<int*> vector("vector", m);
  Kokkos::View<int*> result("result", n);
  
  // Initialize matrix with test values
  matrix(0, 0) = 130; matrix(0, 1) = 137; matrix(0, 2) = 115;
  matrix(1, 0) = 224; matrix(1, 1) = 158; matrix(1, 2) = 187;
  matrix(2, 0) = 54;  matrix(2, 1) = 211; matrix(2, 2) = 120;
  
  // Initialize vector with test values
  vector(0) = 221;
  vector(1) = 12;
  vector(2) = 157;
  
  // Do matrix addition using the function
  matrixMultiplication(matrix, vector, result);
  
  // Output all views, including result 
  std::cout << "Matrix:\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      std::cout << matrix(i, j) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nVector:\n";
  for (int j = 0; j < n; j++) {
    std::cout << vector(j) << " ";
  }
  std::cout << "\n\n";
  
  std::cout << "Result:\n";
  for (int i = 0; i < n; i++) {
    std::cout << result(i) << " ";
  }

  // Additional view to show that function needs the correct dimensions/sizes
  Kokkos::View<int*> vector2("vector2", 2);
  std::cout << "\n\n\nWe are now going to demonstrate that the function requires the correct dimensions/sizes.";
  std::cout << "\nWe are going to attempt to multiply a 3 x 3 matrix with a size 2 vector.";
  matrixMultiplication(matrix, vector2, result);
  
  }
  Kokkos::finalize();
}
