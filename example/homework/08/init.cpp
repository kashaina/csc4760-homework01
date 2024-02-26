#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
// Create a program that does matrix addition between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions
// Extra credit function that also checks for correct shape and dimensions
template <typename MatrixViewType, typename VectorViewType, typename ResultViewType>
void matrixAddition(const MatrixViewType& matrix, const VectorViewType& vector, ResultViewType& result) {
  bool comp = true;
  // Check if each view has the correct number of dimensions
  if (matrix.rank() != 2) {
    std::cerr << "\nError: Matrix must be a 2D view." << std::endl;
    exit(EXIT_FAILURE);
    comp = false;
    Kokkos::abort("\nError: Matrix columns must be equal to vector size.\n");
  }
  if (vector.rank() != 1) {
    std::cerr << "\nError: Vector must be a 1D view." << std::endl;
    exit(EXIT_FAILURE);
    comp = false;
    Kokkos::abort("\nError: Matrix columns must be equal to vector size.\n");
  }
  if (result.rank() != 1) {
    std::cerr << "\nError: Result must be a 1D view." << std::endl;
    exit(EXIT_FAILURE);
    comp = false;
    Kokkos::abort("\nError: Matrix columns must be equal to vector size.\n");
  }
  // Extract number of indices of each dimension of each View
  const size_t matrixRows = matrix.extent(0);
  const size_t matrixCols = matrix.extent(1);
  const size_t vectorSize = vector.extent(0);
  const size_t resultSize = result.extent(0);
  // Check if matrix and vector can be multiplied
  if (matrixCols != vectorSize) {
      std::cerr << "\nError: Matrix columns (" << matrixCols << ") must be equal to vector size (" << vectorSize << ")." << std::endl;
      exit(EXIT_FAILURE);
      comp = false;
    Kokkos::abort("\nError: Matrix columns must be equal to vector size.\n");
  }
  // Check if the result matrix has compatible dimensions
  if (matrixRows != resultSize) {
    std::cerr << "\nError: Result matrix dimensions (" << matrixRows << " x 1 are not compatible." << std::endl;
    exit(EXIT_FAILURE);
    comp = false;
    Kokkos::abort("\nError: Matrix columns must be equal to vector size.\n");
  }
  // Loop to perform matrix-vector addition
  if (comp = true){
  Kokkos::parallel_for("total matrix addition", matrixRows, KOKKOS_LAMBDA(const int i) {
    int row_sum = 0;
    for (size_t j = 0; j < matrixCols; j++) {
      assert(matrix.extent(1) == vector.extent(0));
      row_sum += matrix(i, j) * vector(j);
    }
    result(i) = row_sum;
  });
  Kokkos::fence();
}}
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
  matrixAddition(matrix, vector, result);
  // Output addition 
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
  // Additional views to show that function needs the correct dimensions/sizes
  //std::cout << "\n\n\nWe are now going to return errors to show the function needs the correct dimensions/sizes";
  //Kokkos::View<int*> vector2("vector2", 2); 
  //std::cout << "\n\nThis vector has the wrong size";
  //matrixAddition(matrix, vector2, result);
  //Kokkos::View<int***> matrix2("matrix2", 3, 3, 3);
  //Kokkos::View<int*> matrix3("matrix3", 3);
  //std::cout << "\n\nThis matrix has the wrong dimensions\n";
  //matrixAddition(matrix2, vector, result);
  std::cout << "\n\n";
  }
  Kokkos::finalize();
}
