
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
using namespace std;

// Create a program that does matrix addition between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions

// Matrix addition function that also checks for correct shape and dimensions
template <typename MatrixViewType, typename VectorViewType, typename ResultViewType>
void matrixAddition(const MatrixViewType& matrix, const VectorViewType& vector, ResultViewType& result) {
  // Check if each view has the correct number of dimensions
  // This function works in theory but does not actually due to compiler checks
  if (matrix.rank() != 2) {
    cerr << "\nError: Matrix must be a 2D view." << endl << endl;
    exit(EXIT_FAILURE);
  }
  if (vector.rank() != 1) {
    cerr << "\nError: Vector must be a 1D view." << endl << endl;
    exit(EXIT_FAILURE);
  }
  if (result.rank() != 2) {
    cerr << "\nError: Result must be a 2D view." << endl << endl;
    exit(EXIT_FAILURE);
  }

  // Extract number of indices of each dimension of each View
  const size_t matrixRows = matrix.extent(0);
  const size_t matrixCols = matrix.extent(1);
  const size_t vectorSize = vector.extent(0);
  const size_t resultSize = result.extent(0);
  
  // Check if matrix and vector can be multiplied
  if (matrixCols != vectorSize) {
      cerr << "\nError: Matrix columns (" << matrixCols << ") must be equal to vector size (" << vectorSize << ")." << endl << endl;
      exit(EXIT_FAILURE);
  }
  // Check if the result matrix has compatible dimensions
  if (matrixRows != resultSize) {
    cerr << "\nError: Result matrix dimensions (" << matrixRows << " x 1 are not compatible." << endl << endl;
    exit(EXIT_FAILURE);
  }
  // Loop to perform matrix-vector addition
  Kokkos::parallel_for("total matrix addition", matrixRows, KOKKOS_LAMBDA(const int i) {
    for (size_t j = 0; j < matrixCols; j++) {
      assert(matrix.extent(1) == vector.extent(0));
      result(i, j) += matrix(i, j) + vector(j);
    }
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
  Kokkos::View<int**> result("result", n, m);
  
  // Initialize matrix with test values
  matrix(0, 0) = 130; matrix(0, 1) = 147; matrix(0, 2) = 115;
  matrix(1, 0) = 224; matrix(1, 1) = 158; matrix(1, 2) = 187;
  matrix(2, 0) = 54;  matrix(2, 1) = 158; matrix(2, 2) = 120;
  
  // Initialize vector with test values
  vector(0) = 221;
  vector(1) = 12;
  vector(2) = 157;
  
  // Do matrix addition using the function
  matrixAddition(matrix, vector, result);
  
  // Output matrix, vector, and result 
  cout << "Matrix:\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      cout << setw(3) << matrix(i, j) << " ";
    }
    cout << "\n";
  }

  cout << "\nVector:\n";
  for (int j = 0; j < n; j++) {
    cout << vector(j) << " ";
  }

  cout << "\n\nAddition result:\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      cout << result(i, j) << " ";
    }
    cout << endl;
  }

  // Additional view to show that function needs the correct dimensions/sizes
  Kokkos::View<int*> vector2("vector2", 2);
  cout << "\n\nWe are now going to demonstrate that the function requires the correct dimensions/sizes.";
  cout << "\nWe are going to attempt to multiply a 3 x 3 matrix with a size 2 vector.";
  matrixAddition(matrix, vector2, result);
  cout << endl << endl << endl;
  }

  Kokkos::finalize();
}



