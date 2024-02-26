#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

// Create a program that compares a parallel for loop and a standard for loop for summing rows of a View with Kokkos Timer.
void printResults(const Kokkos::View<int*> rowSums, const double elapsedTime, const std::string& label) {
  std::cout << label << ":\n";
	
  // Print row sums
  //std::cout << "Row Sums:\n";
  //for (int i = 0; i < rowSums.size(); ++i) {
  //  std::cout << rowSums[i] << " ";
  //}
  //std::cout << std::endl;

  // Print timing information
  std::cout << "Time: "  << elapsedTime << " second\n\n";
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

  int n = 2000; // set number of rows
  int m = 5000; // set number of columns
  std::cout << "View size: " << n << " x " << m << std::endl << std::endl;

  std::srand(std::time(0)); // set seed for random values
  
  // Make View and create values
  Kokkos::View<int**> myView("myView", n, m);
  Kokkos::parallel_for("Loop1", myView.extent(0), KOKKOS_LAMBDA (const int i) { 
    Kokkos::parallel_for("Loop2", myView.extent(1), KOKKOS_LAMBDA (const int j) {
      myView(i, j) = rand() % 10; // fill using random number between 0 and 9
    });
  });

  // Print out the 2D matrix (for testing purposes)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      //std::cout << std::setw(3) << myView(i, j) << " ";
    }
    //std::cout << "\n\n";
  }
  Kokkos::Timer std_timer;
Kokkos::View<int*> std_rowSums("std_RowSums", n);

for (int i = 0; i < n; i++) {
  int rowSum = 0;
  for (int j = 0; j < m; j++) {
    rowSum += myView(i, j);
  }
  std_rowSums(i) = rowSum;
}

double std_totalTime = std_timer.seconds();
printResults(std_rowSums, std_totalTime, "Serial"); // print serial results

  // Parallel for loop to sum rows
  Kokkos::Timer par_timer; 
  Kokkos::View<int*> par_rowSums("par_RowSums", n);
  Kokkos::parallel_for("parallel row sums", n, KOKKOS_LAMBDA(const int i) {
    int rowSum = 0;
    for (int j = 0; j < m; j++) {
       rowSum += myView(i, j);
    }
    par_rowSums[i] = rowSum;
  });
  Kokkos::fence();
  double par_totalTime = par_timer.seconds();

  printResults(par_rowSums, par_totalTime, "Parallel"); // print parallel results

  // Compare the total time
  double speedup = std_totalTime / par_totalTime;
  std::cout << "The parallel loop is " << speedup << " times faster than the standard loop";

  // Assert both row sums are equal
  bool same = true;
  for (int i = 0 ; i < n; i++){
    if (std_rowSums(i) != par_rowSums(i)){
      same = false;
    }
  }
  if (same == true){
    std::cout << "\n\nBoth serial and parallel row sums are equivalent\n\n";
  } else {
    std::cout << "\n\nVALUE ERROR: Both serial and parallel row sums are NOT equivalent\n\n";
  }

  }
  Kokkos::finalize();
}
