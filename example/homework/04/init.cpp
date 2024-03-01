#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
using namespace std;

// Do simple parallel reduce to output the maximum element in a View

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

  int n = 5; // set number of rows
  int m = 5; // set number of columns
 
  std::srand(std::time(0)); // set seed for random values
  
  // Make View and create values
  Kokkos::View<int**> myView("myView", n, m);
  Kokkos::parallel_for("Loop1", myView.extent(0), KOKKOS_LAMBDA (const int i) {
    Kokkos::parallel_for("Loop2", myView.extent(1), KOKKOS_LAMBDA (const int j) {
      myView(i, j) = rand() % 500; // fill using random number between 0 and 499
    });
  });

   // Print out the view (for testing purposes)
   cout << "View:" << endl;
   for (int i = 0; i < n; ++i) {
     for (int j = 0; j < m; ++j) {
       cout << setw(3) << myView(i, j) << " ";
     }
     cout << "\n";
   }

   // Do a parallel reduction to find the maximum element
   int result = 0;
   Kokkos::parallel_reduce("parallel max", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}), KOKKOS_LAMBDA(const int i, const int j, int& lmax) {
       if (myView(i, j) > lmax) {
         lmax = myView(i, j);
        }
      }, Kokkos::Max<int>(result));
   Kokkos::fence();

   // Print the result
   cout << "\nMax Value: " << result << endl << endl;

  }
  Kokkos::finalize();
}
