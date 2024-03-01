#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
using namespace std;

// Problem: Make an n ∗ m View where each index equals 1000 ∗ i ∗ j

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Set n and m, you can change these values
  //int n,m = 16;  //this provided code does not work
  int n = 16;
  int m = 16;
  
  // Make View
  Kokkos::View<int**> myView("myView", n, m);

  // Loop 1: serial method
  // set values to 1000 * i * j;
  //for (int i = 0; i < n; i++){
  //  for (int j = 0; j < m; j++){
  //    myView(i, j) = 1000 * i * j;
  //  }
  //}

  // Loop 2 (better): parallel method
  Kokkos::parallel_for("Loop1", myView.extent(0), KOKKOS_LAMBDA (const int i) {
    Kokkos::parallel_for("Loop2", myView.extent(1), KOKKOS_LAMBDA (const int j) {
      myView(i, j) = 1000 * i * j;
    });
  });
  Kokkos::fence(); 
  
  // Print view
  // I'm not sure if we are supposed to but it's good for testing
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      cout << setw(6) << myView(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;

  }
  Kokkos::finalize();
}
