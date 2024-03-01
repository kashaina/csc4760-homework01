#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
using namespace std;

// Declare a 5 ∗ 7 ∗ 12 ∗ n View

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Declare const sizes
    const int a = 5;
    const int b = 7;
    const int c = 12;

    // Ask the user for input for n
    int n;
    printf("Enter the value of n: ");
    scanf("%d", &n);

    // Check if n is a valid value
    if (n <= 0) {
      printf("Value of n invalid. Terminating.\n\n");
      return 1;
    }

    // Make view
    Kokkos::View<double****> myView("myView", a, b, c, n);

    // Check dimensions (for testing purposes)
    cout << "\nView Dimensions: " << myView.extent(0) << " x " << myView.extent(1) << " x " << myView.extent(2) << " x " << myView.extent(3) << endl;
    cout << "View Size: " << myView.size() << endl << endl;
  
  }
  Kokkos::finalize();
}
