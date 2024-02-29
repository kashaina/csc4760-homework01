#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
using namespace std;

// Problem: Link and run program with Kokkos where you initialize a View and print out its name with the $.label()$ method.

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View
  Kokkos::View<double*[3]> myView ("myView", 8);
	  
  // print name
  cout << "View name: " << myView.label() << endl << endl;
  
  }
  Kokkos::finalize();
  return 0;
}
