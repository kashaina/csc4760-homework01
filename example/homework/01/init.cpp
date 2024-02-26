#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

// Problem: Link and run program with Kokkos where you initialize a View and print out its name with the $.label()$ method.

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View
  Kokkos::View<double*[3]> myView ("myView", 8);
	  
  // print name
  std::cout << "View name: " << myView.label() << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
