#include <Kokkos_Core.hpp>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip>

void createAndFindPrefixSum(int n);

// Define size and call createAndFindPrefixSum
int main() {
   Kokkos::initialize();
   {
   int n = 10;

   std::srand(std::time(0));

   createAndFindPrefixSum(n);
   createAndFindPrefixSum(n);
   createAndFindPrefixSum(n);
    
   std::cout << "As you can see, the times are not always the same but are each very low\n\n";  
   }

   Kokkos::finalize();
}

// Creates a 1d view and outputs its partial sums
void createAndFindPrefixSum(int n) {
    Kokkos::View<int*> original("original", n);
    Kokkos::View<int*> partialSums("partial_sums", n);
    
    // Populate view
    for (int i = 0; i < n; ++i) {
        original(i) = rand() % 10; // fill using random number between 0 and 9
    }

    // Calculate prefix sum
    Kokkos::Timer timer;
    Kokkos::parallel_scan("prefix_sum", n, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        if (final) {
            partialSums(i) = update + original(i);
        }
        update += original(i);
    });
    double total_time = timer.seconds();
    Kokkos::fence();

    // Output both arrays and elapsed time
    std::cout << "Original arrray:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(2) << original(i) << " ";
    }
    std::cout << "\nPartial Sums:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(2) << partialSums(i) << " ";
    }
    std::cout << "\nElapsed Time: " << total_time << " seconds\n\n";
}
