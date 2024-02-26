#include <Kokkos_Core.hpp>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip>

void createAndFindPrefixSum(int rows, int cols);

// Define rows and cols and call createAndFindPrefixSum
int main() {
    Kokkos::initialize();
    {
        std::srand(std::time(0));
        int rows = 3;
        int cols = 4;
        createAndFindPrefixSum(rows, cols);
        createAndFindPrefixSum(rows, cols);
        createAndFindPrefixSum(rows, cols);
    }
    Kokkos::finalize();
}

// Create and fill an original view and find and print its partial sums and time
void createAndFindPrefixSum(int rows, int cols) {
    Kokkos::View<int**> original("original", rows, cols);
    Kokkos::View<int**> partialSums("partial_sums", rows, cols);

    // make original view
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            original(i, j) = rand() % 10; // fill using random number between 0 and 9
        }
    }

    Kokkos::Timer timer;

    // find partial sums
    Kokkos::parallel_for("prefix_sum", rows, KOKKOS_LAMBDA(const int i) {
        int row_sum = 0;
        for (int j = 0; j < cols; ++j) {
            row_sum += original(i, j);
            partialSums(i, j) = row_sum;
        }
    });

    double total_time = timer.seconds();

    // print original
    std::cout << "Original array:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(2) << original(i, j) << " ";
        }
        std::cout << "\n";
    }

    // print partial sum
    std::cout << "\nPartial Sums array:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(2) << partialSums(i, j) << " ";
        }
        std::cout << "\n";
    }

    // print time
    std::cout << "\nElapsed Time: " << total_time << " seconds\n\n";
    std::cout << "-------------------------------------------------\n\n";
}

