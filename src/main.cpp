// Minimal binary for the parakeet library.
#include <iostream>
#include "parakeet.h"

int main() {
  std::cout << "parakeet: " << parakeet::Add(2, 3) << '\n';
  return 0;
}
