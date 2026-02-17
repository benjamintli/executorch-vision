// Minimal binary for the executorch-vision library.
#include <iostream>
#include "executorch_vision.h"

int main() {
  std::cout << "executorch-vision: " << executorch_vision::Add(2, 3) << '\n';
  return 0;
}
