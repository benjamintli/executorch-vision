#include "gtest/gtest.h"
#include "executorch_vision.h"

TEST(ExecutorchVisionTest, AddWorks) {
  EXPECT_EQ(executorch_vision::Add(1, 2), 3);
}
