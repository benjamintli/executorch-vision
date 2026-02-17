#include "gtest/gtest.h"
#include "parakeet.h"

TEST(ParakeetTest, AddWorks) {
  EXPECT_EQ(parakeet::Add(1, 2), 3);
}
