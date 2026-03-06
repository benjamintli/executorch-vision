// model.h
// Public header for the executorch-vision library.

#ifndef MODEL_H_
#define MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch_vision {

struct Tensor {
    std::vector<int32_t> shape;
    std::vector<float> data;
};

struct ObjectDetectionOutput {
    Tensor logits;      // [batch, num_queries, num_classes]
    Tensor pred_boxes;  // [batch, num_queries, 4]
};

class ModelForObjectDetection {
public:
    static std::unique_ptr<ModelForObjectDetection> from_pretrained(
        const std::string& model_path);

    ObjectDetectionOutput forward(const Tensor& pixel_values);

    int image_size() const;
    int num_channels() const;
    const std::unordered_map<int, std::string>& id2label() const;

    ~ModelForObjectDetection();  // defined in .cpp where Impl is complete

private:
    ModelForObjectDetection() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace executorch_vision

#endif  // MODEL_H_
