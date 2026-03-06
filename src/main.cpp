// Minimal binary for the executorch-vision library.
#include <iostream>
#include "model.h"

int main() {
    auto model = executorch_vision::ModelForObjectDetection::from_pretrained("data/model.pte");
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    std::cout << "image_size:   " << model->image_size() << "\n";
    std::cout << "num_channels: " << model->num_channels() << "\n";
    std::cout << "num_labels:   " << model->id2label().size() << "\n";

    // Dummy input: [1, C, H, W] filled with 0.5
    int C = model->num_channels();
    int S = model->image_size();
    executorch_vision::Tensor input;
    input.shape = {1, C, S, S};
    input.data.assign(C * S * S, 0.5f);

    auto output = model->forward(input);

    std::cout << "logits shape:    [";
    for (int i = 0; i < (int)output.logits.shape.size(); ++i) {
        std::cout << output.logits.shape[i];
        if (i + 1 < (int)output.logits.shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "pred_boxes shape: [";
    for (int i = 0; i < (int)output.pred_boxes.shape.size(); ++i) {
        std::cout << output.pred_boxes.shape[i];
        if (i + 1 < (int)output.pred_boxes.shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Print the top logit value from the first query as a sanity check
    if (!output.logits.data.empty()) {
        std::cout << "logits[0][0][0]: " << output.logits.data[0] << "\n";
    }

    return 0;
}
