#include "model.h"

#include <cstring>
#include <filesystem>

#include <third_party/executorch/extension/module/module.h>
#include <third_party/executorch/extension/tensor/tensor_ptr_maker.h>
#include <third_party/executorch/runtime/core/evalue.h>

namespace executorch_vision {

struct ModelForObjectDetection::Impl {
    std::unique_ptr<executorch::extension::Module> module;
    int image_size = 0;
    int num_channels = 0;
    std::unordered_map<int, std::string> id2label;
};

// Required in .cpp so unique_ptr<Impl> can see the complete type.
ModelForObjectDetection::~ModelForObjectDetection() = default;

int ModelForObjectDetection::image_size() const {
    return impl_->image_size;
}
int ModelForObjectDetection::num_channels() const {
    return impl_->num_channels;
}
const std::unordered_map<int, std::string>& ModelForObjectDetection::id2label() const {
    return impl_->id2label;
}

namespace {

executorch_vision::Tensor to_owned_tensor(const executorch::runtime::EValue& ev) {
    const auto& t = ev.toTensor();
    executorch_vision::Tensor out;
    out.shape = {t.sizes().begin(), t.sizes().end()};
    out.data.resize(t.numel());
    std::memcpy(out.data.data(), t.const_data_ptr<float>(), t.numel() * sizeof(float));
    return out;
}

}  // namespace

std::unique_ptr<ModelForObjectDetection> ModelForObjectDetection::from_pretrained(
    const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        return nullptr;
    }

    auto model = std::unique_ptr<ModelForObjectDetection>(new ModelForObjectDetection());
    model->impl_ = std::make_unique<Impl>();
    model->impl_->module = std::make_unique<executorch::extension::Module>(model_path);

    if (model->impl_->module->load_forward() != executorch::runtime::Error::Ok) {
        return nullptr;
    }

    auto names = model->impl_->module->method_names();
    if (!names.ok()) {
        return nullptr;
    }

    if (names->count("image_size")) {
        auto r = model->impl_->module->execute("image_size");
        if (r.ok() && !r->empty()) {
            model->impl_->image_size = static_cast<int>(r->at(0).toInt());
        }
    }

    if (names->count("num_channels")) {
        auto r = model->impl_->module->execute("num_channels");
        if (r.ok() && !r->empty()) {
            model->impl_->num_channels = static_cast<int>(r->at(0).toInt());
        }
    }

    if (names->count("get_label_ids") && names->count("get_label_names")) {
        auto ids_r = model->impl_->module->execute("get_label_ids");
        auto labels_r = model->impl_->module->execute("get_label_names");
        if (ids_r.ok() && labels_r.ok() && ids_r->size() == labels_r->size()) {
            for (size_t i = 0; i < ids_r->size(); ++i) {
                model->impl_->id2label[static_cast<int>(ids_r->at(i).toInt())] =
                    std::string(labels_r->at(i).toString());
            }
        }
    }

    return model;
}

ObjectDetectionOutput ModelForObjectDetection::forward(const Tensor& pixel_values) {
    // Wrap input as a no-copy view for the duration of this call.
    auto input =
        executorch::extension::from_blob(const_cast<float*>(pixel_values.data.data()),
                                         std::vector<executorch::aten::SizesType>(
                                             pixel_values.shape.begin(), pixel_values.shape.end()),
                                         executorch::aten::ScalarType::Float);

    auto result = impl_->module->forward(executorch::runtime::EValue(*input));
    if (!result.ok() || result->size() < 2) {
        return {};
    }

    return {to_owned_tensor((*result)[0]), to_owned_tensor((*result)[1])};
}

}  // namespace executorch_vision
