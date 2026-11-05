package onnx

import (
	"fmt"
	"path/filepath"
	"sort"
)

// Pinned models. The files are the Optimum exports onnx-community publishes
// on Hugging Face; the digests and sizes are the LFS metadata Hugging Face
// reports for them, and the picture preparation is each model's
// preprocessor_config.json.
const (
	// DINOv2Model is facebook/dinov2-base: a self-supervised image encoder,
	// 768 dimensions, no text tower. It ranks by what a picture is of and
	// how it is composed, so it suits instance and near-duplicate matching;
	// Apache-2.0.
	DINOv2Model = "facebook/dinov2-base"

	defaultModelFile = "siglip2-base-patch16-224-vision.onnx"
	dinov2ModelFile  = "dinov2-base.onnx"
)

// pinned maps a model name to its spec with File relative to the models
// directory; Spec fills the directory in.
var pinned = map[string]ModelSpec{
	// the vision tower only, fp32; Apache-2.0
	DefaultModel: {
		Name:   DefaultModel,
		File:   defaultModelFile,
		URL:    "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/vision_model.onnx",
		SHA256: "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad",
		Size:   371807752,
		Side:   224,
		Resize: ResizeSquash,
		Mean:   [3]float32{0.5, 0.5, 0.5},
		Std:    [3]float32{0.5, 0.5, 0.5},
		Dims:   768,
	},
	// fp32; Apache-2.0. The export has no pooler output: the vector is the
	// class token of the last hidden state, as in the model's own code.
	DINOv2Model: {
		Name:          DINOv2Model,
		File:          dinov2ModelFile,
		URL:           "https://huggingface.co/onnx-community/dinov2-base-ONNX/resolve/main/onnx/model.onnx",
		SHA256:        "f16115e628d65b7cc7b1e16c504e2af682169aabf3fff4edfe906118f522e204",
		Size:          346608174,
		Output:        "last_hidden_state",
		Pooling:       PoolingCLS,
		Side:          224,
		Resize:        ResizeCrop,
		ShortestEdge:  256,
		Interpolation: InterpolationBicubic,
		Mean:          [3]float32{0.485, 0.456, 0.406},
		Std:           [3]float32{0.229, 0.224, 0.225},
		Dims:          768,
	},
}

// Models lists the pinned model names, sorted.
func Models() []string {
	names := make([]string, 0, len(pinned))
	for name := range pinned {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Spec returns the pinned spec of the named model ("" means DefaultModel)
// with its file placed inside modelsDir ("" means "models"), or an error
// listing the pinned names.
func Spec(name, modelsDir string) (ModelSpec, error) {
	if name == "" {
		name = DefaultModel
	}
	if modelsDir == "" {
		modelsDir = "models"
	}
	spec, ok := pinned[name]
	if !ok {
		return ModelSpec{}, fmt.Errorf("onnx: unknown model %q: pinned models are %v; use Open with a ModelSpec for another", name, Models())
	}
	spec.File = filepath.Join(modelsDir, spec.File)
	return spec, nil
}
