package onnx

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestSpec_PinnedModels(t *testing.T) {
	if got := Models(); strings.Join(got, ",") != DINOv2Model+","+DefaultModel {
		t.Errorf("Models() = %v", got)
	}
	for _, name := range Models() {
		spec, err := Spec(name, "data/models")
		if err != nil {
			t.Fatalf("Spec(%s): %v", name, err)
		}
		if spec.Name != name || filepath.Dir(spec.File) != filepath.Clean("data/models") {
			t.Errorf("Spec(%s) = %+v", name, spec)
		}
		if spec.URL == "" || len(spec.SHA256) != 64 || spec.Size == 0 || spec.Dims != 768 || spec.Side != 224 {
			t.Errorf("Spec(%s) is not fully pinned: %+v", name, spec)
		}
		if _, err := spec.normalized(); err != nil {
			t.Errorf("Spec(%s) does not validate: %v", name, err)
		}
	}
	def := DefaultSpec("")
	if def.Name != DefaultModel || def.File != filepath.Join("models", defaultModelFile) {
		t.Errorf("DefaultSpec = %+v", def)
	}
	if spec, err := Spec("", ""); err != nil || spec != def {
		t.Errorf("Spec(\"\", \"\") = %+v, %v; want the default", spec, err)
	}
	_, err := Spec("nobody/nothing", "")
	if err == nil || !strings.Contains(err.Error(), DefaultModel) {
		t.Errorf("unknown model: err = %v, want the pinned names listed", err)
	}
}

func TestModelSpec_Normalized(t *testing.T) {
	good := ModelSpec{Name: "m", File: "m.onnx", Side: 224}
	spec, err := good.normalized()
	if err != nil {
		t.Fatal(err)
	}
	if spec.Input != "pixel_values" || spec.Output != "pooler_output" || spec.Pooling != PoolingNone || spec.Resize != ResizeSquash ||
		spec.Interpolation != InterpolationBilinear || spec.ShortestEdge != 224 || spec.Std != [3]float32{1, 1, 1} {
		t.Errorf("defaults not filled: %+v", spec)
	}
	bad := map[string]ModelSpec{
		"name":          {File: "m.onnx", Side: 224},
		"file":          {Name: "m", Side: 224},
		"url no digest": {Name: "m", File: "m.onnx", Side: 224, URL: "https://x/m.onnx"},
		"side":          {Name: "m", File: "m.onnx"},
		"resize":        {Name: "m", File: "m.onnx", Side: 224, Resize: "pad"},
		"shortest edge": {Name: "m", File: "m.onnx", Side: 224, Resize: ResizeCrop, ShortestEdge: 200},
		"interpolation": {Name: "m", File: "m.onnx", Side: 224, Interpolation: "lanczos"},
		"pooling":       {Name: "m", File: "m.onnx", Side: 224, Pooling: "max"},
		"dims":          {Name: "m", File: "m.onnx", Side: 224, Dims: -1},
	}
	for name, s := range bad {
		if _, err := s.normalized(); err == nil {
			t.Errorf("%s: %+v validated", name, s)
		}
	}
}

func TestProvider_Normalized(t *testing.T) {
	if p, err := Provider("").normalized(); err != nil || p != ProviderCPU {
		t.Errorf("empty provider = %q, %v", p, err)
	}
	for _, p := range providers {
		if got, err := p.normalized(); err != nil || got != p {
			t.Errorf("%s: %q, %v", p, got, err)
		}
	}
	if _, err := Provider("tpu").normalized(); err == nil || !strings.Contains(err.Error(), "cuda") {
		t.Errorf("unknown provider: err = %v", err)
	}
}
