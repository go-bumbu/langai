package onnx

import (
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// ortAPI implements sessionAPI over onnxruntime_go. This is the only file
// that imports the bindings: everything that decides something (load once,
// version lock, sizes, batching, cleanup) lives in backend.go and onnx.go,
// where a fake covers it, and what remains here are thin calls that need
// the native library to run.
type ortAPI struct{}

func (ortAPI) load(libPath string, _ *slog.Logger) error {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		return fmt.Errorf("load onnxruntime from %s: %w", libPath, err)
	}
	return nil
}

func (ortAPI) version() string { return ort.GetVersion() }

// open reads the model's tensor names and shapes, checks the ones the spec
// names, and creates the session with the provider and thread count asked
// for.
func (ortAPI) open(path string, opts sessionOptions) (session, int, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, 0, fmt.Errorf("read model %s: %w", path, err)
	}
	if _, err := find(inputs, opts.input, "input"); err != nil {
		return nil, 0, err
	}
	out, err := find(outputs, opts.output, "output")
	if err != nil {
		return nil, 0, err
	}
	dims := 0
	if n := len(out.Dimensions); n > 0 && out.Dimensions[n-1] > 0 {
		dims = int(out.Dimensions[n-1])
	}
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = so.Destroy() }()
	if err := so.SetIntraOpNumThreads(opts.threads); err != nil {
		return nil, 0, err
	}
	if err := appendProvider(so, opts); err != nil {
		return nil, 0, err
	}
	s, err := ort.NewDynamicAdvancedSession(path, []string{opts.input}, []string{opts.output}, so)
	if err != nil {
		return nil, 0, fmt.Errorf("create session for %s: %w", path, err)
	}
	return &ortSession{s: s}, dims, nil
}

// find returns the tensor named name among infos, or an error listing what
// the model has.
func find(infos []ort.InputOutputInfo, name, kind string) (ort.InputOutputInfo, error) {
	names := make([]string, len(infos))
	for i, info := range infos {
		if info.Name == name {
			return info, nil
		}
		names[i] = info.Name
	}
	return ort.InputOutputInfo{}, fmt.Errorf("the model has no %s tensor %q; it has %s", kind, name, strings.Join(names, ", "))
}

// appendProvider adds the execution provider opts asks for; CPU needs
// nothing, it is what a session falls back to.
func appendProvider(so *ort.SessionOptions, opts sessionOptions) error {
	switch opts.provider {
	case ProviderCUDA:
		co, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return fmt.Errorf("cuda provider: %w", err)
		}
		defer func() { _ = co.Destroy() }()
		if err := co.Update(map[string]string{"device_id": strconv.Itoa(opts.deviceID)}); err != nil {
			return fmt.Errorf("cuda provider: %w", err)
		}
		if err := so.AppendExecutionProviderCUDA(co); err != nil {
			return fmt.Errorf("cuda provider (is this the gpu build of onnxruntime, with CUDA 12 and cuDNN 9 installed?): %w", err)
		}
	case ProviderCoreML:
		if err := so.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			return fmt.Errorf("coreml provider (is this the macOS build of onnxruntime?): %w", err)
		}
	}
	return nil
}

// ortSession is a session over one DynamicAdvancedSession.
type ortSession struct {
	s *ort.DynamicAdvancedSession
}

// run lets the runtime allocate the output, whose shape is only known for
// sure once the model ran (a token sequence has a dynamic length), and
// copies it out.
func (s *ortSession) run(pixels []float32, n, side int) ([]float32, []int64, error) {
	if n <= 0 || len(pixels) != n*3*side*side {
		return nil, nil, fmt.Errorf("batch of %d values is not %d pictures of %dx%d", len(pixels), n, side, side)
	}
	in, err := ort.NewTensor(ort.NewShape(int64(n), 3, int64(side), int64(side)), pixels)
	if err != nil {
		return nil, nil, fmt.Errorf("input tensor: %w", err)
	}
	defer func() { _ = in.Destroy() }()
	outputs := []ort.Value{nil}
	if err := s.s.Run([]ort.Value{in}, outputs); err != nil {
		return nil, nil, fmt.Errorf("run: %w", err)
	}
	defer func() { _ = outputs[0].Destroy() }()
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, errors.New("the model's output is not a float32 tensor")
	}
	return append([]float32(nil), out.GetData()...), append([]int64(nil), out.GetShape()...), nil
}

func (s *ortSession) close() { _ = s.s.Destroy() }
