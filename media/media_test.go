package media

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSamplerFunc_CallsTheFunction(t *testing.T) {
	want := errors.New("boom")
	var gotPath string
	s := SamplerFunc(func(_ context.Context, path string) ([]Frame, error) {
		gotPath = path
		return []Frame{{At: 3 * time.Second}}, want
	})
	frames, err := s.Sample(context.Background(), "clip.mp4")
	if gotPath != "clip.mp4" {
		t.Errorf("path = %q, want clip.mp4", gotPath)
	}
	if !errors.Is(err, want) {
		t.Errorf("err = %v, want %v", err, want)
	}
	if len(frames) != 1 || frames[0].At != 3*time.Second {
		t.Errorf("frames = %v, want the one at 3s", frames)
	}
}
