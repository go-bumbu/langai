package main

import (
	"fmt"
	"strings"

	"github.com/go-bumbu/sense/provider/onnx"
	"github.com/go-bumbu/sense/provider/yzma"
	"github.com/go-bumbu/sense/zarf/examples/duplicates"
	"github.com/go-bumbu/sense/zarf/examples/status"
)

// The overview's model choices: for every feature, each model it needs and
// every model that can fill that role, read from the providers' pinned
// catalogues (onnx.Models, yzma.VisionModels, yzma.Models,
// yzma.CompleterModels), so the names, dimensions and download sizes are the
// pins' own. Only the words on what a model is good at are written here.

const switchNote = "each model keeps an index of its own, so switching back finds it as it was"

// mediaChoice lists the models the media search page can run: the ONNX
// image models and the vision models, any of them picked with -model.
func mediaChoice(o options, m *mediaModel) status.Choice {
	does := map[string]string{
		onnx.DefaultModel: "Images, and videos as sampled frames. Trained with captions, so it matches what a picture shows.",
		onnx.DINOv2Model:  "Images, and videos as sampled frames. Trained on pictures alone, so it matches how a picture looks.",
	}
	onnxEngine := "ONNX Runtime " + onnx.LibVersion
	c := status.Choice{Role: "media model", Note: "make run MODEL=<name>; " + switchNote}
	for _, name := range onnx.Models() {
		spec, err := onnx.LookupSpec(name, "")
		if err != nil {
			continue
		}
		c.Options = append(c.Options, status.Option{
			Name: name, Engine: onnxEngine, Dims: spec.Dims, Bytes: spec.Size, Does: does[name],
			Select: "MODEL=" + name, Key: keyIf(name == m.name, "media"),
		})
	}
	for _, name := range yzma.VisionModels() {
		spec, err := yzma.LookupVisionSpec(name, "")
		if err != nil {
			continue
		}
		sel := "MODEL=" + name
		if name == yzma.DefaultVisionModel {
			sel = "MODEL=" + qwenAliases[0]
		}
		c.Options = append(c.Options, status.Option{
			Name: name, Engine: "llama.cpp " + yzma.LibTag + " with libmtmd", Dims: spec.Dims, Bytes: spec.Size + spec.ProjectorSize,
			Does:   "Text, images, and videos as segments, in one space: the page searches by words too. Loaded at startup, not lazily.",
			Select: sel, Key: keyIf(name == m.name, "media"),
		})
	}
	return c
}

// textChoice lists the text embedding models provider/yzma pins. The
// example runs the default one; the others are the library's, for an
// application to pick with yzma.Config.Model.
func textChoice(t *textModel) status.Choice {
	c := status.Choice{Role: "text embedding model", Note: "the example runs " + yzma.DefaultModel + "; provider/yzma runs the others too (yzma.Config.Model)"}
	for _, name := range yzma.Models() {
		spec, err := yzma.LookupSpec(name, "")
		if err != nil {
			continue
		}
		o := status.Option{Name: name, Engine: "llama.cpp " + yzma.LibTag, Dims: spec.Dims, Bytes: spec.Size, Key: keyIf(name == t.name, "text")}
		if name == yzma.DefaultModel {
			o.Does = "Multilingual, 512 tokens a chunk: a question in one language finds documents in another."
		}
		c.Options = append(c.Options, o)
	}
	return c
}

// chatChoice lists the chat models provider/yzma pins.
func chatChoice(ch *chatModel) status.Choice {
	c := status.Choice{Role: "chat model"}
	for _, name := range yzma.CompleterModels() {
		spec, err := yzma.LookupCompleterSpec(name, "")
		if err != nil {
			continue
		}
		c.Options = append(c.Options, status.Option{
			Name: name, Engine: "llama.cpp " + yzma.LibTag, Bytes: spec.Size,
			Does: "Writes new tags for a note; loads on the first recommendation.",
			Key:  keyIf(name == ch.name, "chat"),
		})
	}
	if len(c.Options) == 1 {
		c.Note = "the only chat model provider/yzma pins"
	}
	return c
}

// hashChoice lists the duplicates page's hashes: no model, all four
// computed in pure Go, picked on the page.
func hashChoice(f *duplicates.Finder) status.Choice {
	does := map[string]string{
		"phash": "from the picture's lowest frequencies (DCT): unrelated pictures agree least by chance",
		"dhash": "from each cell against its neighbour: a small mark flips only the bits under it",
	}
	c := status.Choice{Role: "perceptual hash", Note: "provider/phash, pure Go: no model to load or download; pick the hash and its bits on the page"}
	for _, k := range f.Kinds() {
		c.Options = append(c.Options, status.Option{
			Name: k.Name, Does: fmt.Sprintf("%d bits, %s.", k.Bits, does[strings.ToLower(k.Algorithm)]),
		})
	}
	return c
}

// keyIf is key when the option is the model the service runs, so the page
// can show its state; "" otherwise.
func keyIf(running bool, key string) string {
	if running {
		return key
	}
	return ""
}
