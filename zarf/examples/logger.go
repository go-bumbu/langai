package main

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/mattn/go-isatty"
	console "github.com/phsym/console-slog"
)

// newLogger is the command's logger, after aether's: on a terminal a
// coloured line a record ("11:20:40 INF scan done indexed=3 ..."), piped or
// redirected one JSON object a record, for a tool to read. Both go to stderr:
// stdout is -write-docs' output.
func newLogger(level slog.Level) *slog.Logger {
	fd := os.Stderr.Fd()
	if isatty.IsTerminal(fd) || isatty.IsCygwinTerminal(fd) {
		return slog.New(console.NewHandler(os.Stderr, &console.HandlerOptions{
			Level: level,
			// seconds, not aether's time.Kitchen: a scan logs several
			// records a second
			TimeFormat: "15:04:05",
		}))
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
}

// parseLevel reads the -log-level flag: debug, info, warn or error.
func parseLevel(s string) (slog.Level, error) {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug, nil
	case "", "info":
		return slog.LevelInfo, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error", "err":
		return slog.LevelError, nil
	}
	return 0, fmt.Errorf("log level %q: want debug, info, warn or error", s)
}
