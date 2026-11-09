package langai

import "errors"

// ErrClosed is what every call returns, wrapped, on something of this module
// that has been closed: a provider's model, whose native resources are gone,
// a lazy model, and a tagging.Tagger. It stays unusable. Test for it with
// errors.Is to tell a call that arrived after shutdown from one that failed
// while the model ran.
var ErrClosed = errors.New("langai: closed")
