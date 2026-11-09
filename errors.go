package langai

import "errors"

// ErrClosed is what a provider's calls return, wrapped, once the provider
// has been closed: its model and native resources are gone and it stays
// unusable. Test for it with errors.Is to tell a call that arrived after
// shutdown from one that failed while the model ran.
var ErrClosed = errors.New("langai: closed")
