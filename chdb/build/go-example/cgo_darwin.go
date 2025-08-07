//go:build darwin
// +build darwin

package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -mmacosx-version-min=10.15 -L. -Wl,-force_load,./libchdb.a -framework CoreFoundation
*/
import "C"
