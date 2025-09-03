//go:build darwin
// +build darwin

package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -mmacosx-version-min=10.15 -L. ./libchdb.a -framework CoreFoundation
*/
import "C"
