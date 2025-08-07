//go:build linux
// +build linux

package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -Wl,--whole-archive -lchdb -Wl,--no-whole-archive -lc -lm -lrt -lpthread -ldl
*/
import "C"
