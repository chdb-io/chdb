//go:build linux
// +build linux

package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -Wl,--whole-archive -lchdb -lc -lm -lrt -lpthread -ldl
*/
import "C"
