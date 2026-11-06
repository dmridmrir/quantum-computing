package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

func main() {
	seedBase, _ := strconv.ParseInt(os.Args[1], 10, 64)
	nc := int64(72)
	numTasklets := 11
	ps := []float64{0.001, 0.003, 0.005, 0.007, 0.009}
	syns := make(map[float64][]byte)
	for _, p := range ps {
		full := make([]byte, 0, numTasklets*int(nc))
		for t := 0; t < numTasklets; t++ {
			rng := rand.New(rand.NewSource(seedBase + int64(t)*137))
			for i := int64(0); i < nc; i++ {
				if rng.Float64() < p {
					full = append(full, 1)
				} else {
					full = append(full, 0)
				}
			}
		}
		syns[p] = full
	}
	for _, p := range ps {
		fmt.Printf("p=%.3f weight=%d\n", p, sum(syns[p]))
	}
	fmt.Println()
	// compare adjacent pairs
	for i := 0; i < len(ps)-1; i++ {
		a, b := syns[ps[i]], syns[ps[i+1]]
		diff := 0
		for j := range a {
			if a[j] != b[j] {
				diff++
			}
		}
		fmt.Printf("diff bits between p=%.3f and p=%.3f: %d\n", ps[i], ps[i+1], diff)
	}
	// check 0.003 vs 0.005 vs 0.007 identical?
	id357 := equal(syns[0.003], syns[0.005]) && equal(syns[0.005], syns[0.007])
	fmt.Printf("\np=0.003,0.005,0.007 syndromes bit-identical: %v\n", id357)
}

func sum(b []byte) int {
	s := 0
	for _, v := range b {
		s += int(v)
	}
	return s
}
func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
