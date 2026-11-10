package prim

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"
	"sort"
	"strings"

	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

const (
	tessAstarMaxDepth          = 256
	tessAstarNodeBytes         = 16
	tessAstarArenaBytes        = 8
	tessAstarOutputBytes       = 48 + tessAstarMaxDepth*2
	tessAstarStatusSolved      = 0
	tessAstarStatusLowConf     = 1
	tessAstarStatusFallbck     = 2
	tessAstarNoPosition        = 0xFF
	tessAstarDefaultCap        = 1 << 20
	tessAstarDefaultVisitedCap = 1 << 17
	tessAstarVisitedEntryBytes = 80
	// The DPU ABI reserves two group slots, but the production kernel launches
	// one syndrome per DPU.  Keep the ABI layout stable without wasting half of
	// MRAM on an inactive heap/arena.
	tessAstarABIGroups    = 2
	tessAstarActiveGroups = 1
	tessAstarMaxSiblings  = 255
)

// TesseractAstar replays one recorded syndrome (candidate 0 of a
// tesseract-dpu-trace-v1 file) through the DPU-resident A* kernel and checks
// the returned error chain against a Go implementation of the same
// integer-exact search.
type TesseractAstar struct {
	numDpus       int
	numTasklets   int
	numExecutions int

	arguments    []uint32
	mramInput    []byte
	expected     []byte
	outputOffset int64
}

type tessAstarResult struct {
	status     uint32
	chain      []uint16
	expansions uint32
	candidates uint32
	heapPeak   uint32
	arenaUsed  uint32
	pushed     uint32
	heapSize   uint32
	minNumDets uint32
	maxNumDets uint32
}

func (this *TesseractAstar) Init(commandLineParser *misc.CommandLineParser) {
	numChannels := int(commandLineParser.IntParameter("num_channels"))
	numRanks := int(commandLineParser.IntParameter("num_ranks_per_channel"))
	numDpusPerRank := int(commandLineParser.IntParameter("num_dpus_per_rank"))
	this.numDpus = numChannels * numRanks * numDpusPerRank
	this.numTasklets = int(commandLineParser.IntParameter("num_tasklets"))
	this.numExecutions = 1

	tracePath := strings.TrimSpace(commandLineParser.StringParameter("tesseract_trace"))
	if tracePath == "" {
		panic(errors.New("TESSERACT_ASTAR requires --tesseract_trace; candidate 0 is the syndrome"))
	}
	tracePaths := []string{tracePath}
	if second := strings.TrimSpace(commandLineParser.StringParameter("tesseract_trace2")); second != "" {
		tracePaths = append(tracePaths, second)
	}
	if len(tracePaths) > tessAstarActiveGroups {
		panic(fmt.Errorf("TESSERACT_ASTAR supports at most %d active syndrome per DPU", tessAstarActiveGroups))
	}
	workload := tessLoadTrace(tracePaths[0])
	params := commandLineParser.DataPrepParams()
	capacity := tessAstarDefaultCap
	detBeam := ^uint32(0)
	maxExpansions := uint32(0)
	reverseOrder := false
	noRevisit := false
	pqlimit := ^uint32(0)
	visitedCapacity := 0
	if len(params) >= 1 && params[0] > 0 {
		capacity = params[0]
	}
	if len(params) >= 2 && params[1] >= 0 {
		detBeam = uint32(params[1])
	}
	if len(params) >= 3 && params[2] >= 0 {
		maxExpansions = uint32(params[2])
	}
	if len(params) >= 4 && params[3] != 0 {
		reverseOrder = true
	}
	if len(params) >= 5 && params[4] != 0 {
		noRevisit = true
		visitedCapacity = tessAstarDefaultVisitedCap
	}
	if len(params) >= 6 && params[5] >= 0 {
		pqlimit = uint32(params[5])
	}
	if len(params) >= 7 && params[6] > 0 {
		visitedCapacity = params[6]
	}
	if visitedCapacity != 0 && (visitedCapacity&(visitedCapacity-1)) != 0 {
		panic(errors.New("TESSERACT_ASTAR visited capacity must be a power of two"))
	}
	noRevisitFlag := uint32(0)
	if noRevisit {
		noRevisitFlag = 1
	}
	if workload.numErrors > 65535 {
		panic(errors.New("TESSERACT_ASTAR stores error IDs as 16-bit values"))
	}

	d2eOffsets, d2eErrorIDs := tessBuildD2E(workload.numDetectors, workload.numErrors,
		workload.e2dOffsets, workload.e2dDetectorIDs)
	tessSortD2EByLowerBound(d2eOffsets, d2eErrorIDs, workload.e2dOffsets, workload.errorCostsQ)
	costLutOffsets, costLutValues := tessBuildCostLUTFromBaseCosts(workload.numErrors,
		workload.e2dOffsets, workload.errorCostsQ)

	residualWords := (workload.numDetectors + 31) / 32
	strideWords := (residualWords + 1) &^ 1
	syndromes := make([][]uint32, len(tracePaths))
	for i, path := range tracePaths {
		w := workload
		if i > 0 {
			w = tessLoadTrace(path)
			if w.numDetectors != workload.numDetectors || w.numErrors != workload.numErrors {
				panic(errors.New("TESSERACT_ASTAR traces must share one detector error model"))
			}
		}
		syndromes[i] = make([]uint32, strideWords)
		for pos := w.candidateOffsets[0]; pos < w.candidateOffsets[1]; pos++ {
			detectorID := w.candidateDetectorIDs[pos]
			syndromes[i][detectorID>>5] |= uint32(1) << (detectorID & 31)
		}
	}

	mram := make([]byte, 0)
	d2eOffsetsOffset := tessAppendU32Region(&mram, d2eOffsets)
	d2eErrorIDsOffset := tessAppendU32Region(&mram, d2eErrorIDs)
	e2dOffsetsOffset := tessAppendU32Region(&mram, workload.e2dOffsets)
	e2dDetectorIDsOffset := tessAppendU32Region(&mram, workload.e2dDetectorIDs)
	costLutOffsetsOffset := tessAppendU32Region(&mram, costLutOffsets)
	costLutValuesOffset := tessAppendU32Region(&mram, costLutValues)
	errorRecordsOffset := tessAppendU32Region(&mram, tessAstarErrorRecords(workload.numErrors,
		workload.e2dOffsets, workload.e2dDetectorIDs, costLutOffsets, costLutValues))
	neighborOffsets, neighborIDs := tessAstarNeighbors(workload.numDetectors, workload.numErrors,
		workload.e2dOffsets, workload.e2dDetectorIDs, d2eOffsets, d2eErrorIDs)
	neighborOffsetsOffset := tessAppendU32Region(&mram, neighborOffsets)
	neighborIDsOffset := tessAppendU32Region(&mram, tessPackU16(neighborIDs))
	orderPositionsOffset := ^uint32(0)
	if reverseOrder {
		positions := make([]uint16, workload.numDetectors)
		for detector := range positions {
			positions[detector] = uint16(workload.numDetectors - 1 - detector)
		}
		orderPositionsOffset = tessAppendU32Region(&mram, tessPackU16(positions))
	}
	var syndromeOffsets [tessAstarABIGroups]uint32
	for i := range syndromes {
		syndromeOffsets[i] = tessAppendU32Region(&mram, syndromes[i])
	}
	tessAlign8(&mram)
	outputOffset := uint32(len(mram))
	var outputOffsets [tessAstarABIGroups]uint32
	for i := 0; i < tessAstarABIGroups; i++ {
		outputOffsets[i] = outputOffset + uint32(i*tessAstarOutputBytes)
	}
	mram = append(mram, make([]byte, tessAstarABIGroups*tessAstarOutputBytes)...)
	tessAlign8(&mram)
	cursor := uint64(len(mram))
	var heapOffsets, arenaOffsets [tessAstarABIGroups]uint32
	var visitedOffsets [tessAstarABIGroups]uint32
	for i := 0; i < tessAstarActiveGroups; i++ {
		heapOffsets[i] = uint32(cursor)
		cursor = (cursor + uint64(capacity)*tessAstarNodeBytes + 7) &^ 7
		arenaOffsets[i] = uint32(cursor)
		cursor = (cursor + uint64(capacity)*tessAstarArenaBytes + 7) &^ 7
		if visitedCapacity != 0 {
			visitedOffsets[i] = uint32(cursor)
			cursor = (cursor + uint64(visitedCapacity)*tessAstarVisitedEntryBytes + 7) &^ 7
		}
	}
	if cursor > tesseractMramBytes {
		panic(fmt.Errorf("TESSERACT_ASTAR capacity %d x %d active groups exceeds the DPU MRAM", capacity, tessAstarActiveGroups))
	}
	this.mramInput = mram
	this.outputOffset = int64(outputOffset)

	this.arguments = []uint32{
		uint32(workload.numDetectors),
		uint32(workload.numErrors),
		d2eOffsetsOffset,
		d2eErrorIDsOffset,
		e2dOffsetsOffset,
		e2dDetectorIDsOffset,
		costLutOffsetsOffset,
		costLutValuesOffset,
		errorRecordsOffset,
		neighborOffsetsOffset,
		neighborIDsOffset,
		workload.detectorPenaltyQ,
		detBeam,
		pqlimit,
		noRevisitFlag,
		maxExpansions,
		uint32(len(syndromes)),
		uint32(capacity),
		uint32(capacity),
	}
	this.arguments = append(this.arguments, syndromeOffsets[:]...)
	this.arguments = append(this.arguments, heapOffsets[:]...)
	this.arguments = append(this.arguments, arenaOffsets[:]...)
	this.arguments = append(this.arguments, outputOffsets[:]...)
	this.arguments = append(this.arguments, visitedOffsets[:]...)
	this.arguments = append(this.arguments, uint32(visitedCapacity), 1) // capacity, generation
	this.arguments = append(this.arguments, orderPositionsOffset)
	this.arguments = append(this.arguments, 0, 0) // order_index per group
	this.arguments = append(this.arguments, 0)    // kernel

	this.expected = make([]byte, 0, tessAstarABIGroups*tessAstarOutputBytes)
	for i := 0; i < tessAstarABIGroups; i++ {
		if i < len(syndromes) {
			result := tessAstarSearch(workload.numDetectors, workload.numErrors, d2eOffsets, d2eErrorIDs,
				workload.e2dOffsets, workload.e2dDetectorIDs, costLutOffsets, costLutValues,
				neighborOffsets, neighborIDs,
				syndromes[i][:residualWords], workload.detectorPenaltyQ, detBeam, maxExpansions,
				capacity, reverseOrder, noRevisit, pqlimit, visitedCapacity)
			fmt.Printf("TESSERACT_ASTAR oracle[%d]: status=%d depth=%d expansions=%d candidates=%d heap_peak=%d pushed=%d chain=%v\n",
				i, result.status, len(result.chain), result.expansions, result.candidates, result.heapPeak,
				result.pushed, result.chain)
			this.expected = append(this.expected, tessAstarEncodeOutput(result)...)
		} else {
			this.expected = append(this.expected, make([]byte, tessAstarOutputBytes)...)
		}
	}
}

// tessAstarErrorRecords packs {lut_begin, cost_lut[e][degree], degree,
// detectors[11]} (32 bytes) per error so the kernel's inner scan needs one
// aligned MRAM read per error, detector list included.
func tessAstarErrorRecords(numErrors int, e2dOffsets []uint32, e2dDetectorIDs []uint32,
	lutOffsets []uint32, lutValues []uint32) []uint32 {
	const maxDegree = 11
	records := make([]uint32, 0, numErrors*8)
	for e := 0; e < numErrors; e++ {
		degree := e2dOffsets[e+1] - e2dOffsets[e]
		if degree > maxDegree {
			panic(fmt.Errorf("error %d has degree %d > %d", e, degree, maxDegree))
		}
		var dets [maxDegree + 1]uint16 // [0] = degree, [1..11] = detector IDs
		dets[0] = uint16(degree)
		for k := uint32(0); k < degree; k++ {
			dets[1+k] = uint16(e2dDetectorIDs[e2dOffsets[e]+k])
		}
		records = append(records, lutOffsets[e], lutValues[lutOffsets[e]+degree])
		for k := 0; k < len(dets); k += 2 {
			records = append(records, uint32(dets[k])|uint32(dets[k+1])<<16)
		}
	}
	return records
}

// tessAstarNeighbors builds Tesseract's eneighbors: for each error, the sorted
// detectors of every error sharing a detector with it, minus its own detectors.
func tessAstarNeighbors(numDetectors int, numErrors int, e2dOffsets []uint32,
	e2dDetectorIDs []uint32, d2eOffsets []uint32, d2eErrorIDs []uint32) ([]uint32, []uint16) {
	offsets := make([]uint32, numErrors+1)
	ids := make([]uint16, 0)
	own := make([]int32, numDetectors)  // stamp e+1 when d in D(e)
	seen := make([]int32, numDetectors) // stamp e+1 when d already appended
	for e := 0; e < numErrors; e++ {
		stamp := int32(e + 1)
		for pos := e2dOffsets[e]; pos < e2dOffsets[e+1]; pos++ {
			own[e2dDetectorIDs[pos]] = stamp
		}
		for pos := e2dOffsets[e]; pos < e2dOffsets[e+1]; pos++ {
			d := e2dDetectorIDs[pos]
			for epos := d2eOffsets[d]; epos < d2eOffsets[d+1]; epos++ {
				other := d2eErrorIDs[epos]
				for opos := e2dOffsets[other]; opos < e2dOffsets[other+1]; opos++ {
					od := e2dDetectorIDs[opos]
					if own[od] == stamp || seen[od] == stamp {
						continue
					}
					seen[od] = stamp
					ids = append(ids, uint16(od))
				}
			}
		}
		begin := int(offsets[e])
		sort.Slice(ids[begin:], func(i, j int) bool { return ids[begin+i] < ids[begin+j] })
		offsets[e+1] = uint32(len(ids))
	}
	return offsets, ids
}

func tessPackU16(values []uint16) []uint32 {
	packed := make([]uint32, (len(values)+1)/2)
	for i, v := range values {
		packed[i/2] |= uint32(v) << (16 * uint(i%2))
	}
	return packed
}

func tessAstarEncodeOutput(result tessAstarResult) []byte {
	out := make([]byte, 0, tessAstarOutputBytes)
	for _, value := range []uint32{result.status, uint32(len(result.chain)), result.expansions,
		result.candidates, result.heapPeak, result.arenaUsed, result.pushed, 0,
		result.heapSize, result.minNumDets, result.maxNumDets, 0} {
		out = binary.LittleEndian.AppendUint32(out, value)
	}
	for i := 0; i < tessAstarMaxDepth; i++ {
		value := uint16(0)
		if i < len(result.chain) {
			value = result.chain[i]
		}
		out = binary.LittleEndian.AppendUint16(out, value)
	}
	return out
}

type tessAstarNode struct {
	cost    uint32
	path    uint32
	chain   uint32
	numDets uint16
	depth   uint16
}

type tessAstarLink struct {
	parent      uint32
	errorID     uint16
	minDetector uint16
}

func tessAstarLess(a, b *tessAstarNode) bool {
	return a.cost < b.cost || (a.cost == b.cost && a.numDets > b.numDets)
}

// tessAstarHeap mirrors the kernel's binary heap exactly so tie handling
// among equal keys matches the DPU.
type tessAstarHeap struct {
	nodes []tessAstarNode
	peak  int
	cap   int
}

func (h *tessAstarHeap) push(node tessAstarNode) bool {
	if len(h.nodes) >= h.cap {
		return false
	}
	h.nodes = append(h.nodes, tessAstarNode{})
	index := len(h.nodes) - 1
	if len(h.nodes) > h.peak {
		h.peak = len(h.nodes)
	}
	for index > 0 {
		parent := (index - 1) >> 1
		if !tessAstarLess(&node, &h.nodes[parent]) {
			break
		}
		h.nodes[index] = h.nodes[parent]
		index = parent
	}
	h.nodes[index] = node
	return true
}

func (h *tessAstarHeap) pop() tessAstarNode {
	out := h.nodes[0]
	size := len(h.nodes) - 1
	if size == 0 {
		h.nodes = h.nodes[:0]
		return out
	}
	last := h.nodes[size]
	h.nodes = h.nodes[:size]
	index := 0
	for {
		left := index*2 + 1
		if left >= size {
			break
		}
		best := left
		child := h.nodes[left]
		if left+1 < size && tessAstarLess(&h.nodes[left+1], &child) {
			best = left + 1
			child = h.nodes[left+1]
		}
		if !tessAstarLess(&child, &last) {
			break
		}
		h.nodes[index] = child
		index = best
	}
	h.nodes[index] = last
	return out
}

func tessAstarSearch(numDetectors int, numErrors int, d2eOffsets []uint32, d2eErrorIDs []uint32,
	e2dOffsets []uint32, e2dDetectorIDs []uint32, lutOffsets []uint32, lutValues []uint32,
	neighborOffsets []uint32, neighborIDs []uint16,
	syndrome []uint32, penalty uint32, detBeam uint32, maxExpansions uint32,
	capacity int, reverseOrder bool, noRevisit bool, pqlimit uint32,
	visitedCapacity int) tessAstarResult {
	residualWords := len(syndrome)
	result := tessAstarResult{}
	numDets := uint32(0)
	for word := 0; word < residualWords; word++ {
		bitsLeft := syndrome[word]
		numDets += uint32(bits.OnesCount32(bitsLeft))
		for bitsLeft != 0 {
			detector := uint32(word*32 + bits.TrailingZeros32(bitsLeft))
			bitsLeft &= bitsLeft - 1
			if d2eOffsets[detector] == d2eOffsets[detector+1] {
				result.status = tessAstarStatusLowConf
				return result
			}
		}
	}

	heap := &tessAstarHeap{cap: capacity}
	arena := make([]tessAstarLink, 0)
	position := make([]uint16, numErrors)
	for i := range position {
		position[i] = tessAstarNoPosition
	}
	blocked := make([]uint32, (numErrors+31)/32)
	residual := make([]uint32, residualWords)
	candidate := make([]uint32, residualWords)
	minNumDets := numDets
	maxNumDets := uint32(^uint32(0))
	if detBeam != ^uint32(0) {
		maxNumDets = numDets + detBeam
	}
	heap.push(tessAstarNode{cost: 0, path: 0, chain: ^uint32(0), numDets: uint16(numDets)})
	visited := make(map[string]struct{})
	stateKey := func(target []uint32, count uint32) string {
		buf := make([]byte, 4+len(target)*4)
		binary.LittleEndian.PutUint32(buf, count)
		for i, value := range target {
			binary.LittleEndian.PutUint32(buf[4+i*4:], value)
		}
		return string(buf)
	}

	flip := func(target []uint32, errorID uint32) {
		for pos := e2dOffsets[errorID]; pos < e2dOffsets[errorID+1]; pos++ {
			detector := e2dDetectorIDs[pos]
			target[detector>>5] ^= uint32(1) << (detector & 31)
		}
	}
	const noPrefix = ^uint32(0)
	// evaluateDetector mirrors the kernel's evaluate_detector: cheapest unblocked
	// incident error cost plus penalty, or INF.
	evaluateDetector := func(target []uint32, detector uint32, siblingIndex uint32) uint32 {
		minimum := tesseractInfCost
		for pos := d2eOffsets[detector]; pos < d2eOffsets[detector+1]; pos++ {
			errorID := d2eErrorIDs[pos]
			if (blocked[errorID>>5]>>(errorID&31))&1 != 0 {
				continue
			}
			if siblingIndex != noPrefix && uint32(position[errorID]) <= siblingIndex {
				continue
			}
			degree := e2dOffsets[errorID+1] - e2dOffsets[errorID]
			lutBegin := lutOffsets[errorID]
			if minimum != tesseractInfCost && lutValues[lutBegin+degree] >= minimum {
				break
			}
			overlap := uint32(0)
			for dpos := e2dOffsets[errorID]; dpos < e2dOffsets[errorID+1]; dpos++ {
				incident := e2dDetectorIDs[dpos]
				overlap += (target[incident>>5] >> (incident & 31)) & 1
			}
			if overlap == 0 {
				continue
			}
			if cost := lutValues[lutBegin+overlap]; cost < minimum {
				minimum = cost
			}
		}
		if minimum == tesseractInfCost || minimum >= tesseractInfCost-penalty {
			return tesseractInfCost
		}
		return minimum + penalty
	}
	parentCost := make([]uint32, numDetectors)

	for {
		if maxExpansions != 0 && result.expansions >= maxExpansions {
			result.status = tessAstarStatusFallbck
			result.heapSize = uint32(len(heap.nodes))
			result.minNumDets = minNumDets
			result.maxNumDets = maxNumDets
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			return result
		}
		var node tessAstarNode
		for {
			if len(heap.nodes) == 0 {
				result.status = tessAstarStatusLowConf
				result.expansions = uint32(result.expansions)
				result.heapPeak = uint32(heap.peak)
				result.arenaUsed = uint32(len(arena))
				result.heapSize = uint32(len(heap.nodes))
				result.minNumDets = minNumDets
				result.maxNumDets = maxNumDets
				result.heapPeak = uint32(heap.peak)
				result.arenaUsed = uint32(len(arena))
				return result
			}
			node = heap.pop()
			if uint32(node.numDets) <= maxNumDets {
				break
			}
		}
		copy(residual, syndrome)
		for i := range blocked {
			blocked[i] = 0
		}
		depth := 0
		walker := node.chain
		for walker != ^uint32(0) {
			link := arena[walker]
			for pos := d2eOffsets[link.minDetector]; pos < d2eOffsets[link.minDetector+1]; pos++ {
				other := d2eErrorIDs[pos]
				blocked[other>>5] |= uint32(1) << (other & 31)
				if other == uint32(link.errorID) {
					break
				}
			}
			flip(residual, uint32(link.errorID))
			depth++
			walker = link.parent
		}
		if node.numDets == 0 {
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			if depth > tessAstarMaxDepth {
				heap.push(node)
				result.status = tessAstarStatusFallbck
				result.heapSize = uint32(len(heap.nodes))
				result.minNumDets = minNumDets
				result.maxNumDets = maxNumDets
				result.heapPeak = uint32(heap.peak)
				result.arenaUsed = uint32(len(arena))
				return result
			}
			chain := make([]uint16, depth)
			walker = node.chain
			index := depth
			for walker != ^uint32(0) {
				link := arena[walker]
				index--
				chain[index] = link.errorID
				walker = link.parent
			}
			result.status = tessAstarStatusSolved
			result.chain = chain
			result.heapSize = uint32(len(heap.nodes))
			result.minNumDets = minNumDets
			result.maxNumDets = maxNumDets
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			return result
		}
		if noRevisit {
			key := stateKey(residual, uint32(node.numDets))
			if _, ok := visited[key]; ok {
				continue
			}
			if visitedCapacity == 0 || len(visited)*10 >= visitedCapacity*7 {
				heap.push(node)
				result.status = tessAstarStatusFallbck
				result.heapSize = uint32(len(heap.nodes))
				result.minNumDets = minNumDets
				result.maxNumDets = maxNumDets
				result.heapPeak = uint32(heap.peak)
				result.arenaUsed = uint32(len(arena))
				return result
			}
			visited[key] = struct{}{}
		}
		if uint32(node.numDets) < minNumDets {
			minNumDets = uint32(node.numDets)
			if detBeam != ^uint32(0) && minNumDets+detBeam < maxNumDets {
				maxNumDets = minNumDets + detBeam
			}
		}
		minDetector := uint32(0)
		first := true
		for word := 0; word < residualWords; word++ {
			bitsLeft := residual[word]
			for bitsLeft != 0 {
				detector := uint32(word*32 + bits.TrailingZeros32(bitsLeft))
				bitsLeft &= bitsLeft - 1
				if first || (!reverseOrder && detector < minDetector) ||
					(reverseOrder && detector > minDetector) {
					minDetector = detector
					first = false
				}
			}
		}
		begin := d2eOffsets[minDetector]
		count := d2eOffsets[minDetector+1] - begin
		if count > tessAstarMaxSiblings || len(heap.nodes)+int(count) > capacity ||
			len(arena)+int(count) > capacity {
			heap.push(node)
			result.status = tessAstarStatusFallbck
			result.heapSize = uint32(len(heap.nodes))
			result.minNumDets = minNumDets
			result.maxNumDets = maxNumDets
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			return result
		}
		result.expansions++
		parentTotal := uint32(0)
		parentInf := false
		for word := 0; word < residualWords; word++ {
			bitsLeft := residual[word]
			for bitsLeft != 0 {
				detector := uint32(word*32 + bits.TrailingZeros32(bitsLeft))
				bitsLeft &= bitsLeft - 1
				cost := evaluateDetector(residual, detector, noPrefix)
				parentCost[detector] = cost
				if cost == tesseractInfCost {
					parentInf = true
				} else {
					parentTotal += cost
				}
			}
		}
		for j := uint32(0); j < count; j++ {
			position[d2eErrorIDs[begin+j]] = uint16(j)
		}
		costs := make([]uint32, count)
		dets := make([]uint16, count)
		evaluated := make([]uint32, count)
		for j := uint32(0); j < count; j++ {
			errorID := d2eErrorIDs[begin+j]
			costs[j] = tesseractInfCost
			if (blocked[errorID>>5]>>(errorID&31))&1 != 0 {
				continue
			}
			copy(candidate, residual)
			flip(candidate, errorID)
			total := uint32(0)
			for word := 0; word < residualWords; word++ {
				total += uint32(bits.OnesCount32(candidate[word]))
			}
			if total > maxNumDets {
				continue
			}
			if noRevisit {
				if _, ok := visited[stateKey(candidate, total)]; ok {
					continue
				}
			}
			dets[j] = uint16(total)
			evaluated[j] = 1
			if parentInf {
				continue
			}
			// Incremental update identical to the kernel: detectors flipped by
			// e_j and residual neighbors of e_j are re-evaluated, all other
			// terms are unchanged.
			h := parentTotal
			infinite := false
			for dpos := e2dOffsets[errorID]; dpos < e2dOffsets[errorID+1] && !infinite; dpos++ {
				detector := e2dDetectorIDs[dpos]
				if (residual[detector>>5]>>(detector&31))&1 != 0 {
					h -= parentCost[detector]
				} else {
					cost := evaluateDetector(candidate, detector, j)
					if cost == tesseractInfCost || cost >= tesseractInfCost-h {
						infinite = true
					} else {
						h += cost
					}
				}
			}
			for npos := neighborOffsets[errorID]; npos < neighborOffsets[errorID+1] && !infinite; npos++ {
				detector := uint32(neighborIDs[npos])
				if (residual[detector>>5]>>(detector&31))&1 == 0 {
					continue
				}
				h -= parentCost[detector]
				cost := evaluateDetector(candidate, detector, j)
				if cost == tesseractInfCost || cost >= tesseractInfCost-h {
					infinite = true
				} else {
					h += cost
				}
			}
			if !infinite {
				costs[j] = h
			}
		}
		failed := false
		for j := uint32(0); j < count; j++ {
			errorID := d2eErrorIDs[begin+j]
			position[errorID] = tessAstarNoPosition
			result.candidates += evaluated[j]
			if costs[j] == tesseractInfCost {
				continue
			}
			step := lutValues[lutOffsets[errorID]+1]
			if step >= tesseractInfCost-node.path {
				continue
			}
			path := node.path + step
			if costs[j] >= tesseractInfCost-path {
				continue
			}
			if len(arena) >= capacity {
				failed = true
				break
			}
			arena = append(arena, tessAstarLink{parent: node.chain, errorID: uint16(errorID),
				minDetector: uint16(minDetector)})
			pushed := heap.push(tessAstarNode{cost: path + costs[j], path: path,
				chain: uint32(len(arena) - 1), numDets: dets[j], depth: node.depth + 1})
			if !pushed {
				failed = true
				break
			}
			result.pushed++
			if pqlimit != ^uint32(0) && result.pushed+1 > pqlimit {
				result.status = tessAstarStatusLowConf
				result.heapPeak = uint32(heap.peak)
				result.arenaUsed = uint32(len(arena))
				result.heapSize = uint32(len(heap.nodes))
				result.minNumDets = minNumDets
				result.maxNumDets = maxNumDets
				return result
			}
		}
		if failed {
			result.status = tessAstarStatusFallbck
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			result.heapSize = uint32(len(heap.nodes))
			result.minNumDets = minNumDets
			result.maxNumDets = maxNumDets
			result.heapPeak = uint32(heap.peak)
			result.arenaUsed = uint32(len(arena))
			return result
		}
	}
}

func (this *TesseractAstar) InputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
	this.validateIndices(execution, dpuID)
	stream := new(encoding.ByteStream)
	stream.Init()
	for _, value := range this.arguments {
		valueWord := new(word.Word)
		valueWord.Init(32)
		valueWord.SetValue(int64(value))
		stream.Merge(valueWord.ToByteStream())
	}
	return map[string]*encoding.ByteStream{"DPU_INPUT_ARGUMENTS": stream}
}

func (this *TesseractAstar) OutputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
	this.validateIndices(execution, dpuID)
	return make(map[string]*encoding.ByteStream)
}

func (this *TesseractAstar) InputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.validateIndices(execution, dpuID)
	return 0, tessByteStream(this.mramInput)
}

func (this *TesseractAstar) OutputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.validateIndices(execution, dpuID)
	return this.outputOffset, tessByteStream(this.expected)
}

func (this *TesseractAstar) NumExecutions() int {
	return this.numExecutions
}

func (this *TesseractAstar) validateIndices(execution int, dpuID int) {
	if execution < 0 || execution >= this.numExecutions {
		panic(errors.New("execution is outside the TESSERACT_ASTAR workload"))
	}
	if dpuID < 0 || dpuID >= this.numDpus {
		panic(errors.New("DPU ID is outside the configured DPU set"))
	}
}
