package prim

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math/bits"
	"math/rand"
	"os"
	"sort"
	"strings"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

const (
	tesseractMaxDetectors     = 4096
	tesseractMaxSparseBlocked = 128
	tesseractNoBlocked        = ^uint32(0)
	tesseractInfCost          = ^uint32(0)
	tesseractMramBytes        = 64 * 1024 * 1024
)

// Tesseract prepares either a recorded MSC Tesseract trace or a deterministic
// synthetic detector-error model for the same DPU candidate evaluator.
type Tesseract struct {
	numDpus       int
	numTasklets   int
	numExecutions int

	arguments    []uint32
	mramInput    []byte
	expected     []uint64
	outputOffset int64
}

type tessWorkload struct {
	numDetectors         int
	numErrors            int
	qBits                int
	detectorPenaltyQ     uint32
	e2dOffsets           []uint32
	e2dDetectorIDs       []uint32
	errorCostsQ          []uint32
	candidateOffsets     []uint32
	candidateDetectorIDs []uint32
	blockedWords         []uint64
	expectedCostsQ       []uint32
}

type tessTraceV1 struct {
	Format                     string   `json:"format"`
	QBits                      int      `json:"q_bits"`
	DetectorPenaltyQ           uint32   `json:"detector_penalty_q"`
	NumDetectors               int      `json:"num_detectors"`
	ErrorDetectorOffsets       []uint32 `json:"error_detector_offsets"`
	ErrorDetectorIDs           []uint32 `json:"error_detector_ids"`
	ErrorCostsQ                []uint32 `json:"error_cost_q"`
	CandidateDetectorOffsets   []uint32 `json:"candidate_detector_offsets"`
	CandidateDetectorIDs       []uint32 `json:"candidate_detector_ids"`
	CandidateBlockedErrorWords []uint64 `json:"candidate_blocked_error_words,omitempty"`
	CandidateCostsQ            []uint32 `json:"candidate_cost_q,omitempty"`
}

func (this *Tesseract) Init(commandLineParser *misc.CommandLineParser) {
	numChannels := int(commandLineParser.IntParameter("num_channels"))
	numRanks := int(commandLineParser.IntParameter("num_ranks_per_channel"))
	numDpusPerRank := int(commandLineParser.IntParameter("num_dpus_per_rank"))

	this.numDpus = numChannels * numRanks * numDpusPerRank
	this.numTasklets = int(commandLineParser.IntParameter("num_tasklets"))
	this.numExecutions = 1

	var workload tessWorkload
	tracePath := strings.TrimSpace(commandLineParser.StringParameter("tesseract_trace"))
	if tracePath == "" {
		workload = tessSyntheticWorkload(commandLineParser.DataPrepParams())
	} else {
		workload = tessLoadTrace(tracePath)
	}

	numCandidates := len(workload.candidateOffsets) - 1
	d2eOffsets, d2eErrorIDs := tessBuildD2E(
		workload.numDetectors,
		workload.numErrors,
		workload.e2dOffsets,
		workload.e2dDetectorIDs,
	)
	tessSortD2EByLowerBound(d2eOffsets, d2eErrorIDs, workload.e2dOffsets,
		workload.errorCostsQ)
	costLutOffsets, costLutValues := tessBuildCostLUTFromBaseCosts(
		workload.numErrors,
		workload.e2dOffsets,
		workload.errorCostsQ,
	)
	candidateResidualWords := tessBuildCandidateResidualWords(
		numCandidates,
		workload.numDetectors,
		workload.candidateOffsets,
		workload.candidateDetectorIDs,
	)
	blockedWords32, blockedRecordOffsets, blockedRecords := tessBuildBlockedLayout(
		workload.blockedWords,
		workload.numErrors,
		numCandidates,
	)

	this.expected = tessEvaluateBatchWithOptions(
		workload.numDetectors,
		d2eOffsets,
		d2eErrorIDs,
		workload.e2dOffsets,
		workload.e2dDetectorIDs,
		costLutOffsets,
		costLutValues,
		workload.candidateOffsets,
		workload.candidateDetectorIDs,
		workload.blockedWords,
		workload.detectorPenaltyQ,
	)
	if err := tessVerifyProducerCosts(this.expected, workload.expectedCostsQ); err != nil {
		panic(err)
	}

	mram := make([]byte, 0)
	d2eOffsetsOffset := tessAppendU32Region(&mram, d2eOffsets)
	d2eErrorIDsOffset := tessAppendU32Region(&mram, d2eErrorIDs)
	e2dOffsetsOffset := tessAppendU32Region(&mram, workload.e2dOffsets)
	e2dDetectorIDsOffset := tessAppendU32Region(&mram, workload.e2dDetectorIDs)
	costLutOffsetsOffset := tessAppendU32Region(&mram, costLutOffsets)
	costLutValuesOffset := tessAppendU32Region(&mram, costLutValues)
	candidateOffsetsOffset := tessAppendU32Region(&mram, workload.candidateOffsets)
	candidateDetectorIDsOffset := tessAppendU32Region(&mram, workload.candidateDetectorIDs)
	candidateResidualWordsOffset := tessAppendU32Region(&mram, candidateResidualWords)
	blockedWordsOffset := uint32(tesseractNoBlocked)
	if len(blockedWords32) != 0 {
		blockedWordsOffset = tessAppendU32Region(&mram, blockedWords32)
	}
	blockedRecordOffsetsOffset := uint32(tesseractNoBlocked)
	blockedRecordsOffset := uint32(tesseractNoBlocked)
	if len(blockedRecords) != 0 {
		blockedRecordOffsetsOffset = tessAppendU32Region(&mram, blockedRecordOffsets)
		blockedRecordsOffset = tessAppendU32Region(&mram, blockedRecords)
	}
	outputCostsOffset := uint32(len(mram))
	if len(mram)+len(this.expected)*8 > tesseractMramBytes {
		panic(errors.New("TESSERACT input and output exceed the DPU MRAM capacity"))
	}
	this.mramInput = mram
	this.outputOffset = int64(outputCostsOffset)

	this.arguments = []uint32{
		uint32(workload.numDetectors),
		uint32(workload.numErrors),
		uint32(numCandidates),
		uint32(len(d2eErrorIDs)),
		uint32(len(workload.e2dDetectorIDs)),
		d2eOffsetsOffset,
		d2eErrorIDsOffset,
		e2dOffsetsOffset,
		e2dDetectorIDsOffset,
		costLutOffsetsOffset,
		costLutValuesOffset,
		candidateOffsetsOffset,
		candidateDetectorIDsOffset,
		candidateResidualWordsOffset,
		blockedWordsOffset,
		blockedRecordOffsetsOffset,
		blockedRecordsOffset,
		outputCostsOffset,
		workload.detectorPenaltyQ,
		uint32(workload.qBits),
		0,
		0,
	}
}

// Small blocked sets are transferred as candidate-local records and loaded
// into WRAM once. Each record starts on an 8-byte boundary and contains
// [count, pad, packed sorted uint16 error IDs..., optional padding]. Large sets
// retain the dense 32-bit bitset representation so the kernel never exceeds
// its WRAM bound.
func tessBuildBlockedLayout(blockedWords []uint64, numErrors int,
	numCandidates int) ([]uint32, []uint32, []uint32) {
	if len(blockedWords) == 0 {
		return nil, nil, nil
	}
	if numErrors > 1<<16 {
		return tessRepackBlockedWords32(blockedWords, numErrors, numCandidates), nil, nil
	}
	words64PerCandidate := (numErrors + 63) / 64
	offsets := make([]uint32, numCandidates)
	records := make([]uint32, 0)
	var blockedIDs [tesseractMaxSparseBlocked]uint16
	for candidateID := 0; candidateID < numCandidates; candidateID++ {
		count := 0
		base := candidateID * words64PerCandidate
		for wordID := 0; wordID < words64PerCandidate; wordID++ {
			word := blockedWords[base+wordID]
			for word != 0 {
				bit := bits.TrailingZeros64(word)
				errorID := wordID*64 + bit
				if errorID < numErrors {
					if count == tesseractMaxSparseBlocked {
						return tessRepackBlockedWords32(blockedWords, numErrors, numCandidates), nil, nil
					}
					blockedIDs[count] = uint16(errorID)
					count++
				}
				word &= word - 1
			}
		}
		offsets[candidateID] = uint32(len(records))
		records = append(records, uint32(count))
		records = append(records, 0)
		for index := 0; index < count; index += 2 {
			packed := uint32(blockedIDs[index])
			if index+1 < count {
				packed |= uint32(blockedIDs[index+1]) << 16
			}
			records = append(records, packed)
		}
		if len(records)&1 != 0 {
			records = append(records, 0)
		}
	}
	return nil, offsets, records
}

func tessSyntheticWorkload(params []int) tessWorkload {
	if len(params) < 3 {
		panic(errors.New("TESSERACT data_prep_params must be candidates,detectors,errors[,seed[,q_bits]]"))
	}

	numCandidates := params[0]
	numDetectors := params[1]
	numErrors := params[2]
	seed := 42
	qBits := 16
	if len(params) >= 4 {
		seed = params[3]
	}
	if len(params) >= 5 {
		qBits = params[4]
	}
	if numCandidates <= 0 {
		panic(errors.New("TESSERACT requires at least one candidate"))
	}
	if numDetectors <= 0 || numDetectors > tesseractMaxDetectors {
		panic(fmt.Errorf("TESSERACT detectors must be in [1,%d]", tesseractMaxDetectors))
	}
	if numErrors < numDetectors {
		panic(errors.New("TESSERACT synthetic workload requires errors >= detectors"))
	}
	if qBits < 0 || qBits > 24 {
		panic(errors.New("TESSERACT q_bits must be in [0,24]"))
	}

	rng := rand.New(rand.NewSource(int64(seed)))
	e2dOffsets, e2dDetectorIDs := tessGenerateE2D(numDetectors, numErrors, rng)
	errorCostsQ := tessGenerateErrorCostsQ(numErrors, qBits, rng)
	candidateOffsets, candidateDetectorIDs := tessGenerateCandidates(numCandidates, numDetectors, rng)
	return tessWorkload{
		numDetectors:         numDetectors,
		numErrors:            numErrors,
		qBits:                qBits,
		e2dOffsets:           e2dOffsets,
		e2dDetectorIDs:       e2dDetectorIDs,
		errorCostsQ:          errorCostsQ,
		candidateOffsets:     candidateOffsets,
		candidateDetectorIDs: candidateDetectorIDs,
	}
}

func tessLoadTrace(path string) tessWorkload {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Errorf("read TESSERACT trace %q: %w", path, err))
	}
	trace := new(tessTraceV1)
	if err := json.Unmarshal(data, trace); err != nil {
		panic(fmt.Errorf("parse TESSERACT trace %q: %w", path, err))
	}
	if err := tessValidateTrace(trace); err != nil {
		panic(fmt.Errorf("invalid TESSERACT trace %q: %w", path, err))
	}
	return tessWorkload{
		numDetectors:         trace.NumDetectors,
		numErrors:            len(trace.ErrorCostsQ),
		qBits:                trace.QBits,
		detectorPenaltyQ:     trace.DetectorPenaltyQ,
		e2dOffsets:           trace.ErrorDetectorOffsets,
		e2dDetectorIDs:       trace.ErrorDetectorIDs,
		errorCostsQ:          trace.ErrorCostsQ,
		candidateOffsets:     trace.CandidateDetectorOffsets,
		candidateDetectorIDs: trace.CandidateDetectorIDs,
		blockedWords:         trace.CandidateBlockedErrorWords,
		expectedCostsQ:       trace.CandidateCostsQ,
	}
}

func tessValidateTrace(trace *tessTraceV1) error {
	if trace.Format != "tesseract-dpu-trace-v1" {
		return fmt.Errorf("format = %q, want tesseract-dpu-trace-v1", trace.Format)
	}
	if trace.QBits < 0 || trace.QBits > 24 {
		return errors.New("q_bits must be in [0,24]")
	}
	if trace.NumDetectors <= 0 || trace.NumDetectors > tesseractMaxDetectors {
		return fmt.Errorf("num_detectors must be in [1,%d]", tesseractMaxDetectors)
	}
	if len(trace.ErrorCostsQ) == 0 {
		return errors.New("error_cost_q must not be empty")
	}
	for errorID, cost := range trace.ErrorCostsQ {
		if cost == tesseractInfCost {
			return fmt.Errorf("error_cost_q[%d] uses the reserved INF value", errorID)
		}
	}
	if err := tessValidateOffsets("error_detector", trace.ErrorDetectorOffsets,
		len(trace.ErrorCostsQ), len(trace.ErrorDetectorIDs)); err != nil {
		return err
	}
	if len(trace.CandidateDetectorOffsets) < 2 {
		return errors.New("candidate_detector_offsets must describe at least one candidate")
	}
	if err := tessValidateOffsets("candidate_detector", trace.CandidateDetectorOffsets,
		len(trace.CandidateDetectorOffsets)-1, len(trace.CandidateDetectorIDs)); err != nil {
		return err
	}
	for pos, detectorID := range trace.ErrorDetectorIDs {
		if detectorID >= uint32(trace.NumDetectors) {
			return fmt.Errorf("error_detector_ids[%d] = %d is out of range", pos, detectorID)
		}
	}
	for pos, detectorID := range trace.CandidateDetectorIDs {
		if detectorID >= uint32(trace.NumDetectors) {
			return fmt.Errorf("candidate_detector_ids[%d] = %d is out of range", pos, detectorID)
		}
	}
	for candidateID := 0; candidateID+1 < len(trace.CandidateDetectorOffsets); candidateID++ {
		seen := make(map[uint32]bool)
		begin := trace.CandidateDetectorOffsets[candidateID]
		end := trace.CandidateDetectorOffsets[candidateID+1]
		for pos := begin; pos < end; pos++ {
			detectorID := trace.CandidateDetectorIDs[pos]
			if seen[detectorID] {
				return fmt.Errorf("candidate %d repeats detector %d", candidateID, detectorID)
			}
			seen[detectorID] = true
		}
	}
	if len(trace.CandidateBlockedErrorWords) != 0 {
		numCandidates := len(trace.CandidateDetectorOffsets) - 1
		wordsPerCandidate := (len(trace.ErrorCostsQ) + 63) / 64
		want := numCandidates * wordsPerCandidate
		if len(trace.CandidateBlockedErrorWords) != want {
			return fmt.Errorf("candidate_blocked_error_words has %d words, want %d",
				len(trace.CandidateBlockedErrorWords), want)
		}
	}
	numCandidates := len(trace.CandidateDetectorOffsets) - 1
	if len(trace.CandidateCostsQ) != 0 && len(trace.CandidateCostsQ) != numCandidates {
		return fmt.Errorf("candidate_cost_q has %d entries, want %d",
			len(trace.CandidateCostsQ), numCandidates)
	}
	return nil
}

func tessValidateOffsets(name string, offsets []uint32, rows int, ids int) error {
	if len(offsets) != rows+1 {
		return fmt.Errorf("%s_offsets has %d entries, want %d", name, len(offsets), rows+1)
	}
	if offsets[0] != 0 {
		return fmt.Errorf("%s_offsets[0] must be zero", name)
	}
	for index := 1; index < len(offsets); index++ {
		if offsets[index] < offsets[index-1] {
			return fmt.Errorf("%s_offsets is not monotonic at index %d", name, index)
		}
	}
	if offsets[len(offsets)-1] != uint32(ids) {
		return fmt.Errorf("last %s offset is %d, want %d", name, offsets[len(offsets)-1], ids)
	}
	return nil
}

func tessVerifyProducerCosts(oracle []uint64, producer []uint32) error {
	if len(producer) == 0 {
		return nil
	}
	if len(producer) != len(oracle) {
		return fmt.Errorf("producer cost count is %d, Go oracle count is %d",
			len(producer), len(oracle))
	}
	for candidateID, expected := range producer {
		if oracle[candidateID] != uint64(expected) {
			return fmt.Errorf(
				"TESSERACT trace candidate %d cost mismatch: Go oracle=%d producer=%d",
				candidateID, oracle[candidateID], expected)
		}
	}
	return nil
}

func (this *Tesseract) InputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
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

func (this *Tesseract) OutputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
	this.validateIndices(execution, dpuID)
	return make(map[string]*encoding.ByteStream)
}

func (this *Tesseract) InputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.validateIndices(execution, dpuID)
	return 0, tessByteStream(this.mramInput)
}

func (this *Tesseract) OutputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.validateIndices(execution, dpuID)
	stream := new(encoding.ByteStream)
	stream.Init()
	for _, value := range this.expected {
		valueWord := new(word.Word)
		valueWord.Init(64)
		valueWord.SetValue(int64(value))
		stream.Merge(valueWord.ToByteStream())
	}
	return this.outputOffset, stream
}

func (this *Tesseract) NumExecutions() int {
	return this.numExecutions
}

func (this *Tesseract) validateIndices(execution int, dpuID int) {
	if execution < 0 || execution >= this.numExecutions {
		panic(errors.New("execution is outside the TESSERACT workload"))
	}
	if dpuID < 0 || dpuID >= this.numDpus {
		panic(errors.New("DPU ID is outside the configured DPU set"))
	}
}

func tessGenerateE2D(numDetectors int, numErrors int, rng *rand.Rand) ([]uint32, []uint32) {
	offsets := make([]uint32, numErrors+1)
	detectorIDs := make([]uint32, 0, numErrors*4)
	for errorID := 0; errorID < numErrors; errorID++ {
		degree := 2 + rng.Intn(5)
		if degree > numDetectors {
			degree = numDetectors
		}

		selected := map[int]bool{errorID % numDetectors: true}
		for len(selected) < degree {
			selected[rng.Intn(numDetectors)] = true
		}
		ids := make([]int, 0, len(selected))
		for detectorID := range selected {
			ids = append(ids, detectorID)
		}
		sort.Ints(ids)
		for _, detectorID := range ids {
			detectorIDs = append(detectorIDs, uint32(detectorID))
		}
		offsets[errorID+1] = uint32(len(detectorIDs))
	}
	return offsets, detectorIDs
}

func tessBuildD2E(numDetectors int, numErrors int, e2dOffsets []uint32, e2dDetectorIDs []uint32) ([]uint32, []uint32) {
	counts := make([]uint32, numDetectors)
	for _, detectorID := range e2dDetectorIDs {
		counts[detectorID]++
	}

	offsets := make([]uint32, numDetectors+1)
	for detectorID := 0; detectorID < numDetectors; detectorID++ {
		offsets[detectorID+1] = offsets[detectorID] + counts[detectorID]
	}

	errorIDs := make([]uint32, len(e2dDetectorIDs))
	cursors := append([]uint32(nil), offsets[:numDetectors]...)
	for errorID := 0; errorID < numErrors; errorID++ {
		for pos := e2dOffsets[errorID]; pos < e2dOffsets[errorID+1]; pos++ {
			detectorID := e2dDetectorIDs[pos]
			errorIDs[cursors[detectorID]] = uint32(errorID)
			cursors[detectorID]++
		}
	}
	return offsets, errorIDs
}

func tessSortD2EByLowerBound(d2eOffsets []uint32, d2eErrorIDs []uint32,
	e2dOffsets []uint32, errorCostsQ []uint32) {
	for detectorID := 0; detectorID+1 < len(d2eOffsets); detectorID++ {
		begin := int(d2eOffsets[detectorID])
		end := int(d2eOffsets[detectorID+1])
		sort.Slice(d2eErrorIDs[begin:end], func(i int, j int) bool {
			lhs := d2eErrorIDs[begin+i]
			rhs := d2eErrorIDs[begin+j]
			lhsDegree := uint64(e2dOffsets[lhs+1] - e2dOffsets[lhs])
			rhsDegree := uint64(e2dOffsets[rhs+1] - e2dOffsets[rhs])
			lhsScaled := uint64(errorCostsQ[lhs]) * rhsDegree
			rhsScaled := uint64(errorCostsQ[rhs]) * lhsDegree
			if lhsScaled != rhsScaled {
				return lhsScaled < rhsScaled
			}
			return lhs < rhs
		})
	}
}

func tessGenerateErrorCostsQ(numErrors int, qBits int, rng *rand.Rand) []uint32 {
	costs := make([]uint32, numErrors)
	scale := uint64(1) << uint(qBits)
	for errorID := range costs {
		costs[errorID] = uint32(scale * uint64(1+rng.Intn(15)))
	}
	return costs
}

func tessBuildCostLUTFromBaseCosts(numErrors int, e2dOffsets []uint32,
	errorCostsQ []uint32) ([]uint32, []uint32) {
	offsets := make([]uint32, numErrors+1)
	values := make([]uint32, 0)
	for errorID := 0; errorID < numErrors; errorID++ {
		degree := int(e2dOffsets[errorID+1] - e2dOffsets[errorID])
		values = append(values, tesseractInfCost)
		baseCost := uint64(errorCostsQ[errorID])
		for overlap := 1; overlap <= degree; overlap++ {
			value := (baseCost + uint64(overlap)/2) / uint64(overlap)
			if value >= uint64(tesseractInfCost) {
				value = uint64(tesseractInfCost - 1)
			}
			values = append(values, uint32(value))
		}
		offsets[errorID+1] = uint32(len(values))
	}
	return offsets, values
}

func tessGenerateCandidates(numCandidates int, numDetectors int, rng *rand.Rand) ([]uint32, []uint32) {
	offsets := make([]uint32, numCandidates+1)
	detectorIDs := make([]uint32, 0)
	maxEvents := numDetectors / 32
	if maxEvents < 1 {
		maxEvents = 1
	}
	if maxEvents > 64 {
		maxEvents = 64
	}

	for candidateID := 0; candidateID < numCandidates; candidateID++ {
		numEvents := 1 + rng.Intn(maxEvents)
		selected := make(map[int]bool, numEvents)
		for len(selected) < numEvents {
			selected[rng.Intn(numDetectors)] = true
		}
		ids := make([]int, 0, len(selected))
		for detectorID := range selected {
			ids = append(ids, detectorID)
		}
		sort.Ints(ids)
		for _, detectorID := range ids {
			detectorIDs = append(detectorIDs, uint32(detectorID))
		}
		offsets[candidateID+1] = uint32(len(detectorIDs))
	}
	return offsets, detectorIDs
}

func tessBuildCandidateResidualWords(numCandidates int, numDetectors int,
	candidateOffsets []uint32, candidateDetectorIDs []uint32) []uint32 {
	wordsPerCandidate := (numDetectors + 31) / 32
	// Each candidate starts on an 8-byte boundary so the DPU can DMA the
	// complete residual into WRAM with one aligned transfer.
	strideWords := (wordsPerCandidate + 1) &^ 1
	words := make([]uint32, numCandidates*strideWords)
	for candidateID := 0; candidateID < numCandidates; candidateID++ {
		for pos := candidateOffsets[candidateID]; pos < candidateOffsets[candidateID+1]; pos++ {
			detectorID := candidateDetectorIDs[pos]
			wordIndex := candidateID*strideWords + int(detectorID>>5)
			words[wordIndex] |= uint32(1) << (detectorID & 31)
		}
	}
	return words
}

// Traces use 64-bit JSON words to keep their producer contract compact. The
// DPU is a 32-bit machine, so repack candidate-major masks before transfer and
// avoid 64-bit shifts in the kernel's hottest filtering path.
func tessRepackBlockedWords32(blockedWords []uint64, numErrors int,
	numCandidates int) []uint32 {
	if len(blockedWords) == 0 {
		return nil
	}
	words64PerCandidate := (numErrors + 63) / 64
	words32PerCandidate := (numErrors + 31) / 32
	result := make([]uint32, 0, numCandidates*words32PerCandidate)
	for candidateID := 0; candidateID < numCandidates; candidateID++ {
		base := candidateID * words64PerCandidate
		for word32 := 0; word32 < words32PerCandidate; word32++ {
			value := blockedWords[base+word32/2]
			result = append(result, uint32(value>>uint((word32&1)*32)))
		}
	}
	return result
}

func tessEvaluateBatch(numDetectors int, d2eOffsets []uint32, d2eErrorIDs []uint32,
	e2dOffsets []uint32, e2dDetectorIDs []uint32, costLutOffsets []uint32,
	costLutValues []uint32, candidateOffsets []uint32, candidateDetectorIDs []uint32) []uint64 {
	return tessEvaluateBatchWithOptions(
		numDetectors,
		d2eOffsets,
		d2eErrorIDs,
		e2dOffsets,
		e2dDetectorIDs,
		costLutOffsets,
		costLutValues,
		candidateOffsets,
		candidateDetectorIDs,
		nil,
		0,
	)
}

func tessEvaluateBatchWithOptions(numDetectors int, d2eOffsets []uint32,
	d2eErrorIDs []uint32, e2dOffsets []uint32, e2dDetectorIDs []uint32,
	costLutOffsets []uint32, costLutValues []uint32, candidateOffsets []uint32,
	candidateDetectorIDs []uint32, blockedWords []uint64, detectorPenaltyQ uint32) []uint64 {
	numCandidates := len(candidateOffsets) - 1
	numErrors := len(e2dOffsets) - 1
	results := make([]uint64, numCandidates)
	residual := make([]bool, numDetectors)

	for candidateID := 0; candidateID < numCandidates; candidateID++ {
		for index := range residual {
			residual[index] = false
		}
		begin := candidateOffsets[candidateID]
		end := candidateOffsets[candidateID+1]
		for pos := begin; pos < end; pos++ {
			residual[candidateDetectorIDs[pos]] = true
		}

		total := uint64(0)
		for pos := begin; pos < end; pos++ {
			detectorID := candidateDetectorIDs[pos]
			minimum := tesseractInfCost
			for errorPos := d2eOffsets[detectorID]; errorPos < d2eOffsets[detectorID+1]; errorPos++ {
				errorID := d2eErrorIDs[errorPos]
				if tessErrorIsBlocked(blockedWords, numErrors, candidateID, int(errorID)) {
					continue
				}
				overlap := uint32(0)
				for detectorPos := e2dOffsets[errorID]; detectorPos < e2dOffsets[errorID+1]; detectorPos++ {
					if residual[e2dDetectorIDs[detectorPos]] {
						overlap++
					}
				}
				if overlap == 0 {
					continue
				}
				cost := costLutValues[costLutOffsets[errorID]+overlap]
				if cost < minimum {
					minimum = cost
				}
			}
			if minimum != tesseractInfCost {
				minimumWithPenalty := uint64(minimum) + uint64(detectorPenaltyQ)
				if minimumWithPenalty >= uint64(tesseractInfCost) {
					minimum = tesseractInfCost
				} else {
					minimum = uint32(minimumWithPenalty)
				}
			}
			if minimum == tesseractInfCost || total+uint64(minimum) >= uint64(tesseractInfCost) {
				total = uint64(tesseractInfCost)
				break
			}
			total += uint64(minimum)
		}
		results[candidateID] = total
	}
	return results
}

func tessErrorIsBlocked(blockedWords []uint64, numErrors int, candidateID int,
	errorID int) bool {
	if len(blockedWords) == 0 {
		return false
	}
	wordsPerCandidate := (numErrors + 63) / 64
	wordIndex := candidateID*wordsPerCandidate + errorID/64
	return (blockedWords[wordIndex]>>(uint(errorID)&63))&1 != 0
}

func tessAppendU32Region(buffer *[]byte, values []uint32) uint32 {
	tessAlign8(buffer)
	offset := uint32(len(*buffer))
	for _, value := range values {
		*buffer = binary.LittleEndian.AppendUint32(*buffer, value)
	}
	tessAlign8(buffer)
	return offset
}

func tessAppendU64Region(buffer *[]byte, values []uint64) uint32 {
	tessAlign8(buffer)
	offset := uint32(len(*buffer))
	for _, value := range values {
		*buffer = binary.LittleEndian.AppendUint64(*buffer, value)
	}
	return offset
}

func tessAlign8(buffer *[]byte) {
	for len(*buffer)%8 != 0 {
		*buffer = append(*buffer, 0)
	}
}

func tessByteStream(data []byte) *encoding.ByteStream {
	stream := new(encoding.ByteStream)
	stream.Init()
	for _, value := range data {
		valueWord := new(word.Word)
		valueWord.Init(8)
		valueWord.SetValue(int64(value))
		stream.Merge(valueWord.ToByteStream())
	}
	return stream
}
