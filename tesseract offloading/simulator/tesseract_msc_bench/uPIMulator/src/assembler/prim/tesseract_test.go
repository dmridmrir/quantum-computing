package prim

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestTessEvaluateBatch(t *testing.T) {
	e2dOffsets := []uint32{0, 2, 4, 5}
	e2dDetectorIDs := []uint32{0, 1, 0, 2, 1}
	d2eOffsets, d2eErrorIDs := tessBuildD2E(3, 3, e2dOffsets, e2dDetectorIDs)
	costLutOffsets := []uint32{0, 3, 6, 8}
	costLutValues := []uint32{
		tesseractInfCost, 4, 2,
		tesseractInfCost, 6, 3,
		tesseractInfCost, 1,
	}
	candidateOffsets := []uint32{0, 2}
	candidateDetectorIDs := []uint32{0, 1}

	got := tessEvaluateBatch(
		3,
		d2eOffsets,
		d2eErrorIDs,
		e2dOffsets,
		e2dDetectorIDs,
		costLutOffsets,
		costLutValues,
		candidateOffsets,
		candidateDetectorIDs,
	)
	want := []uint64{3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("candidate cost = %v, want %v", got, want)
	}
}

func TestTessSortD2EByLowerBound(t *testing.T) {
	d2eOffsets := []uint32{0, 3}
	d2eErrorIDs := []uint32{0, 1, 2}
	e2dOffsets := []uint32{0, 1, 3, 7}
	errorCostsQ := []uint32{9, 10, 12}
	tessSortD2EByLowerBound(d2eOffsets, d2eErrorIDs, e2dOffsets, errorCostsQ)
	want := []uint32{2, 1, 0}
	if !reflect.DeepEqual(d2eErrorIDs, want) {
		t.Fatalf("lower-bound order = %v, want %v", d2eErrorIDs, want)
	}
}

func TestTessSyntheticTopologyCoversEveryDetector(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	e2dOffsets, e2dDetectorIDs := tessGenerateE2D(64, 128, rng)
	d2eOffsets, _ := tessBuildD2E(64, 128, e2dOffsets, e2dDetectorIDs)
	for detectorID := 0; detectorID < 64; detectorID++ {
		if d2eOffsets[detectorID] == d2eOffsets[detectorID+1] {
			t.Fatalf("detector %d has no incident error", detectorID)
		}
	}
}

func TestTessSyntheticWorkloadIsDeterministic(t *testing.T) {
	first := rand.New(rand.NewSource(7))
	second := rand.New(rand.NewSource(7))
	firstOffsets, firstIDs := tessGenerateE2D(32, 64, first)
	secondOffsets, secondIDs := tessGenerateE2D(32, 64, second)
	if !reflect.DeepEqual(firstOffsets, secondOffsets) || !reflect.DeepEqual(firstIDs, secondIDs) {
		t.Fatal("same seed produced different detector-error topology")
	}
}

func TestTessMRAMRegionsAreAligned(t *testing.T) {
	buffer := []byte{1, 2, 3}
	offset := tessAppendU32Region(&buffer, []uint32{1, 2, 3})
	if offset%8 != 0 {
		t.Fatalf("region offset %d is not 8-byte aligned", offset)
	}
	if len(buffer)%8 != 0 {
		t.Fatalf("region end %d is not 8-byte aligned", len(buffer))
	}
}

func TestTessCandidateResidualWords(t *testing.T) {
	offsets := []uint32{0, 3, 5}
	detectorIDs := []uint32{0, 31, 32, 1, 63}
	got := tessBuildCandidateResidualWords(2, 64, offsets, detectorIDs)
	want := []uint32{1 | 1<<31, 1, 2, 1 << 31}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("candidate residual words = %v, want %v", got, want)
	}
}

func TestTessCandidateResidualWordsPadsCandidateStrideForDMA(t *testing.T) {
	offsets := []uint32{0, 1, 2}
	detectorIDs := []uint32{31, 0}
	got := tessBuildCandidateResidualWords(2, 32, offsets, detectorIDs)
	want := []uint32{1 << 31, 0, 1, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DMA-aligned candidate residual words = %v, want %v", got, want)
	}
}

func TestTessRepackBlockedWords32PreservesCandidateBoundaries(t *testing.T) {
	blocked64 := []uint64{
		0x0123456789abcdef, 0x1111111122222222,
		0xfedcba9876543210, 0x3333333344444444,
	}
	got := tessRepackBlockedWords32(blocked64, 65, 2)
	want := []uint32{
		0x89abcdef, 0x01234567, 0x22222222,
		0x76543210, 0xfedcba98, 0x44444444,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("32-bit blocked words = %#v, want %#v", got, want)
	}
}

func TestTessBuildBlockedLayoutUsesAlignedSparseRecords(t *testing.T) {
	blocked64 := []uint64{
		1, 1 | uint64(1)<<63,
		2, 0,
	}
	dense, offsets, records := tessBuildBlockedLayout(blocked64, 65, 2)
	if dense != nil {
		t.Fatalf("dense fallback = %#v, want nil", dense)
	}
	wantOffsets := []uint32{0, 4}
	wantRecords := []uint32{2, 0, 64 << 16, 0, 1, 0, 1, 0}
	if !reflect.DeepEqual(offsets, wantOffsets) {
		t.Fatalf("sparse offsets = %#v, want %#v", offsets, wantOffsets)
	}
	if !reflect.DeepEqual(records, wantRecords) {
		t.Fatalf("sparse records = %#v, want %#v", records, wantRecords)
	}
}

func TestTessBuildBlockedLayoutFallsBackWhenSparseSetExceedsWRAMLimit(t *testing.T) {
	const numErrors = 300
	blocked64 := make([]uint64, (numErrors+63)/64)
	for errorID := 0; errorID <= tesseractMaxSparseBlocked; errorID++ {
		blocked64[errorID/64] |= uint64(1) << (uint(errorID) & 63)
	}
	dense, offsets, records := tessBuildBlockedLayout(blocked64, numErrors, 1)
	if len(dense) != (numErrors+31)/32 {
		t.Fatalf("dense fallback has %d words", len(dense))
	}
	if offsets != nil || records != nil {
		t.Fatalf("sparse layout should be disabled on fallback")
	}
}

func TestTessTraceV1PreservesBlockedErrorsAndDetectorPenalty(t *testing.T) {
	trace := tessTraceV1{
		Format:                     "tesseract-dpu-trace-v1",
		QBits:                      0,
		DetectorPenaltyQ:           2,
		NumDetectors:               2,
		ErrorDetectorOffsets:       []uint32{0, 2, 3},
		ErrorDetectorIDs:           []uint32{0, 1, 1},
		ErrorCostsQ:                []uint32{8, 6},
		CandidateDetectorOffsets:   []uint32{0, 2, 3},
		CandidateDetectorIDs:       []uint32{0, 1, 1},
		CandidateBlockedErrorWords: []uint64{0, 2},
		CandidateCostsQ:            []uint32{12, 10},
	}
	data, err := json.Marshal(trace)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "trace.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	workload := tessLoadTrace(path)
	d2eOffsets, d2eErrorIDs := tessBuildD2E(
		workload.numDetectors,
		workload.numErrors,
		workload.e2dOffsets,
		workload.e2dDetectorIDs,
	)
	costOffsets, costValues := tessBuildCostLUTFromBaseCosts(
		workload.numErrors,
		workload.e2dOffsets,
		workload.errorCostsQ,
	)
	got := tessEvaluateBatchWithOptions(
		workload.numDetectors,
		d2eOffsets,
		d2eErrorIDs,
		workload.e2dOffsets,
		workload.e2dDetectorIDs,
		costOffsets,
		costValues,
		workload.candidateOffsets,
		workload.candidateDetectorIDs,
		workload.blockedWords,
		workload.detectorPenaltyQ,
	)
	want := []uint64{12, 10}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("trace candidate costs = %v, want %v", got, want)
	}
}

func TestTessTraceV1RejectsWrongCandidateCostCount(t *testing.T) {
	trace := &tessTraceV1{
		Format:                   "tesseract-dpu-trace-v1",
		NumDetectors:             1,
		ErrorDetectorOffsets:     []uint32{0, 1},
		ErrorDetectorIDs:         []uint32{0},
		ErrorCostsQ:              []uint32{1},
		CandidateDetectorOffsets: []uint32{0, 1},
		CandidateDetectorIDs:     []uint32{0},
		CandidateCostsQ:          []uint32{1, 2},
	}
	if err := tessValidateTrace(trace); err == nil {
		t.Fatal("wrong candidate cost count was accepted")
	}
}

func TestTessVerifyProducerCostsRejectsMismatch(t *testing.T) {
	err := tessVerifyProducerCosts([]uint64{12, 10}, []uint32{12, 11})
	if err == nil {
		t.Fatal("producer/oracle cost mismatch was accepted")
	}
}

func TestTessTraceV1RejectsDuplicateCandidateDetectors(t *testing.T) {
	trace := &tessTraceV1{
		Format:                   "tesseract-dpu-trace-v1",
		NumDetectors:             2,
		ErrorDetectorOffsets:     []uint32{0, 1},
		ErrorDetectorIDs:         []uint32{0},
		ErrorCostsQ:              []uint32{1},
		CandidateDetectorOffsets: []uint32{0, 2},
		CandidateDetectorIDs:     []uint32{1, 1},
	}
	if err := tessValidateTrace(trace); err == nil {
		t.Fatal("duplicate candidate detector was accepted")
	}
}
