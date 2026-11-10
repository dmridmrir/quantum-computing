package core

import (
	"sync/atomic"
	"testing"
)

type concurrencyJob struct {
	active    *int32
	maximum   *int32
	started   chan<- struct{}
	release   <-chan struct{}
	completed *int32
}

func (job *concurrencyJob) Execute() {
	active := atomic.AddInt32(job.active, 1)
	for {
		maximum := atomic.LoadInt32(job.maximum)
		if active <= maximum || atomic.CompareAndSwapInt32(job.maximum, maximum, active) {
			break
		}
	}
	job.started <- struct{}{}
	<-job.release
	atomic.AddInt32(job.active, -1)
	atomic.AddInt32(job.completed, 1)
}

func TestThreadPoolHonorsWorkerLimit(t *testing.T) {
	const workerCount = 3
	const jobCount = 9
	var active int32
	var maximum int32
	var completed int32
	started := make(chan struct{}, jobCount)
	release := make(chan struct{})

	pool := new(ThreadPool)
	pool.Init(workerCount)
	for index := 0; index < jobCount; index++ {
		pool.Enque(&concurrencyJob{
			active:    &active,
			maximum:   &maximum,
			started:   started,
			release:   release,
			completed: &completed,
		})
	}

	done := make(chan struct{})
	go func() {
		pool.Start()
		close(done)
	}()
	for index := 0; index < workerCount; index++ {
		<-started
	}
	if got := atomic.LoadInt32(&maximum); got != workerCount {
		t.Fatalf("maximum concurrent jobs = %d, want %d", got, workerCount)
	}
	close(release)
	<-done
	if got := atomic.LoadInt32(&completed); got != jobCount {
		t.Fatalf("completed jobs = %d, want %d", got, jobCount)
	}
}
