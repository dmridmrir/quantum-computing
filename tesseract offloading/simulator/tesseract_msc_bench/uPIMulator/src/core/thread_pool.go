package core

import (
	"errors"
	"sync"
)

type ThreadPool struct {
	num_threads  int
	channel_size int

	jobs []Job

	wg sync.WaitGroup
}

func (this *ThreadPool) Init(num_threads int) {
	if num_threads <= 0 {
		err := errors.New("num threads <= 0")
		panic(err)
	}

	this.num_threads = num_threads

	this.jobs = make([]Job, 0)
}

func (this *ThreadPool) Enque(job Job) {
	this.wg.Add(1)
	this.jobs = append(this.jobs, job)
}

func (this *ThreadPool) Start() {
	if len(this.jobs) == 0 {
		return
	}

	// num_threads is a host-side simulator limit, not a modeled DPU tasklet
	// count.  The previous implementation launched one goroutine per DPU and
	// silently ignored the configured limit, which made 8/16-thread scale-out
	// experiments incomparable.
	jobs := make(chan Job)
	workers := this.num_threads
	if workers > len(this.jobs) {
		workers = len(this.jobs)
	}
	for worker := 0; worker < workers; worker++ {
		go func() {
			for job := range jobs {
				this.Dispatch(job)
			}
		}()
	}
	for _, job := range this.jobs {
		jobs <- job
	}
	close(jobs)
	this.wg.Wait()
}

func (this *ThreadPool) Dispatch(job Job) {
	defer this.wg.Done()
	job.Execute()
}
