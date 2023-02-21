
def divide_work(workers, jobs_id):
    if workers >= len(jobs_id):
        return [[job_id] for job_id in jobs_id]


    jobs_per_worker = len(jobs_id) // workers
    work_division = []
    i = 0

    # Divide the work equally
    while i < len(jobs_id) and len(work_division) < workers:
        work_division.append([])
        while i < len(jobs_id) and len(work_division[-1]) != jobs_per_worker:
            work_division[-1].append(jobs_id[i])
            i += 1
    
    # If we can't divide the work equally some workers get more jobs
    if len(work_division) != len(jobs_id):
        i, j = 0, len(work_division) * jobs_per_worker
        while j < len(jobs_id):
            work_division[i].append(jobs_id[j])
            i += 1
            j += 1
    
    return work_division
    
if __name__ == "__main__":
    print(divide_work(8, list(range(2))))
    print(divide_work(8, list(range(8))))
    print(divide_work(8, list(range(20))))
    print(divide_work(2, list(range(20))))
    print(divide_work(3, list(range(20))))
    print(divide_work(4, list(range(20))))
    print(divide_work(5, list(range(20))))
    print(divide_work(3, list(range(1))))
    print(divide_work(5, list(range(3))))