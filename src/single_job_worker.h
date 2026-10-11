#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace qwen_native {

// One producer, one pending/in-flight job. Call wait() to observe job failures.
// Jobs must not recursively submit/wait on this same worker. drain() and
// destruction finish accepted work without throwing, so owners may release
// captured buffers only after draining the worker.
class SingleJobWorker {
public:
    SingleJobWorker() : thread_([this] { run(); }) {}
    SingleJobWorker(const SingleJobWorker &) = delete;
    SingleJobWorker &operator=(const SingleJobWorker &) = delete;

    ~SingleJobWorker() {
        drain();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        start_.notify_one();
        thread_.join();
    }

    void submit(std::function<void()> job) {
        if (!job) throw std::invalid_argument("worker job is empty");
        std::lock_guard<std::mutex> lock(mutex_);
        if (!done_) throw std::logic_error("worker already has an in-flight job");
        if (error_) std::rethrow_exception(error_);
        job_ = std::move(job);
        done_ = false;
        start_.notify_one();
    }

    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return done_; });
    }

    void wait() {
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            finished_.wait(lock, [this] { return done_; });
            error = std::exchange(error_, nullptr);
        }
        if (error) std::rethrow_exception(error);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            start_.wait(lock, [this] { return stop_ || !done_; });
            if (stop_) return;
            auto job = std::move(job_);
            lock.unlock();
            std::exception_ptr error;
            try {
                job();
            } catch (...) {
                error = std::current_exception();
            }
            job = {}; // Release captures before publishing buffer ownership.
            lock.lock();
            error_ = error;
            done_ = true;
            finished_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable finished_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool done_ = true;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace qwen_native
