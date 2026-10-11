#include "single_job_worker.h"

#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void failure_and_reuse_test() {
    qwen_native::SingleJobWorker worker;
    worker.wait();
    worker.submit([] { throw std::runtime_error("expected job failure"); });
    worker.drain();
    bool submit_failed = false;
    try {
        worker.submit([] {});
    } catch (const std::runtime_error &error) {
        submit_failed = std::string(error.what()) == "expected job failure";
    }
    require(submit_failed, "unobserved worker failure silently overwritten");
    bool wait_failed = false;
    try {
        worker.wait();
    } catch (const std::runtime_error &error) {
        wait_failed = std::string(error.what()) == "expected job failure";
    }
    require(wait_failed, "worker failure did not reach caller");
    int result = 0;
    for (int index = 0; index < 32; ++index) {
        worker.submit([&result] { ++result; });
        worker.wait();
    }
    require(result == 32, "worker reuse lost an accepted job");
    bool empty_failed = false;
    try {
        worker.submit({});
    } catch (const std::invalid_argument &) {
        empty_failed = true;
    }
    require(empty_failed, "empty worker job accepted");
}

void overlap_test() {
    qwen_native::SingleJobWorker worker;
    std::promise<void> started;
    std::promise<void> released;
    auto release = released.get_future();
    worker.submit([&] {
        started.set_value();
        release.wait();
    });
    started.get_future().wait();
    bool rejected = false;
    try {
        worker.submit([] {});
    } catch (const std::logic_error &) {
        rejected = true;
    }
    released.set_value(); // Always release the first job before assertions.
    worker.wait();
    require(rejected, "in-flight job was overwritten");
}

void lifetime_test() {
    int completed = 0;
    {
        qwen_native::SingleJobWorker worker;
        worker.submit([&] { completed = 1; });
    }
    require(completed == 1, "destruction discarded a queued/in-flight job");

    struct Capture {
        std::atomic<int> &released;
        explicit Capture(std::atomic<int> &counter) : released(counter) {}
        ~Capture() { ++released; }
    };
    std::atomic<int> released = 0;
    qwen_native::SingleJobWorker worker;
    auto capture = std::make_shared<Capture>(released);
    worker.submit([capture] {});
    capture.reset();
    worker.wait();
    require(released == 1, "wait returned before job captures were released");
}

} // namespace

int main() {
    try {
        failure_and_reuse_test();
        overlap_test();
        lifetime_test();
        std::cout << "single-job worker: error propagation, reuse, overlap refusal, "
                     "destruction drain and capture lifetimes passed (CPU only)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
