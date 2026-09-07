// End-to-end cases for selecting the UCX exchange at runtime rather than from
// the plan. One function per case; LAB_ONLY=<letter> runs just one.

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cudf/contiguous_split.hpp>

#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/CudfPackedPage.h"
#include "velox/experimental/ucx-exchange/DynamicUcxOutputBufferReader.h"
#include "velox/experimental/ucx-exchange/DynamicUcxTransport.h"
#include "velox/experimental/ucx-exchange/ExchangeFormatRegistry.h"
#include "velox/experimental/ucx-exchange/UcxCudfDriverAdapter.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxQueues.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::ucx_exchange;

namespace {

constexpr vector_size_t kNumRows{20'000};
constexpr int64_t kExpectedSum{
    static_cast<int64_t>(kNumRows) * (kNumRows - 1) / 2};

uint16_t gListenerPort{21346};
int gFailures{0};

void check(bool condition, std::string_view what) {
  std::cout << "  [" << (condition ? "PASS" : "FAIL") << "] " << what << "\n";
  if (!condition) {
    ++gFailures;
  }
}

std::vector<std::string> operatorTypes(const std::shared_ptr<Task>& task) {
  std::vector<std::string> types;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      types.push_back(op.operatorType);
    }
  }
  return types;
}

bool contains(const std::vector<std::string>& types, std::string_view name) {
  return std::find(types.begin(), types.end(), name) != types.end();
}

void printOps(const char* label, const std::vector<std::string>& ops) {
  std::cout << "  " << label << ":";
  for (const auto& op : ops) {
    std::cout << " " << op;
  }
  std::cout << "\n";
}

// The URL shape UcxExchangeSource parses a task and partition out of.
Split remoteSplit(const std::string& taskId, int partition) {
  return Split(
      std::make_shared<RemoteConnectorSplit>(fmt::format(
          "http://127.0.0.1:{}/v1/task/{}/results/{}",
          gListenerPort - 3,
          taskId,
          partition)));
}

RowVectorPtr makeInput(memory::MemoryPool* pool) {
  facebook::velox::test::VectorMaker maker{pool};
  return maker.rowVector(
      {"id"},
      {maker.flatVector<int64_t>(kNumRows, [](auto row) { return row; })});
}

struct QueryResult {
  int64_t rows{0};
  int64_t checksum{0};
  std::vector<std::string> producerOps;
  std::vector<std::string> consumerOps;
};

// Starts a cuDF producer. 'nameTransport' writes TransportKind::kUcx into the
// plan, which is the only thing that differs between the two paths.
std::shared_ptr<Task> startProducer(
    const std::string& taskId,
    const RowVectorPtr& input,
    bool nameTransport,
    bool onGpu,
    folly::Executor* executor) {
  PlanBuilder builder;
  builder.values({input});
  if (onGpu) {
    // A real cuDF operator, so the chain ends on the device.
    builder.filter(fmt::format("id < {}", kNumRows));
  }
  if (nameTransport) {
    builder.partitionedOutput(
        /*keys=*/{},
        /*numPartitions=*/1,
        /*outputLayout=*/{},
        /*serdeKind=*/"Presto",
        std::string{core::TransportKind::kUcx});
  } else {
    builder.partitionedOutput(/*keys=*/{}, /*numPartitions=*/1);
  }
  auto queryCtx = onGpu
      ? core::QueryCtx::create(executor)
      : core::QueryCtx::create(
            executor,
            core::QueryConfig{
                {{std::string(cudf_velox::CudfConfig::kCudfEnabled),
                  "false"}}});

  auto task = Task::create(
      taskId,
      builder.planFragment(),
      /*destination=*/0,
      std::move(queryCtx),
      Task::ExecutionMode::kParallel);
  task->start(/*maxDrivers=*/1);
  return task;
}

// Runs one producer into one consumer across the UCX exchange and sums the id
// column, whichever form the rows arrive in.
QueryResult runExchange(
    const std::string& taskId,
    bool nameTransport,
    bool producerOnGpu,
    bool consumerUsesGpu,
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  auto input = makeInput(pool);
  auto producer =
      startProducer(taskId, input, nameTransport, producerOnGpu, executor);

  // The consumer names the transport only when the producer did: with
  // discovery on, neither end says anything about transports.
  core::PlanNodeId exchangeId;
  PlanBuilder consumerBuilder;
  if (nameTransport) {
    consumerBuilder.exchange(
        asRowType(input->type()),
        "Presto",
        std::string{core::TransportKind::kUcx});
  } else {
    consumerBuilder.exchange(asRowType(input->type()), "Presto");
  }
  consumerBuilder.capturePlanNodeId(exchangeId);
  if (consumerUsesGpu) {
    // Puts a cuDF operator between the exchange and the sink, so the data has
    // to be on the device by then whichever way it arrived.
    consumerBuilder.filter(fmt::format("id < {}", kNumRows / 2));
  }
  auto consumerPlan = consumerBuilder.planFragment();

  std::atomic<int64_t> rows{0};
  std::atomic<int64_t> checksum{0};
  auto collect = [&](RowVectorPtr data, bool, ContinueFuture*) {
    if (data == nullptr) {
      return BlockingReason::kNotBlocked;
    }
    int64_t sum{0};
    vector_size_t numRows{0};
    if (auto* cudfVector = dynamic_cast<cudf_velox::CudfVector*>(data.get())) {
      auto view = cudfVector->getTableView();
      numRows = view.num_rows();
      std::vector<int64_t> host(numRows, 0);
      cudaMemcpy(
          host.data(),
          view.column(0).data<int64_t>(),
          numRows * sizeof(int64_t),
          cudaMemcpyDeviceToHost);
      for (auto value : host) {
        sum += value;
      }
    } else {
      numRows = data->size();
      auto* ids = data->childAt(0)->as<SimpleVector<int64_t>>();
      for (vector_size_t i = 0; i < numRows; ++i) {
        sum += ids->valueAt(i);
      }
    }
    rows.fetch_add(numRows, std::memory_order_relaxed);
    checksum.fetch_add(sum, std::memory_order_relaxed);
    return BlockingReason::kNotBlocked;
  };

  auto consumer = Task::create(
      taskId + "-consumer",
      consumerPlan,
      /*destination=*/0,
      core::QueryCtx::create(executor),
      Task::ExecutionMode::kParallel,
      collect);
  consumer->start(/*maxDrivers=*/1);

  QueryResult result;
  result.producerOps = operatorTypes(producer);
  result.consumerOps = operatorTypes(consumer);

  consumer->addSplit(exchangeId, remoteSplit(taskId, 0));
  consumer->noMoreSplits(exchangeId);
  consumer->taskCompletionFuture().wait(std::chrono::seconds(60));

  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));

  result.rows = rows.load();
  result.checksum = checksum.load();
  return result;
}

// A. The path as it stands: the plan names the transport, discovery is off.
// Here to show the switch changes nothing when it is not set.
void scenarioPlanDeclared(memory::MemoryPool* pool, folly::Executor* executor) {
  std::cout << "\n== A. transport named in the plan, discovery off ==\n";
  enableDynamicUcx(false);

  auto result = runExchange(
      "declared-producer",
      /*nameTransport=*/true,
      /*producerOnGpu=*/true,
      /*consumerUsesGpu=*/false,
      pool,
      executor);
  printOps("producer ops", result.producerOps);
  printOps("consumer ops", result.consumerOps);
  check(
      contains(result.producerOps, "cudfPartitionedOutput"),
      "producer ran the cuDF UCX output");
  check(
      contains(result.consumerOps, "UcxExchange"), "consumer ran UcxExchange");
  check(result.rows == kNumRows, "every row arrived");
  check(result.checksum == kExpectedSum, "values are correct");
}

// B. Residency matrix: host/gpu producer against host/gpu consumer. Nothing
// should convert merely to make the transport work.
void scenarioResidency(
    const char* label,
    const std::string& taskId,
    bool producerOnGpu,
    bool consumerUsesGpu,
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== B. " << label << " ==\n";
  enableDynamicUcx(true);

  auto result = runExchange(
      taskId,
      /*nameTransport=*/false,
      producerOnGpu,
      consumerUsesGpu,
      pool,
      executor);
  printOps("producer ops", result.producerOps);
  printOps("consumer ops", result.consumerOps);

  std::cout << "  rows " << result.rows << ", checksum " << result.checksum
            << "\n";

  const int64_t expectedRows = consumerUsesGpu ? kNumRows / 2 : kNumRows;
  const int64_t expectedSum =
      consumerUsesGpu ? (expectedRows * (expectedRows - 1)) / 2 : kExpectedSum;

  if (producerOnGpu) {
    check(
        contains(result.producerOps, "cudfPartitionedOutput"),
        "producer packed on the device");
  } else {
    check(
        contains(result.producerOps, "PartitionedOutput") &&
            !contains(result.producerOps, "cudfPartitionedOutput"),
        "producer stayed on the host, nothing uploaded to send");
  }
  check(
      contains(result.consumerOps, "UcxExchange"),
      "UCX carried it, with no transport in the plan");
  if (consumerUsesGpu) {
    check(
        contains(result.consumerOps, "CudfFilterProject"),
        "a cuDF operator ran on the data");
  }
  check(result.rows == expectedRows, "every expected row arrived");
  check(result.checksum == expectedSum, "values are correct");
}

// C. A reader that never asked over UCX gets ordinary Presto bytes.
void scenarioHostOnlyReader(
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== C. the same pages read by a host-only reader ==\n";
  enableDynamicUcx(true);

  auto input = makeInput(pool);
  auto producer = startProducer(
      "host-read-producer",
      input,
      /*nameTransport=*/false,
      /*onGpu=*/true,
      executor);

  auto reader = std::make_shared<DynamicUcxOutputBufferReader>(
      exec::DefaultOutputBufferManager::getInstanceRef(),
      "host-read-producer",
      /*destination=*/0);

  std::vector<std::unique_ptr<folly::IOBuf>> collected;
  bool atEnd{false};
  for (int attempt = 0; attempt < 400 && !atEnd; ++attempt) {
    std::atomic<bool> done{false};
    reader->request(1 << 20, [&](DynamicUcxOutputBufferReader::Data data) {
      for (auto& page : data.pages) {
        collected.push_back(std::move(page));
      }
      atEnd = data.atEnd;
      done.store(true);
    });
    while (!done.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  int64_t rows{0};
  int64_t checksum{0};
  bool allHost{true};
  auto* serde =
      getNamedVectorSerde(VectorSerde::kindName(VectorSerde::Kind::kPresto));
  for (auto& iobuf : collected) {
    // Ask CUDA where each segment lives; chain length says nothing.
    const auto* segment = iobuf.get();
    do {
      cudaPointerAttributes attributes{};
      if (cudaPointerGetAttributes(&attributes, segment->data()) !=
          cudaSuccess) {
        cudaGetLastError();
      } else if (attributes.type == cudaMemoryTypeDevice) {
        allHost = false;
      }
      segment = segment->next();
    } while (segment != iobuf.get());

    auto page = std::make_unique<exec::PrestoSerializedPage>(std::move(iobuf));
    auto stream = page->prepareStreamForDeserialize();
    RowVectorPtr result;
    while (!stream->atEnd()) {
      VectorStreamGroup::read(
          stream.get(),
          pool,
          asRowType(input->type()),
          serde,
          &result,
          nullptr);
      auto* ids = result->childAt(0)->as<SimpleVector<int64_t>>();
      for (vector_size_t i = 0; i < result->size(); ++i) {
        checksum += ids->valueAt(i);
      }
      rows += result->size();
    }
  }

  std::cout << "  pages " << collected.size() << ", rows " << rows
            << ", checksum " << checksum << "\n";
  check(allHost, "pages rendered as host bytes, not device memory");
  check(rows == kNumRows, "every row came back through the host path");
  check(checksum == kExpectedSum, "host-rendered values are correct");
  // A host reader never announces itself; the first render is what marks it.
  check(
      ExchangeFormatRegistry::instance().isVeloxExchange(
          "host-read-producer", 0),
      "the producer learned this destination reads host bytes");

  // The buffer references the task, so let the producer finish first.
  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
  reader->close();
  reader->deleteResults();

  // No exchange server exists on this path, so the producer's own completion
  // hook must be what releases the registration.
  auto* queueManager = UcxOutputQueueManager::getInstanceRef().get();
  for (int attempt = 0; attempt < 200 &&
       queueManager->getQueueIfExists("host-read-producer") != nullptr;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  check(
      queueManager->getQueueIfExists("host-read-producer") == nullptr,
      "the producer released its own registration with no server to do it");
}

// D. A non-cuDF query in the same process is left untouched.
void scenarioNonCudfQueryUntouched(
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== D. a non-cuDF query in the same process ==\n";
  enableDynamicUcx(true);

  auto input = makeInput(pool);
  auto plan = PlanBuilder()
                  .values({input})
                  .partitionedOutput(/*keys=*/{}, /*numPartitions=*/1)
                  .planFragment();
  auto task = Task::create(
      "cpu-only-producer",
      plan,
      /*destination=*/0,
      core::QueryCtx::create(
          executor,
          core::QueryConfig{
              {{std::string(cudf_velox::CudfConfig::kCudfEnabled), "false"}}}),
      Task::ExecutionMode::kParallel);
  task->start(/*maxDrivers=*/1);

  auto ops = operatorTypes(task);
  printOps("producer ops", ops);
  check(
      contains(ops, "PartitionedOutput"),
      "producer kept the standard PartitionedOutput");
  check(
      !contains(ops, "cudfPartitionedOutput"),
      "the cuDF output operator was not forced on it");
  check(!contains(ops, "CudfFromVelox"), "nothing was uploaded to the device");

  task->requestCancel();
  task->taskCompletionFuture().wait(std::chrono::seconds(30));
}

// K. A consumer cancelled mid-stream. Checks the teardown paths; does not
// cover failover, which a cancel legitimately ends.
void scenarioConsumerDiesMidStream(
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== K. a consumer dies mid-stream, another takes over ==\n";
  enableDynamicUcx(true);

  const std::string taskId{"failover-producer"};
  auto input = makeInput(pool);
  // Enough batches, against a small buffer, that the producer is still holding
  // pages when the first consumer goes away.
  constexpr int kBatches = 12;
  std::vector<RowVectorPtr> inputs(kBatches, input);
  auto plan = PlanBuilder()
                  .values(inputs)
                  .partitionedOutput(/*keys=*/{}, /*numPartitions=*/1)
                  .planFragment();
  auto producer = Task::create(
      taskId,
      plan,
      /*destination=*/0,
      core::QueryCtx::create(
          executor,
          core::QueryConfig{
              {{core::QueryConfig::kMaxOutputBufferSize, "16384"},
               {core::QueryConfig::kMaxPartitionedOutputBufferSize, "16384"}}}),
      Task::ExecutionMode::kParallel);
  producer->start(/*maxDrivers=*/1);

  auto runConsumer =
      [&](const std::string& name, std::atomic<int64_t>& rows, bool slow) {
        core::PlanNodeId exchangeId;
        auto consumerPlan = PlanBuilder()
                                .exchange(asRowType(input->type()), "Presto")
                                .capturePlanNodeId(exchangeId)
                                .planFragment();
        auto collect = [&rows, slow](RowVectorPtr data, bool, ContinueFuture*) {
          if (data != nullptr) {
            if (slow) {
              // So there is a mid-stream to cancel in.
              std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            rows.fetch_add(data->size(), std::memory_order_relaxed);
          }
          return BlockingReason::kNotBlocked;
        };
        auto consumer = Task::create(
            name,
            consumerPlan,
            /*destination=*/0,
            core::QueryCtx::create(executor),
            Task::ExecutionMode::kParallel,
            collect);
        consumer->start(/*maxDrivers=*/1);
        consumer->addSplit(exchangeId, remoteSplit(taskId, 0));
        consumer->noMoreSplits(exchangeId);
        return consumer;
      };

  std::atomic<int64_t> firstRows{0};
  auto first = runConsumer("failover-consumer-1", firstRows, /*slow=*/true);
  for (int attempt = 0; attempt < 400 && firstRows.load() == 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  printOps("producer ops", operatorTypes(producer));
  printOps("first consumer ops", operatorTypes(first));
  const int64_t readBeforeAbort = firstRows.load();
  std::cout << "  first consumer read " << readBeforeAbort << " rows\n";
  check(readBeforeAbort > 0, "the first consumer got something to lose");
  check(
      readBeforeAbort < kNumRows * kBatches,
      "and was cancelled before the stream ended");

  first->requestCancel();
  first->taskCompletionFuture().wait(std::chrono::seconds(30));
  first.reset();

  // Cancelling a consumer ends the producer but must not fail it.
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
  std::cout << "  producer ended in state "
            << taskStateString(producer->state()) << "\n";
  check(
      producer->errorMessage().empty(),
      "the consumer going away did not fail the producer");

  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
  producer.reset();

  // Everything the transport held for this task is gone.
  auto* queueManager = UcxOutputQueueManager::getInstanceRef().get();
  for (int attempt = 0;
       attempt < 400 && queueManager->getQueueIfExists(taskId) != nullptr;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  check(
      queueManager->getQueueIfExists(taskId) == nullptr,
      "the task registration was released after the abort");
  check(
      !ExchangeFormatRegistry::instance().isVeloxExchange(taskId, 0) &&
          !ExchangeFormatRegistry::instance().iscuDFExchange(taskId, 0),
      "the reader record was erased after the abort");
}

// J. Broadcast. One page reaches every destination, so both consumers must be
// able to read it.
void scenarioBroadcast(memory::MemoryPool* pool, folly::Executor* executor) {
  constexpr int kConsumers = 2;
  std::cout << "\n== J. broadcast to " << kConsumers << " consumers ==\n";
  enableDynamicUcx(true);

  const std::string taskId{"broadcast-producer"};
  auto input = makeInput(pool);
  auto plan =
      PlanBuilder().values({input}).partitionedOutputBroadcast().planFragment();
  auto producer = Task::create(
      taskId,
      plan,
      /*destination=*/0,
      core::QueryCtx::create(executor),
      Task::ExecutionMode::kParallel);
  producer->start(/*maxDrivers=*/1);
  producer->updateOutputBuffers(kConsumers, /*noMoreBuffers=*/true);

  std::array<std::atomic<int64_t>, kConsumers> perConsumer{};
  std::vector<std::shared_ptr<Task>> consumers;
  for (int destination = 0; destination < kConsumers; ++destination) {
    core::PlanNodeId exchangeId;
    auto consumerPlan = PlanBuilder()
                            .exchange(asRowType(input->type()), "Presto")
                            .capturePlanNodeId(exchangeId)
                            .planFragment();
    auto collect = [&, destination](RowVectorPtr data, bool, ContinueFuture*) {
      if (data != nullptr) {
        perConsumer[destination].fetch_add(
            data->size(), std::memory_order_relaxed);
      }
      return BlockingReason::kNotBlocked;
    };
    auto consumer = Task::create(
        fmt::format("{}-consumer-{}", taskId, destination),
        consumerPlan,
        destination,
        core::QueryCtx::create(executor),
        Task::ExecutionMode::kParallel,
        collect);
    consumer->start(/*maxDrivers=*/1);
    consumer->addSplit(exchangeId, remoteSplit(taskId, destination));
    consumer->noMoreSplits(exchangeId);
    consumers.push_back(std::move(consumer));
  }

  for (auto& consumer : consumers) {
    consumer->taskCompletionFuture().wait(std::chrono::seconds(60));
  }
  bool everyoneGotAll = true;
  for (int destination = 0; destination < kConsumers; ++destination) {
    std::cout << "  consumer " << destination << " rows "
              << perConsumer[destination].load() << "\n";
    everyoneGotAll &= perConsumer[destination].load() == kNumRows;
  }
  check(everyoneGotAll, "every consumer read the whole broadcast");

  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
}

// H. A plan that still names kUcx with discovery on. Task builds the output
// operator from the registry, but the pages go to the ordinary buffer.
void scenarioPlanNamesUcxWithDiscovery(
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== H. plan still names the transport, discovery on ==\n";
  enableDynamicUcx(true);

  auto result = runExchange(
      "named-with-discovery-producer",
      /*nameTransport=*/true,
      /*producerOnGpu=*/true,
      /*consumerUsesGpu=*/false,
      pool,
      executor);
  printOps("producer ops", result.producerOps);
  printOps("consumer ops", result.consumerOps);
  check(result.rows == kNumRows, "every row arrived");
  check(result.checksum == kExpectedSum, "values are correct");
  // Task pointed its teardown at the manager kUcx named, so whoever set this
  // buffer up has to retire it.
  check(
      exec::DefaultOutputBufferManager::getInstanceRef()->getBufferIfExists(
          "named-with-discovery-producer") == nullptr,
      "the ordinary output buffer was retired with the task");
}

// G. A packed table with no columns, as a count(*) shuffle produces. It has no
// device segment, so shape alone cannot tell it from a serialized page.
void scenarioZeroColumnPage(memory::MemoryPool* pool) {
  std::cout << "\n== G. a packed table with no columns ==\n";

  const std::string taskId{"zero-column-page"};
  // track() before record(), the way the driver adapter and the handshake do
  // it: a record for a task nobody is following is dropped.
  ExchangeFormatRegistry::instance().trackTask(taskId);
  ExchangeFormatRegistry::instance().setExchangeFormat(
      taskId, 0, ExchangeFormatRegistry::Format::kCudf);

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto packed = std::make_unique<cudf::packed_columns>(
      cudf::pack(cudf::table_view{}, stream, cudf_velox::get_temp_mr()));
  stream.synchronize();

  CudfPackedPage page(
      std::move(packed),
      taskId,
      /*destination=*/0,
      /*numRows=*/kNumRows,
      // The truthful type for a table with no columns.
      ROW({}, {}),
      "Presto",
      /*sharedAcrossDestinations=*/false,
      pool);

  std::unique_ptr<folly::IOBuf> rendered{page.getIOBuf()};
  const auto segments = rendered->countChainElements();
  uint32_t magic = 0;
  if (rendered->length() >= sizeof(magic)) {
    std::memcpy(&magic, rendered->data(), sizeof(magic));
  }
  std::cout << "  segments " << segments << ", magic "
            << (magic == kCudfPackedPageMagic ? "present" : "absent") << "\n";
  check(
      magic == kCudfPackedPageMagic,
      "the page says it is a packed table rather than leaving it to be guessed");

  UcxGpuPayload payload;
  payload.deviceResident = true;
  payload.buffer = std::shared_ptr<folly::IOBuf>(std::move(rendered));
  check(payload.metadata() != nullptr, "its metadata is reachable");
  check(payload.payloadBytes() == 0, "it reports no device bytes");
  check(
      payload.payload() == nullptr,
      "and no device pointer, rather than aliasing its own header");

  // The same table read by a consumer that never announced itself.
  const std::string hostTaskId{"zero-column-page-host"};
  auto emptyRowType = ROW({}, {});
  auto hostPacked = std::make_unique<cudf::packed_columns>(
      cudf::pack(cudf::table_view{}, stream, cudf_velox::get_temp_mr()));
  stream.synchronize();

  CudfPackedPage hostPage(
      std::move(hostPacked),
      hostTaskId,
      /*destination=*/0,
      /*numRows=*/kNumRows,
      emptyRowType,
      "Presto",
      /*sharedAcrossDestinations=*/false,
      pool);

  auto hostRendered = std::unique_ptr<folly::IOBuf>(hostPage.getIOBuf());
  uint32_t hostMagic = 0;
  if (hostRendered->length() >= sizeof(hostMagic)) {
    std::memcpy(&hostMagic, hostRendered->data(), sizeof(hostMagic));
  }
  check(
      hostMagic != kCudfPackedPageMagic,
      "an unannounced reader gets host bytes, not a packed table");

  auto* serde = getNamedVectorSerde("Presto");
  auto hostSerialized =
      std::make_unique<exec::PrestoSerializedPage>(std::move(hostRendered));
  auto hostStream = hostSerialized->prepareStreamForDeserialize();
  int64_t hostRows{0};
  RowVectorPtr hostResult;
  while (!hostStream->atEnd()) {
    VectorStreamGroup::read(
        hostStream.get(), pool, emptyRowType, serde, &hostResult, nullptr);
    hostRows += hostResult->size();
  }
  std::cout << "  host-rendered rows " << hostRows << "\n";
  check(
      hostRows == kNumRows,
      "the row count survives the host render, with no column to carry it");
  ExchangeFormatRegistry::instance().forgetTask(hostTaskId);
  ExchangeFormatRegistry::instance().forgetTask(taskId);
}

// F. Several partitions, which take different enqueue sites than one.
void scenarioMultiplePartitions(
    const char* label,
    const std::string& taskId,
    int64_t maxOutputBufferBytes,
    bool startConsumersLate,
    int numInputBatches,
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  constexpr int kPartitions = 4;
  std::cout << "\n== " << label << " ==\n";
  enableDynamicUcx(true);

  auto input = makeInput(pool);
  // Several batches so addInput() is called again while the buffer is still
  // full, which is where a dropped blocking reason shows up.
  std::vector<RowVectorPtr> inputs(numInputBatches, input);
  auto plan = PlanBuilder()
                  .values(inputs)
                  .partitionedOutput({"id"}, kPartitions)
                  .planFragment();
  auto producer = Task::create(
      taskId,
      plan,
      /*destination=*/0,
      core::QueryCtx::create(
          executor,
          core::QueryConfig{
              {// What the buffer itself will hold before it pushes back.
               {core::QueryConfig::kMaxOutputBufferSize,
                std::to_string(maxOutputBufferBytes)},
               // And how much of that one page may be, so the stream is many
               // pages rather than one per partition.
               {core::QueryConfig::kMaxPartitionedOutputBufferSize,
                std::to_string(maxOutputBufferBytes)}}}),
      Task::ExecutionMode::kParallel);
  producer->start(/*maxDrivers=*/1);

  if (startConsumersLate) {
    // Let the producer hit the limit before anybody drains.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  std::atomic<int64_t> rows{0};
  std::atomic<int64_t> checksum{0};
  std::array<std::atomic<int64_t>, kPartitions> perPartition{};
  std::vector<std::shared_ptr<Task>> consumers;
  for (int partition = 0; partition < kPartitions; ++partition) {
    core::PlanNodeId exchangeId;
    auto consumerPlan = PlanBuilder()
                            .exchange(asRowType(input->type()), "Presto")
                            .capturePlanNodeId(exchangeId)
                            .planFragment();
    auto collect = [&, partition](RowVectorPtr data, bool, ContinueFuture*) {
      if (data == nullptr) {
        return BlockingReason::kNotBlocked;
      }
      int64_t sum{0};
      vector_size_t numRows{0};
      if (auto* cudfVector =
              dynamic_cast<cudf_velox::CudfVector*>(data.get())) {
        auto view = cudfVector->getTableView();
        numRows = view.num_rows();
        std::vector<int64_t> host(numRows, 0);
        cudaMemcpy(
            host.data(),
            view.column(0).data<int64_t>(),
            numRows * sizeof(int64_t),
            cudaMemcpyDeviceToHost);
        for (auto value : host) {
          sum += value;
        }
      } else {
        numRows = data->size();
        auto* ids = data->childAt(0)->as<SimpleVector<int64_t>>();
        for (vector_size_t i = 0; i < numRows; ++i) {
          sum += ids->valueAt(i);
        }
      }
      rows.fetch_add(numRows, std::memory_order_relaxed);
      checksum.fetch_add(sum, std::memory_order_relaxed);
      perPartition[partition].fetch_add(numRows, std::memory_order_relaxed);
      return BlockingReason::kNotBlocked;
    };

    auto consumer = Task::create(
        fmt::format("{}-consumer-{}", taskId, partition),
        consumerPlan,
        partition,
        core::QueryCtx::create(executor),
        Task::ExecutionMode::kParallel,
        collect);
    consumer->start(/*maxDrivers=*/1);
    consumer->addSplit(exchangeId, remoteSplit(taskId, partition));
    consumer->noMoreSplits(exchangeId);
    consumers.push_back(std::move(consumer));
  }

  for (auto& consumer : consumers) {
    consumer->taskCompletionFuture().wait(std::chrono::seconds(60));
  }
  for (int partition = 0; partition < kPartitions; ++partition) {
    std::cout << "  partition " << partition << " rows "
              << perPartition[partition].load() << "\n";
  }
  std::cout << "  rows " << rows.load() << ", checksum " << checksum.load()
            << "\n";
  check(
      rows.load() == kNumRows * numInputBatches,
      "every row arrived across all partitions");
  check(
      checksum.load() == kExpectedSum * numInputBatches, "values are correct");

  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
}

// E. A producer nobody reads. It has no server, so nothing else releases the
// task registration.
void scenarioUnreadProducer(
    memory::MemoryPool* pool,
    folly::Executor* executor) {
  std::cout << "\n== E. a producer nobody reads ==\n";
  enableDynamicUcx(true);

  const std::string taskId{"unread-producer"};
  auto input = makeInput(pool);
  auto producer = startProducer(
      taskId, input, /*nameTransport=*/false, /*onGpu=*/true, executor);
  producer->requestCancel();
  producer->taskCompletionFuture().wait(std::chrono::seconds(30));
  producer.reset();

  const bool stillRegistered =
      UcxOutputQueueManager::getInstanceRef()->getQueueIfExists(taskId) !=
      nullptr;
  std::cout << "  still registered with the queue manager: "
            << (stillRegistered ? "yes" : "no") << "\n";
  check(!stillRegistered, "the task registration was released");
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};

  if (const char* value = std::getenv("VELOX_UCX_LAB_PORT")) {
    gListenerPort = static_cast<uint16_t>(std::stoi(value));
  }
  setenv("UCX_CM_REUSEADDR", "y", /*overwrite=*/0);

  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  if (!isRegisteredNamedVectorSerde(
          VectorSerde::kindName(VectorSerde::Kind::kPresto))) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }

  // The same entry point a deployment uses.
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  cudfConfig.exchange = true;
  cudfConfig.exchangeServerPort = gListenerPort;

  // PlanBuilder::filter() parses an expression, which needs both.
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();

  cudf_velox::registerCudf();
  if (!startCudfUcxExchange()) {
    std::cerr << "startCudfUcxExchange() declined to start\n";
    return 1;
  }

  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(8);
  auto rootPool = memory::memoryManager()->addRootPool("dynamic_ucx_lab");
  auto pool = rootPool->addLeafChild("dynamic_ucx_lab_leaf");

  const std::string only =
      std::getenv("LAB_ONLY") ? std::getenv("LAB_ONLY") : "";
  if (only.empty() || only == "A") {
    scenarioPlanDeclared(pool.get(), executor.get());
  }
  if (only.empty() || only == "B") {
    scenarioResidency(
        "host -> cudf -> host",
        "host-host",
        false,
        false,
        pool.get(),
        executor.get());
    scenarioResidency(
        "host -> cudf -> gpu",
        "host-gpu",
        false,
        true,
        pool.get(),
        executor.get());
    scenarioResidency(
        "gpu -> cudf -> host",
        "gpu-host",
        true,
        false,
        pool.get(),
        executor.get());
    scenarioResidency(
        "gpu -> cudf -> gpu",
        "gpu-gpu",
        true,
        true,
        pool.get(),
        executor.get());
  }
  if (only.empty() || only == "C") {
    scenarioHostOnlyReader(pool.get(), executor.get());
  }
  if (only.empty() || only == "D") {
    scenarioNonCudfQueryUntouched(pool.get(), executor.get());
  }
  if (only.empty() || only == "E") {
    scenarioUnreadProducer(pool.get(), executor.get());
  }
  if (only.empty() || only == "F") {
    scenarioMultiplePartitions(
        "F. 4 partitions",
        "partitioned-producer",
        /*maxOutputBufferBytes=*/32 << 20,
        /*startConsumersLate=*/false,
        /*numInputBatches=*/1,
        pool.get(),
        executor.get());
  }
  if (only.empty() || only == "I") {
    // The same shuffle against a buffer too small to hold it, so the producer
    // must block and resume repeatedly.
    scenarioMultiplePartitions(
        "I. 4 partitions against a full output buffer",
        "backpressure-producer",
        /*maxOutputBufferBytes=*/16384,
        /*startConsumersLate=*/true,
        /*numInputBatches=*/8,
        pool.get(),
        executor.get());
  }
  if (only.empty() || only == "G") {
    scenarioZeroColumnPage(pool.get());
  }
  if (only.empty() || only == "K") {
    scenarioConsumerDiesMidStream(pool.get(), executor.get());
  }
  if (only.empty() || only == "J") {
    scenarioBroadcast(pool.get(), executor.get());
  }
  if (only.empty() || only == "H") {
    scenarioPlanNamesUcxWithDiscovery(pool.get(), executor.get());
  }

  std::cout << "\n--- pool tree at shutdown ---\n"
            << rootPool->treeMemoryUsage(false) << "\n";

  std::cout << "\n"
            << (gFailures == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT")
            << " (" << gFailures << " failure(s))" << std::endl;
  return gFailures == 0 ? 0 : 1;
}
