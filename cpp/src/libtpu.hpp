/**
 * @file libtpu.hpp
 * @brief Strongly-typed, zero-exception C++20 RAII wrapper over `libtpu.h`.
 */

#ifndef LIBTPU_HPP_
#define LIBTPU_HPP_

#include "libtpu.h"
#include <cstddef>
#include <cstdint>

/**
 * @namespace libtpu
 * @brief Root C++20 namespace for strongly-typed `libtpu-rs` RAII classes.
 */
namespace libtpu {

/**
 * @brief Strongly-typed PJRT status code enumeration.
 */
enum class ErrorCode : int32_t {
  /** @brief Operation succeeded (`0`). */
  Ok = 0,
  /** @brief Invalid argument (`3`). */
  InvalidArgument = 3,
  /** @brief Requested entity not found (`5`). */
  NotFound = 5,
  /** @brief Entity already exists (`6`). */
  AlreadyExists = 6,
  /** @brief Resource exhausted (`8`). */
  ResourceExhausted = 8,
  /** @brief Failed precondition (`9`). */
  FailedPrecondition = 9,
  /** @brief Internal runtime error (`13`). */
  Internal = 13,
};

/**
 * @brief Strongly-typed tensor element type enumeration matching
 * `PJRT_Buffer_Type`.
 */
enum class DataType : int32_t {
  /** @brief Invalid / unspecified element type (`0`). */
  Invalid = 0,
  /** @brief Boolean predicate (`1`). */
  Pred = 1,
  /** @brief 8-bit signed integer (`2`). */
  S8 = 2,
  /** @brief 16-bit signed integer (`3`). */
  S16 = 3,
  /** @brief 32-bit signed integer (`4`). */
  S32 = 4,
  /** @brief 64-bit signed integer (`5`). */
  S64 = 5,
  /** @brief 8-bit unsigned integer (`6`). */
  U8 = 6,
  /** @brief 16-bit unsigned integer (`7`). */
  U16 = 7,
  /** @brief 32-bit unsigned integer (`8`). */
  U32 = 8,
  /** @brief 64-bit unsigned integer (`9`). */
  U64 = 9,
  /** @brief 16-bit IEEE floating-point (`10`). */
  F16 = 10,
  /** @brief 32-bit IEEE floating-point (`11`). */
  F32 = 11,
  /** @brief 64-bit IEEE floating-point (`12`). */
  F64 = 12,
  /** @brief 64-bit complex floating-point (`14`). */
  C64 = 14,
  /** @brief 128-bit complex floating-point (`15`). */
  C128 = 15,
  /** @brief 16-bit brain floating-point (`16`). */
  Bf16 = 16,
};

/**
 * @brief Strongly-typed Cloud TPU generation enumeration matching
 * `TpuVersionEnum`.
 */
enum class TpuVersion : int32_t {
  /** @brief Unspecified TPU generation (`0`). */
  Unspecified = 0,
  /** @brief Cloud TPU v2 (`2`). */
  V2 = 2,
  /** @brief Cloud TPU v3 (`Jellyfish`, `3`). */
  V3 = 3,
  /** @brief Cloud TPU v4 (`Pufferfish`, `4`). */
  V4 = 4,
  /** @brief Cloud TPU v5e (`Viperfish`, `5`). */
  V5e = 5,
  /** @brief Cloud TPU v5p (`Ghostfish`, `6`). */
  V5p = 6,
  /** @brief Cloud TPU v6e (`Zebrafish` / Trillium, `7`). */
  V6e = 7,
};

/**
 * @brief Zero-exception RAII status wrapper over `LibTpuCStatus`.
 */
class Status {
public:
  /**
   * @brief Constructs a `Status` taking ownership of a raw `LibTpuCStatus` and
   * an optional static/cached message pointer.
   * @param raw Raw `LibTpuCStatus` returned from the C FFI.
   * @param msg_str Cached null-terminated message string pointer (defaults to
   * `""`).
   */
  explicit Status(LibTpuCStatus raw, const char *msg_str = "") noexcept;

  /**
   * @brief Destroys the `Status` and frees any owned error message.
   */
  ~Status() noexcept;

  /**
   * @brief Deleted copy constructor to enforce unique ownership of the C
   * message buffer.
   * @param other Other status instance.
   */
  Status(const Status &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other status instance.
   * @return Reference to this status.
   */
  Status &operator=(const Status &other) = delete;

  /**
   * @brief Move constructor transferring ownership of the underlying
   * `LibTpuCStatus`.
   * @param other Rvalue status to move from.
   */
  Status(Status &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the underlying
   * `LibTpuCStatus`.
   * @param other Rvalue status to move from.
   * @return Reference to this status.
   */
  Status &operator=(Status &&other) noexcept;

  /**
   * @brief Returns `true` if the operation succeeded (`code == 0`).
   * @return `true` when OK, `false` otherwise.
   */
  [[nodiscard]] bool is_ok() const noexcept;

  /**
   * @brief Returns the raw integer status code.
   * @return 32-bit signed status code.
   */
  [[nodiscard]] int32_t code() const noexcept;

  /**
   * @brief Returns the null-terminated error message string (or `""` if OK).
   * @return Pointer to null-terminated UTF-8 string.
   */
  [[nodiscard]] const char *message() const noexcept;

private:
  /** @brief Underlying C status struct. */
  LibTpuCStatus raw_;
  /** @brief Cached null-terminated diagnostic message string. */
  const char *msg_str_;
};

/**
 * @brief RAII wrapper over `PJRT_Buffer*` representing an on-device HBM tensor.
 */
class Buffer {
public:
  /**
   * @brief Constructs an empty `Buffer` or takes ownership of `handle`.
   * @param handle Raw `PJRT_Buffer*` pointer (defaults to `nullptr`).
   */
  explicit Buffer(PJRT_Buffer *handle = nullptr) noexcept;

  /**
   * @brief Destroys the `Buffer` and releases on-device HBM memory.
   */
  ~Buffer() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other buffer.
   */
  Buffer(const Buffer &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other buffer.
   * @return Reference to this buffer.
   */
  Buffer &operator=(const Buffer &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `PJRT_Buffer*`.
   * @param other Rvalue buffer to move from.
   */
  Buffer(Buffer &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of `PJRT_Buffer*`.
   * @param other Rvalue buffer to move from.
   * @return Reference to this buffer.
   */
  Buffer &operator=(Buffer &&other) noexcept;

  /**
   * @brief Copies the on-device buffer bytes into `host_dst`.
   * @param host_dst Destination host byte array.
   * @param len Number of bytes to copy into `host_dst`.
   * @return `Status` indicating success or transfer error.
   */
  [[nodiscard]] Status to_host(uint8_t *host_dst, size_t len) const noexcept;

  /**
   * @brief Returns the rank (number of dimensions) of this buffer.
   * @return Number of dimensions (`0` for scalars).
   */
  [[nodiscard]] size_t rank() const noexcept;

  /**
   * @brief Returns the extent of dimension `axis`.
   * @param axis Zero-based dimension axis index.
   * @return Dimension extent along `axis`.
   */
  [[nodiscard]] int64_t dimension(size_t axis) const noexcept;

  /**
   * @brief Returns the `DataType` of this buffer.
   * @return Strongly-typed `DataType`.
   */
  [[nodiscard]] DataType element_type() const noexcept;

  /**
   * @brief Returns the number of hardware layout tiles on this buffer via
   * `PJRT_Buffer_GetMemoryLayout`.
   * @return Number of hardware tiles (`1` for standard `(8, 128)` 2D tiling).
   */
  [[nodiscard]] size_t tile_count() const noexcept;

  /**
   * @brief Copies this buffer to `dst_device_id` on `client`.
   * @param client Owning `PJRT_Client*` handle.
   * @param dst_device_id Destination logical TPU device ID.
   * @param out_buffer Output `Buffer` receiving the copied handle.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status copy_to_device(PJRT_Client *client,
                                      uint32_t dst_device_id,
                                      Buffer *out_buffer) const noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_Buffer*` pointer.
   * @return Raw `PJRT_Buffer*` handle.
   */
  [[nodiscard]] const PJRT_Buffer *raw() const noexcept;

private:
  /** @brief Owned C buffer handle. */
  PJRT_Buffer *handle_;
};

/**
 * @brief RAII wrapper over `PJRT_LoadedExecutable*`.
 */
class LoadedExecutable {
public:
  /**
   * @brief Constructs an empty `LoadedExecutable` or takes ownership of
   * `handle`.
   * @param handle Raw `PJRT_LoadedExecutable*` pointer (defaults to `nullptr`).
   */
  explicit LoadedExecutable(PJRT_LoadedExecutable *handle = nullptr) noexcept;

  /**
   * @brief Destroys the `LoadedExecutable`.
   */
  ~LoadedExecutable() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other executable.
   */
  LoadedExecutable(const LoadedExecutable &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other executable.
   * @return Reference to this executable.
   */
  LoadedExecutable &operator=(const LoadedExecutable &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `PJRT_LoadedExecutable*`.
   * @param other Rvalue executable to move from.
   */
  LoadedExecutable(LoadedExecutable &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of
   * `PJRT_LoadedExecutable*`.
   * @param other Rvalue executable to move from.
   * @return Reference to this executable.
   */
  LoadedExecutable &operator=(LoadedExecutable &&other) noexcept;

  /**
   * @brief Executes this program on `device_id` with two input buffers.
   * @param device_id Target logical TPU device ID.
   * @param input0 First input buffer.
   * @param input1 Second input buffer.
   * @param output_out Output `Buffer` receiving the result tensor.
   * @return `Status` indicating execution success or failure.
   */
  [[nodiscard]] Status execute_binary(uint32_t device_id, const Buffer &input0,
                                      const Buffer &input1,
                                      Buffer *output_out) const noexcept;

  /**
   * @brief Executes this program on `device_id` with arbitrary `num_inputs` and
   * `num_outputs`.
   * @param device_id Target logical TPU device ID.
   * @param inputs Array of input `const PJRT_Buffer*` pointers.
   * @param num_inputs Number of input buffers.
   * @param outputs_out Caller-allocated array of `PJRT_Buffer*` of length
   * `num_outputs`.
   * @param num_outputs Number of output buffer slots in `outputs_out`.
   * @return `Status` indicating execution success or failure.
   */
  [[nodiscard]] Status execute(uint32_t device_id,
                               const PJRT_Buffer *const *inputs,
                               size_t num_inputs, PJRT_Buffer **outputs_out,
                               size_t num_outputs) const noexcept;

  /**
   * @brief Serializes this executable into `out_bytes` (or queries byte length
   * when `out_bytes == nullptr`).
   * @param out_bytes Optional destination byte array.
   * @param max_len Capacity of `out_bytes` in bytes.
   * @param out_len Output pointer receiving the total serialized byte count.
   * @return `Status` indicating serialization success or failure.
   */
  [[nodiscard]] Status serialize(uint8_t *out_bytes, size_t max_len,
                                 size_t *out_len) const noexcept;

  /**
   * @brief Copies the 32-character hex fingerprint into `out_buf` and returns
   * its byte length.
   * @param out_buf Optional destination character buffer.
   * @param buf_len Capacity of `out_buf` in bytes.
   * @return Length of the fingerprint in bytes (`32`).
   */
  [[nodiscard]] size_t fingerprint(char *out_buf,
                                   size_t buf_len) const noexcept;

  /**
   * @brief Returns the estimated FLOPs of this executable.
   * @return Estimated floating-point operations.
   */
  [[nodiscard]] float flops() const noexcept;

  /**
   * @brief Returns the number of outputs produced by this executable via
   * `PJRT_Executable_OutputMemoryKinds`.
   * @return Number of output tensors (`>= 1`).
   */
  [[nodiscard]] size_t num_outputs() const noexcept;

  /**
   * @brief Queries the optimized HLO program via
   * `PJRT_Executable_OptimizedProgram` using the two-call buffer sizing
   * protocol.
   * @param out_bytes Destination byte array receiving the optimized program.
   * @param max_len Capacity of `out_bytes` in bytes.
   * @param out_len Output pointer receiving the optimized program byte length.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status optimized_program(uint8_t *out_bytes, size_t max_len,
                                         size_t *out_len) const noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_LoadedExecutable*` pointer.
   * @return Raw `PJRT_LoadedExecutable*` handle.
   */
  [[nodiscard]] const PJRT_LoadedExecutable *raw() const noexcept;

private:
  /** @brief Owned C loaded executable handle. */
  PJRT_LoadedExecutable *handle_;
};

/**
 * @brief RAII wrapper over `PJRT_Event*` driving asynchronous event readiness
 * and callbacks.
 */
class Event {
public:
  /**
   * @brief Constructs an empty `Event` or takes ownership of `handle`.
   * @param handle Raw `PJRT_Event*` pointer (defaults to `nullptr`).
   */
  explicit Event(PJRT_Event *handle = nullptr) noexcept;

  /**
   * @brief Destroys the `Event` via `PJRT_Event_Destroy`.
   */
  ~Event() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other event.
   */
  Event(const Event &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other event.
   * @return Reference to this event.
   */
  Event &operator=(const Event &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `PJRT_Event*`.
   * @param other Rvalue event to move from.
   */
  Event(Event &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of `PJRT_Event*`.
   * @param other Rvalue event to move from.
   * @return Reference to this event.
   */
  Event &operator=(Event &&other) noexcept;

  /**
   * @brief Creates a new pending `Event` via `PJRT_Event_Create`.
   * @param event_out Output `Event` receiving the created handle.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] static Status create(Event *event_out) noexcept;

  /**
   * @brief Queries whether this event has completed via `PJRT_Event_IsReady`.
   * @return `true` if ready, `false` if pending.
   */
  [[nodiscard]] bool is_ready() const noexcept;

  /**
   * @brief Resolves this event with `error_code` and `message` via
   * `PJRT_Event_Set`.
   * @param error_code Status code (`0` for OK).
   * @param message Optional error message string.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status set(int32_t error_code = 0,
                           const char *message = nullptr) noexcept;

  /**
   * @brief Registers a readiness callback via `PJRT_Event_OnReady`.
   * @param callback C callback function pointer (`void (*)(PJRT_Error*,
   * void*)`).
   * @param user_arg Opaque user argument pointer passed to `callback`.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status on_ready(PJRT_Event_OnReadyCallback callback,
                                void *user_arg) noexcept;

  /**
   * @brief Blocks until this event completes via `PJRT_Event_Await`.
   * @return `Status` indicating event completion status.
   */
  [[nodiscard]] Status await() const noexcept;

private:
  /** @brief Owned C event handle. */
  PJRT_Event *handle_;
};

/**
 * @brief RAII wrapper over `PJRT_Client*`.
 */
class Client {
public:
  /**
   * @brief Constructs an empty `Client` or takes ownership of `handle`.
   * @param handle Raw `PJRT_Client*` pointer (defaults to `nullptr`).
   */
  explicit Client(PJRT_Client *handle = nullptr) noexcept;

  /**
   * @brief Destroys the `Client` and all owned device/memory handles.
   */
  ~Client() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other client.
   */
  Client(const Client &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other client.
   * @return Reference to this client.
   */
  Client &operator=(const Client &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `PJRT_Client*`.
   * @param other Rvalue client to move from.
   */
  Client(Client &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of `PJRT_Client*`.
   * @param other Rvalue client to move from.
   * @return Reference to this client.
   */
  Client &operator=(Client &&other) noexcept;

  /**
   * @brief Factory method creating and initializing a `Client`.
   * @param client_out Output `Client` receiving the initialized handle.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] static Status create(Client *client_out) noexcept;

  /**
   * @brief Returns the number of addressable TPU devices managed by this
   * client.
   * @return Device count (`>= 0`), or `-1` if uninitialized.
   */
  [[nodiscard]] int32_t device_count() const noexcept;

  /**
   * @brief Returns the number of per-device `PJRT_NamedValue` attributes for
   * `device_id` via `PJRT_Device_GetAttributes`.
   * @param device_id Target logical TPU device ID.
   * @return Number of device attributes (`3` for valid devices, `0` on invalid
   * device).
   */
  [[nodiscard]] size_t
  device_attribute_count(uint32_t device_id) const noexcept;

  /**
   * @brief Allocates a device `Buffer` from host bytes.
   * @param device_id Target logical TPU device ID.
   * @param dtype Element data type.
   * @param dims Pointer to dimension extents array.
   * @param num_dims Number of dimensions.
   * @param host_data Source host byte pointer.
   * @param host_data_len Number of bytes in `host_data`.
   * @param buffer_out Output `Buffer` receiving the allocated tensor.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status buffer_from_host(uint32_t device_id, DataType dtype,
                                        const int64_t *dims, size_t num_dims,
                                        const uint8_t *host_data,
                                        size_t host_data_len,
                                        Buffer *buffer_out) noexcept;

  /**
   * @brief Compiles and loads an HLO or `StableHLO` program.
   * @param format Null-terminated format string (`"mlir"` or `"hlo"`).
   * @param code Pointer to program source bytes.
   * @param code_len Length of `code` in bytes.
   * @param exec_out Output `LoadedExecutable` receiving the compiled program.
   * @return `Status` indicating compilation success or failure.
   */
  [[nodiscard]] Status compile(const char *format, const uint8_t *code,
                               size_t code_len,
                               LoadedExecutable *exec_out) noexcept;

  /**
   * @brief Deserializes and loads a previously serialized executable.
   * @param bytes Pointer to serialized executable bytes.
   * @param bytes_len Length of `bytes` in bytes.
   * @param exec_out Output `LoadedExecutable` receiving the deserialized
   * program.
   * @return `Status` indicating deserialization success or failure.
   */
  [[nodiscard]] Status
  deserialize_executable(const uint8_t *bytes, size_t bytes_len,
                         LoadedExecutable *exec_out) noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_Client*` pointer.
   * @return Raw `PJRT_Client*` handle.
   */
  [[nodiscard]] PJRT_Client *raw() const noexcept;

private:
  /** @brief Owned C client handle. */
  PJRT_Client *handle_;
};

/**
 * @brief RAII wrapper over `TpuProfiler*`.
 */
class Profiler {
public:
  /**
   * @brief Creates and initializes a `Profiler` session.
   */
  Profiler() noexcept;

  /**
   * @brief Destroys the `Profiler` session.
   */
  ~Profiler() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other profiler.
   */
  Profiler(const Profiler &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other profiler.
   * @return Reference to this profiler.
   */
  Profiler &operator=(const Profiler &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `TpuProfiler*` and
   * `TpuStatus*`.
   * @param other Rvalue profiler to move from.
   */
  Profiler(Profiler &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of `TpuProfiler*`
   * and `TpuStatus*`.
   * @param other Rvalue profiler to move from.
   * @return Reference to this profiler.
   */
  Profiler &operator=(Profiler &&other) noexcept;

  /**
   * @brief Starts hardware and runtime trace collection.
   * @return `true` if started successfully, `false` otherwise.
   */
  [[nodiscard]] bool start() noexcept;

  /**
   * @brief Stops hardware and runtime trace collection.
   * @return `true` if stopped successfully, `false` otherwise.
   */
  [[nodiscard]] bool stop() noexcept;

  /**
   * @brief Returns the byte size of collected `XSpace` trace data.
   * @return Number of bytes available.
   */
  [[nodiscard]] size_t collect_size() noexcept;

private:
  /** @brief Owned C profiler handle. */
  TpuProfiler *handle_;
  /** @brief Owned C status handle used for profiler operations. */
  TpuStatus *status_;
};

/**
 * @brief Strongly-typed C++20 wrapper over the global `TpuTopology*` singleton.
 */
class Topology {
public:
  /**
   * @brief Constructs a `Topology` handle referencing the global TPU topology.
   */
  Topology() noexcept;

  /**
   * @brief Returns the number of logical devices per host.
   * @return Logical devices per host.
   */
  [[nodiscard]] int32_t logical_devices_per_host() const noexcept;

  /**
   * @brief Returns the X chip mesh bound.
   * @return X chip mesh bound.
   */
  [[nodiscard]] int32_t chip_bounds_x() const noexcept;

  /**
   * @brief Returns the Y chip mesh bound.
   * @return Y chip mesh bound.
   */
  [[nodiscard]] int32_t chip_bounds_y() const noexcept;

  /**
   * @brief Returns the Z chip mesh bound.
   * @return Z chip mesh bound.
   */
  [[nodiscard]] int32_t chip_bounds_z() const noexcept;

  /**
   * @brief Returns the TPU generation version integer (`TpuVersionEnum`).
   * @return TPU generation version (`5` for `v5e`).
   */
  [[nodiscard]] int32_t version() const noexcept;

private:
  /** @brief Non-owning pointer to global `TpuTopology`. */
  const TpuTopology *handle_;
};

/**
 * @brief Strongly-typed C++20 RAII wrapper over `LibtpuSdk_Client*` driving
 * `GetLibtpuSdkApi()`.
 */
class SdkClient {
public:
  /**
   * @brief Constructs and initializes an `SdkClient` via
   * `GetLibtpuSdkApi()->client_create`.
   */
  SdkClient() noexcept;

  /**
   * @brief Destroys the `SdkClient` via `GetLibtpuSdkApi()->client_destroy`.
   */
  ~SdkClient() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other SDK client.
   */
  SdkClient(const SdkClient &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other SDK client.
   * @return Reference to this SDK client.
   */
  SdkClient &operator=(const SdkClient &other) = delete;

  /**
   * @brief Move constructor transferring ownership of `LibtpuSdk_Client*`.
   * @param other Rvalue SDK client to move from.
   */
  SdkClient(SdkClient &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of
   * `LibtpuSdk_Client*`.
   * @param other Rvalue SDK client to move from.
   * @return Reference to this SDK client.
   */
  SdkClient &operator=(SdkClient &&other) noexcept;

  /**
   * @brief Queries the byte length of the runtime core state summary via
   * `GetLibtpuSdkApi()`.
   * @return Length of the core state summary string (`> 0`).
   */
  [[nodiscard]] size_t core_state_summary_size() const noexcept;

  /**
   * @brief Queries the number of per-device values for `metric_name` via
   * `GetLibtpuSdkApi()`.
   * @param metric_name Null-terminated metric identifier string.
   * @return Number of metric values across devices.
   */
  [[nodiscard]] size_t
  metric_value_count(const char *metric_name) const noexcept;

  /**
   * @brief Queries the number of TPU chips discovered by
   * `GetLibtpuSdkApi()->get_chip_coordinates`.
   * @return Number of TPU chips on this host.
   */
  [[nodiscard]] size_t chip_count() const noexcept;

  /**
   * @brief Registers a debug callback via
   * `GetLibtpuSdkApi()->register_debug_callback`.
   * @param callback C callback function pointer (`void (*)(void*)`).
   * @param user_data Opaque user context pointer passed to `callback`.
   * @return Registered callback ID (`> 0`).
   */
  [[nodiscard]] int64_t register_debug_callback(const void *callback,
                                                void *user_data) noexcept;

  /**
   * @brief Unregisters a debug callback via
   * `GetLibtpuSdkApi()->unregister_debug_callback`.
   * @param callback_id Callback ID returned by `register_debug_callback`.
   * @return `true` if unregistered without error, `false` otherwise.
   */
  [[nodiscard]] bool unregister_debug_callback(int64_t callback_id) noexcept;

  /**
   * @brief Invokes all registered debug callbacks on this `SdkClient`.
   * @return Number of callbacks invoked.
   */
  [[nodiscard]] size_t trigger_debug_callbacks() const noexcept;

private:
  /** @brief Pointer to the static `LibtpuSdk_Api` vtable. */
  const LibtpuSdk_Api *api_;
  /** @brief Owned `LibtpuSdk_Client*` handle. */
  LibtpuSdk_Client *handle_;
};

/**
 * @brief Strongly-typed C++20 view over the static `PJRT_Api` (`1,144` bytes)
 * vtable and its 17-node `PJRT_Extension_Base` linked list.
 */
class PjrtApiView {
public:
  /**
   * @brief Constructs a `PjrtApiView` bound to `GetPjrtApi()`.
   */
  PjrtApiView() noexcept;

  /**
   * @brief Returns the `PJRT_Api` byte size (`1144`).
   * @return Byte size of `PJRT_Api`.
   */
  [[nodiscard]] size_t struct_size() const noexcept;

  /**
   * @brief Returns the `PJRT_Api_Version` minor version (`116`).
   * @return PJRT minor version number.
   */
  [[nodiscard]] int32_t minor_version() const noexcept;

  /**
   * @brief Counts the number of `PJRT_Extension_Base` nodes in the extension
   * chain (`17`).
   * @return Number of extensions linked from `extension_start`.
   */
  [[nodiscard]] size_t extension_count() const noexcept;

  /**
   * @brief Searches the `PJRT_Extension_Base` linked list for `ext_type`.
   * @param ext_type Numeric `PJRT_Extension_Type` discriminant.
   * @return Pointer to matching `PJRT_Extension_Base` or `nullptr` if not
   * found.
   */
  [[nodiscard]] const PJRT_Extension_Base *
  find_extension(int32_t ext_type) const noexcept;

  /**
   * @brief Queries the number of device memory descriptions (`4`: HBM, VMEM,
   * SMEM, pinned_host) via `PJRT_MemoryDescriptions_Extension`.
   * @return Number of device memory space descriptions (`4`).
   */
  [[nodiscard]] size_t memory_description_count() const noexcept;

private:
  /** @brief Non-owning pointer to static `PJRT_Api` vtable. */
  const PJRT_Api *api_;
};

/**
 * @brief Non-owning strongly-typed C++20 view over a `PJRT_Device*`.
 */
class Device {
public:
  /**
   * @brief Constructs a `Device` view over `handle`.
   * @param handle Raw `PJRT_Device*` pointer (defaults to `nullptr`).
   */
  explicit Device(PJRT_Device *handle = nullptr) noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_Device*` pointer.
   * @return Raw `PJRT_Device*` handle.
   */
  [[nodiscard]] PJRT_Device *raw() const noexcept;

private:
  /** @brief Non-owning C device handle. */
  PJRT_Device *handle_;
};

/**
 * @brief Non-owning strongly-typed C++20 view over a `PJRT_Memory*`.
 */
class Memory {
public:
  /**
   * @brief Constructs a `Memory` view over `handle`.
   * @param handle Raw `PJRT_Memory*` pointer (defaults to `nullptr`).
   */
  explicit Memory(PJRT_Memory *handle = nullptr) noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_Memory*` pointer.
   * @return Raw `PJRT_Memory*` handle.
   */
  [[nodiscard]] PJRT_Memory *raw() const noexcept;

private:
  /** @brief Non-owning C memory handle. */
  PJRT_Memory *handle_;
};

/**
 * @brief Non-owning strongly-typed C++20 view over a `PJRT_Executable*`.
 */
class Executable {
public:
  /**
   * @brief Constructs an `Executable` view over `handle`.
   * @param handle Raw `PJRT_Executable*` pointer (defaults to `nullptr`).
   */
  explicit Executable(PJRT_Executable *handle = nullptr) noexcept;

  /**
   * @brief Returns the underlying raw `PJRT_Executable*` pointer.
   * @return Raw `PJRT_Executable*` handle.
   */
  [[nodiscard]] PJRT_Executable *raw() const noexcept;

private:
  /** @brief Non-owning C executable handle. */
  PJRT_Executable *handle_;
};

/** @brief Type alias for `SdkClient` (`SdkMonitor`). */
using SdkMonitor = SdkClient;

/**
 * @brief Strongly-typed C++20 RAII wrapper over `PJRT_PhaseCompile_Extension`.
 */
class PhaseCompiler {
public:
  /**
   * @brief Acquires a `PhaseCompiler` handle via
   * `PJRT_PhaseCompile_GetCompiler`.
   */
  PhaseCompiler() noexcept;

  /**
   * @brief Releases the `PhaseCompiler` handle via
   * `PJRT_PhaseCompile_DestroyCompiler`.
   */
  ~PhaseCompiler() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other phase compiler.
   */
  PhaseCompiler(const PhaseCompiler &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other phase compiler.
   * @return Reference to this phase compiler.
   */
  PhaseCompiler &operator=(const PhaseCompiler &other) = delete;

  /**
   * @brief Queries the number of supported compilation phases (`5`).
   * @return Number of canonical compilation phases.
   */
  [[nodiscard]] size_t phase_count() const noexcept;

  /**
   * @brief Runs a named compilation phase on `input_bytes`.
   * @param phase_name Null-terminated compilation phase name.
   * @param input_bytes Pointer to MLIR or HLO program bytes.
   * @param input_size Byte length of `input_bytes`.
   * @return Output serialized partial program byte size (`> 0`).
   */
  [[nodiscard]] size_t run_phase(const char *phase_name,
                                 const char *input_bytes,
                                 size_t input_size) noexcept;

private:
  /** @brief Opaque phase compiler handle ID. */
  size_t handle_;
};

/**
 * @brief Strongly-typed C++20 RAII wrapper over `PJRT_Megascale_Extension`.
 */
class MegascaleContext {
public:
  /**
   * @brief Creates a `MegascaleContext` via
   * `PJRT_Megascale_CreateClientContext`.
   */
  MegascaleContext() noexcept;

  /**
   * @brief Deletes the `MegascaleContext` via
   * `PJRT_Megascale_DeleteClientContext`.
   */
  ~MegascaleContext() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other Megascale context.
   */
  MegascaleContext(const MegascaleContext &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other Megascale context.
   * @return Reference to this Megascale context.
   */
  MegascaleContext &operator=(const MegascaleContext &other) = delete;

  /**
   * @brief Executes a multi-slice Megascale barrier and returns the barrier
   * count.
   * @return Completed barrier count (`>= 1`).
   */
  [[nodiscard]] size_t barrier() noexcept;

  /**
   * @brief Queries the number of slices in the Megascale topology.
   * @return Number of slices (`>= 1`).
   */
  [[nodiscard]] size_t num_slices() const noexcept;

private:
  /** @brief Opaque Megascale context handle ID. */
  size_t handle_;
};

/**
 * @brief Strongly-typed C++20 RAII wrapper over `TpuEmbeddingEngineState*`.
 */
class EmbeddingEngine {
public:
  /**
   * @brief Allocates a `TpuEmbeddingEngineState` via
   * `TpuEmbeddingEngineState_Create`.
   */
  EmbeddingEngine() noexcept;

  /**
   * @brief Frees the `TpuEmbeddingEngineState` via
   * `TpuEmbeddingEngineState_Free`.
   */
  ~EmbeddingEngine() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other embedding engine.
   */
  EmbeddingEngine(const EmbeddingEngine &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other embedding engine.
   * @return Reference to this embedding engine.
   */
  EmbeddingEngine &operator=(const EmbeddingEngine &other) = delete;

  /**
   * @brief Returns the internal `TpuEmbeddingEngineState` state pointer.
   * @return Non-null internal state pointer.
   */
  [[nodiscard]] const void *raw_state() const noexcept;

private:
  /** @brief Owned `TpuEmbeddingEngineState*` handle. */
  TpuEmbeddingEngineState *state_;
};

} // namespace libtpu

#endif // LIBTPU_HPP_
