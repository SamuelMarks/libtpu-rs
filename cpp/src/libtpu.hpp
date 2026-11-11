/**
 * @file libtpu.hpp
 * @brief Strongly-typed, zero-exception C++20 RAII wrapper over `libtpu.h`.
 */

#ifndef LIBTPU_HPP_
#define LIBTPU_HPP_

#include "libtpu.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

/**
 * @namespace libtpu
 * @brief Root C++20 namespace for strongly-typed `libtpu-rs` RAII classes.
 */
namespace libtpu {

/**
 * @brief Strongly-typed PJRT status code enumeration (`PJRT_Error_Code`).
 */
enum class ErrorCode : int32_t {
  /** @brief Operation succeeded (`0`). */
  Ok = 0,
  /** @brief Operation was cancelled (`1`). */
  Cancelled = 1,
  /** @brief Unknown error (`2`). */
  Unknown = 2,
  /** @brief Invalid argument (`3`). */
  InvalidArgument = 3,
  /** @brief Deadline exceeded (`4`). */
  DeadlineExceeded = 4,
  /** @brief Requested entity not found (`5`). */
  NotFound = 5,
  /** @brief Entity already exists (`6`). */
  AlreadyExists = 6,
  /** @brief Permission denied (`7`). */
  PermissionDenied = 7,
  /** @brief Resource exhausted (`8`). */
  ResourceExhausted = 8,
  /** @brief Failed precondition (`9`). */
  FailedPrecondition = 9,
  /** @brief Operation aborted (`10`). */
  Aborted = 10,
  /** @brief Out of range (`11`). */
  OutOfRange = 11,
  /** @brief Unimplemented operation (`12`). */
  Unimplemented = 12,
  /** @brief Internal runtime error (`13`). */
  Internal = 13,
  /** @brief Service or device unavailable (`14`). */
  Unavailable = 14,
  /** @brief Unrecoverable data loss (`15`). */
  DataLoss = 15,
  /** @brief Unauthenticated request (`16`). */
  Unauthenticated = 16,
};

/**
 * @brief Strongly-typed tensor element type enumeration matching all 29
 * `PJRT_Buffer_Type` variants.
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
  /** @brief 16-bit brain floating-point (`13`). */
  Bf16 = 13,
  /** @brief 64-bit complex floating-point (`14`). */
  C64 = 14,
  /** @brief 128-bit complex floating-point (`15`). */
  C128 = 15,
  /** @brief 8-bit floating-point E5M2 (`16`). */
  F8E5M2 = 16,
  /** @brief 8-bit floating-point E4M3FN (`17`). */
  F8E4M3Fn = 17,
  /** @brief 8-bit floating-point E4M3B11FNUZ (`18`). */
  F8E4M3B11Fnuz = 18,
  /** @brief 8-bit floating-point E5M2FNUZ (`19`). */
  F8E5M2Fnuz = 19,
  /** @brief 8-bit floating-point E4M3FNUZ (`20`). */
  F8E4M3Fnuz = 20,
  /** @brief 4-bit signed integer (`21`). */
  S4 = 21,
  /** @brief 4-bit unsigned integer (`22`). */
  U4 = 22,
  /** @brief Ordering token (`23`). */
  Token = 23,
  /** @brief 2-bit signed integer (`24`). */
  S2 = 24,
  /** @brief 2-bit unsigned integer (`25`). */
  U2 = 25,
  /** @brief 4-bit floating-point E2M1FN (`26`). */
  F4E2M1Fn = 26,
  /** @brief 8-bit block exponent E8M0FNU (`27`). */
  F8E8M0Fnu = 27,
  /** @brief Tuple composite type (`28`). */
  Tuple = 28,
};

/**
 * @brief Strongly-typed Cloud TPU generation enumeration matching
 * `TpuVersionEnum`.
 */
enum class TpuVersion : int32_t {
  /** @brief Unspecified TPU generation (`0`). */
  Unspecified = 0,
  /** @brief Cloud TPU v2 (`Jellyfish`, `2`). */
  V2 = 2,
  /** @brief Cloud TPU v3 (`Dragonfish`, `3`). */
  V3 = 3,
  /** @brief Cloud TPU v4 (`Pufferfish`, `4`). */
  V4 = 4,
  /** @brief Cloud TPU v5e (`Viperfish Lite`, `5`). */
  V5e = 5,
  /** @brief Cloud TPU v5p (`Viperfish`, `6`). */
  V5p = 6,
  /** @brief Cloud TPU v6e (`Ghostfish` / Trillium, `7`). */
  V6e = 7,
  /** @brief Cloud TPU7x (`Ironwood`, `8`). */
  V7x = 8,
  /** @brief Cloud TPU v4 Lite (`Pufferfish Lite`, `9`). */
  V4Lite = 9,
};

/**
 * @brief Zero-exception RAII status wrapper over `LibTpuCStatus`.
 */
class Status {
public:
  /**
   * @brief Constructs a `Status` taking ownership of a raw `LibTpuCStatus` and
   * an optional diagnostic message string.
   * @param raw Raw `LibTpuCStatus` returned from the C FFI.
   * @param msg_str Diagnostic message string (defaults to `""`).
   */
  explicit Status(LibTpuCStatus raw, std::string msg_str = "") noexcept;

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
  /** @brief Owned diagnostic message string of arbitrary length. */
  std::string msg_str_;
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
   * @brief Copies the on-device buffer bytes into a C++20 `std::span<uint8_t>`.
   * @param host_dst Destination mutable byte span.
   * @return `Status` indicating success or transfer error.
   */
  [[nodiscard]] Status to_host(std::span<uint8_t> host_dst) const noexcept;

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
   * @brief Returns the underlying raw `const PJRT_Buffer*` pointer.
   * @return Raw `const PJRT_Buffer*` handle.
   */
  [[nodiscard]] const PJRT_Buffer *raw() const noexcept;

  /**
   * @brief Returns the mutable raw `PJRT_Buffer*` pointer for C PJRT APIs that
   * take non-const `PJRT_Buffer*`.
   * @return Mutable raw `PJRT_Buffer*` handle.
   */
  [[nodiscard]] PJRT_Buffer *raw_mut() noexcept;

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
   * @brief Executes this SPMD program across `num_devices` addressable devices
   * simultaneously via `PJRT_LoadedExecutable_Execute`.
   * @param arg_lists Per-device input `const PJRT_Buffer* const*` arrays of
   * length `num_devices`.
   * @param num_devices Number of participating devices (`>= 1`).
   * @param num_args Number of input buffers per device.
   * @param output_lists Per-device output `PJRT_Buffer** const*` arrays of
   * length `num_devices`.
   * @return `Status` indicating multi-device execution success or failure.
   */
  [[nodiscard]] Status
  execute_sharded(const PJRT_Buffer *const *const *arg_lists,
                  size_t num_devices, size_t num_args,
                  PJRT_Buffer **const *output_lists) const noexcept;

  /**
   * @brief Returns the number of cost-analysis properties exposed by
   * `PJRT_Executable_GetCostAnalysis`.
   * @return Number of `PJRT_NamedValue` cost analysis properties (`>= 3`).
   */
  [[nodiscard]] size_t cost_analysis_property_count() const noexcept;

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
   * @brief Allocates a device `Buffer` from C++20 `std::span` dimension and
   * host byte views.
   * @param device_id Target logical TPU device ID.
   * @param dtype Element data type.
   * @param dims Span of dimension extents.
   * @param host_data Span of source host bytes.
   * @param buffer_out Output `Buffer` receiving the allocated tensor.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status buffer_from_host(uint32_t device_id, DataType dtype,
                                        std::span<const int64_t> dims,
                                        std::span<const uint8_t> host_data,
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
   * @brief Compiles and executes an N-dimensional program over two typed host
   * buffers via `libtpu_rs_compile_and_execute_nd`.
   * @param device_id Target logical TPU device ID.
   * @param format Null-terminated format string (`"mlir"` or `"hlo"`).
   * @param code Pointer to program source or bytecode bytes.
   * @param code_len Byte length of `code`.
   * @param dtype Element data type.
   * @param dims Pointer to N-dimensional extent array.
   * @param num_dims Rank (`number of dimensions`).
   * @param in0_data First input host byte buffer.
   * @param in1_data Second input host byte buffer.
   * @param byte_len Byte length of `in0_data`, `in1_data`, and `out_data`.
   * @param out_data Destination host byte buffer of length `byte_len`.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status
  compile_and_execute_nd(uint32_t device_id, const char *format,
                         const uint8_t *code, size_t code_len, DataType dtype,
                         const int64_t *dims, size_t num_dims,
                         const uint8_t *in0_data, const uint8_t *in1_data,
                         size_t byte_len, uint8_t *out_data) noexcept;

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

  /**
   * @brief Queries the 3D process coordinates `(x, y, z)` for `process_id` via
   * `PJRT_TpuTopology_ProcessCoordFromId`.
   * @param process_id Target process ID.
   * @param coords_out Output 3-element `int32_t` array receiving `(x, y, z)`.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status
  process_coord_from_id(size_t process_id,
                        int32_t coords_out[3]) const noexcept;

  /**
   * @brief Queries the global chip ID for 3D chip coordinates `(x, y, z)` via
   * `PJRT_TpuTopology_ChipIdFromCoord`.
   * @param x Chip X coordinate.
   * @param y Chip Y coordinate.
   * @param z Chip Z coordinate.
   * @return Global chip ID.
   */
  [[nodiscard]] size_t chip_id_from_coord(int32_t x, int32_t y,
                                          int32_t z) const noexcept;

  /**
   * @brief Computes the bounding subslice origin `(x, y, z)` for `device_ids`
   * via `PJRT_TpuTopology_ComputeSubsliceForDevices`.
   * @param device_ids Pointer to logical device IDs array.
   * @param num_devices Number of device IDs.
   * @param origin_out Output 3-element `int32_t` array receiving the origin.
   * @return `Status` indicating success or failure.
   */
  [[nodiscard]] Status
  compute_subslice_for_devices(const int32_t *device_ids, size_t num_devices,
                               int32_t origin_out[3]) const noexcept;

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

  /**
   * @brief Queries and immediately destroys the default TPU memory layout via
   * `PJRT_Layouts_PJRT_Topology_GetDefaultLayout`.
   * @return `true` if a non-null `PJRT_Layouts_MemoryLayout*` was allocated.
   */
  [[nodiscard]] bool default_layout_allocated() const noexcept;

  /**
   * @brief Creates a `PJRT_RawBuffer` alias of `buf`, queries its device
   * pointer, and destroys the alias handle.
   * @param buf Source on-device `Buffer`.
   * @return Non-zero device pointer address.
   */
  [[nodiscard]] size_t create_raw_buffer_device_ptr(Buffer &buf) const noexcept;

  /**
   * @brief Initiates a simulated cross-host DCN transfer via
   * `PJRT_Transfers_CrossHostSend`.
   * @param transfer_tag Monotonic transfer tag.
   * @return Completed transfer status value (`>= 1`).
   */
  [[nodiscard]] size_t cross_host_send(size_t transfer_tag) const noexcept;

  /**
   * @brief Queries the serialized `PjRtRuntimeAbiVersionProto` byte length via
   * `PJRT_AbiVersion_RuntimeToProto`.
   * @return Serialized runtime ABI version byte length (`> 0`).
   */
  [[nodiscard]] size_t runtime_abi_version_proto_size() const noexcept;

  /**
   * @brief Queries the serialized `PjRtExecutableAbiVersionProto` byte length
   * via `PJRT_AbiVersion_ExecutableToProto`.
   * @return Serialized executable ABI version byte length (`> 0`).
   */
  [[nodiscard]] size_t executable_abi_version_proto_size() const noexcept;

  /**
   * @brief Queries the 64-bit platform ID via
   * `PJRT_AbiVersion_RuntimePlatformId`.
   * @return Non-zero TPU platform ID (`0x54505500`).
   */
  [[nodiscard]] size_t abi_platform_id() const noexcept;

  /**
   * @brief Allocates and immediately destroys a `PJRT_ExecutableMetadata`
   * holder via `PJRT_ExecutableMetadata_Get` and
   * `PJRT_ExecutableMetadata_Destroy`.
   * @return `true` if a non-null metadata holder was allocated.
   */
  [[nodiscard]] bool executable_metadata_allocated() const noexcept;

  /**
   * @brief Allocates and immediately frees `size` bytes of pinned host memory
   * via `PJRT_HostMemoryAllocator_Allocate`.
   * @param size Requested byte size (`>= 1`).
   * @return `true` if a non-null host buffer was allocated and freed.
   */
  [[nodiscard]] bool allocate_host_memory(size_t size) const noexcept;

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
   * @brief Move constructor transferring ownership of the phase compiler
   * handle.
   * @param other Rvalue phase compiler to move from.
   */
  PhaseCompiler(PhaseCompiler &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the phase
   * compiler handle.
   * @param other Rvalue phase compiler to move from.
   * @return Reference to this phase compiler.
   */
  PhaseCompiler &operator=(PhaseCompiler &&other) noexcept;

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
   * @brief Move constructor transferring ownership of the Megascale context.
   * @param other Rvalue Megascale context to move from.
   */
  MegascaleContext(MegascaleContext &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the Megascale
   * context.
   * @param other Rvalue Megascale context to move from.
   * @return Reference to this Megascale context.
   */
  MegascaleContext &operator=(MegascaleContext &&other) noexcept;

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

  /**
   * @brief Returns the strongly-typed `TpuEmbeddingEngineState*` handle.
   * @return Non-null `TpuEmbeddingEngineState*` handle.
   */
  [[nodiscard]] const TpuEmbeddingEngineState *state_handle() const noexcept;

  /**
   * @brief Runs `TpuEmbeddingEngine_ExecutePartitioner` on `config_bytes`.
   * @param config_bytes Pointer to serialized `TpuEmbeddingConfigurationProto`
   * bytes.
   * @param config_size Byte length of `config_bytes`.
   * @return Partitioned common configuration string.
   */
  [[nodiscard]] std::string partition_parameters(const char *config_bytes,
                                                 size_t config_size) noexcept;

  /**
   * @brief Writes float parameters into `table_name` via
   * `TpuEmbeddingEngine_WriteParameters`.
   * @param table_name Null-terminated embedding table name.
   * @param table_name_len Byte length of `table_name`.
   * @param values Source `float` parameter array.
   * @param count Number of `float` elements in `values`.
   * @return Number of `float` parameters written.
   */
  [[nodiscard]] size_t write_parameters(const char *table_name,
                                        size_t table_name_len,
                                        const float *values,
                                        size_t count) noexcept;

  /**
   * @brief Reads `count` float parameters from `table_name` into `out_values`
   * via `TpuEmbeddingEngine_ReadParameters`.
   * @param table_name Null-terminated embedding table name.
   * @param table_name_len Byte length of `table_name`.
   * @param out_values Destination `float` parameter array.
   * @param count Number of `float` elements to read.
   * @return Number of `float` parameters read from `table_name`.
   */
  [[nodiscard]] size_t read_parameters(const char *table_name,
                                       size_t table_name_len, float *out_values,
                                       size_t count) noexcept;

  /**
   * @brief Builds the XLA computation for receiving SparseCore embedding
   * activations via `TpuEmbeddingEngine_RecvActivationsComputation`.
   * @return Serialized XLA HLO computation byte size (`> 0`).
   */
  [[nodiscard]] size_t build_load_optimizer_hlo() noexcept;

private:
  /** @brief Owned `TpuEmbeddingEngineState*` handle. */
  TpuEmbeddingEngineState *state_;
};

/**
 * @brief Strongly-typed C++20 RAII wrapper over `PJRT_Collectives_Extension`
 * communicator handles.
 */
class Communicator {
public:
  /**
   * @brief Creates an ICI/DCN `Communicator` via
   * `PJRT_Collectives_CreateCommunicator`.
   */
  Communicator() noexcept;

  /**
   * @brief Destroys the `Communicator` via `PJRT_Communicator_Destroy`.
   */
  ~Communicator() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other communicator.
   */
  Communicator(const Communicator &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other communicator.
   * @return Reference to this communicator.
   */
  Communicator &operator=(const Communicator &other) = delete;

  /**
   * @brief Move constructor transferring ownership of the communicator handle.
   * @param other Rvalue communicator to move from.
   */
  Communicator(Communicator &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the communicator
   * handle.
   * @param other Rvalue communicator to move from.
   * @return Reference to this communicator.
   */
  Communicator &operator=(Communicator &&other) noexcept;

  /**
   * @brief Executes an `AllReduce` operation copying/reducing from `send_buf`
   * into `recv_buf`.
   * @param send_buf Source on-device `Buffer`.
   * @param recv_buf Destination on-device `Buffer`.
   * @param count Number of elements to reduce.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status all_reduce(Buffer &send_buf, Buffer *recv_buf,
                                  size_t count) noexcept;

  /**
   * @brief Executes a `ReduceScatter` operation from `send_buf` into
   * `recv_buf`.
   * @param send_buf Source on-device `Buffer`.
   * @param recv_buf Destination on-device `Buffer`.
   * @param count Number of elements to reduce-scatter.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status reduce_scatter(Buffer &send_buf, Buffer *recv_buf,
                                      size_t count) noexcept;

  /**
   * @brief Executes an `AllGather` operation from `send_buf` into `recv_buf`.
   * @param send_buf Source on-device `Buffer`.
   * @param recv_buf Destination on-device `Buffer`.
   * @param count Number of elements to gather.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status all_gather(Buffer &send_buf, Buffer *recv_buf,
                                  size_t count) noexcept;

  /**
   * @brief Executes an `AllToAll` operation from `send_buf` into `recv_buf`.
   * @param send_buf Source on-device `Buffer`.
   * @param recv_buf Destination on-device `Buffer`.
   * @param count Number of elements to exchange.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status all_to_all(Buffer &send_buf, Buffer *recv_buf,
                                  size_t count) noexcept;

  /**
   * @brief Executes a `CollectivePermute` operation from `send_buf` into
   * `recv_buf`.
   * @param send_buf Source on-device `Buffer`.
   * @param recv_buf Destination on-device `Buffer`.
   * @param count Number of elements to permute.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status collective_permute(Buffer &send_buf, Buffer *recv_buf,
                                          size_t count) noexcept;

  /**
   * @brief Queries the communicator description string via
   * `PJRT_Communicator_ToString`.
   * @return Communicator description `std::string`.
   */
  [[nodiscard]] std::string to_string() const noexcept;

private:
  /** @brief Owned strongly-typed `PJRT_Collectives_Communicator*` handle. */
  PJRT_Collectives_Communicator *handle_;
};

/**
 * @brief Strongly-typed C++20 move-only RAII wrapper over
 * `PJRT_AsyncHostToDeviceTransferManager*`.
 */
class AsyncTransferManager {
public:
  /**
   * @brief Creates an `AsyncTransferManager` for `client` via
   * `PJRT_Client_CreateBuffersForAsyncHostToDevice`.
   * @param client Owning `Client` reference.
   */
  explicit AsyncTransferManager(const Client &client) noexcept;

  /**
   * @brief Destroys the underlying `PJRT_AsyncHostToDeviceTransferManager*`
   * handle.
   */
  ~AsyncTransferManager() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other transfer manager.
   */
  AsyncTransferManager(const AsyncTransferManager &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other transfer manager.
   * @return Reference to this transfer manager.
   */
  AsyncTransferManager &operator=(const AsyncTransferManager &other) = delete;

  /**
   * @brief Move constructor transferring ownership of the transfer manager
   * handle.
   * @param other Rvalue transfer manager to move from.
   */
  AsyncTransferManager(AsyncTransferManager &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the transfer
   * manager handle.
   * @param other Rvalue transfer manager to move from.
   * @return Reference to this transfer manager.
   */
  AsyncTransferManager &operator=(AsyncTransferManager &&other) noexcept;

  /**
   * @brief Transfers `transfer_size` bytes from `data` into buffer
   * `buffer_index`.
   * @param buffer_index Target buffer index (`>= 0`).
   * @param data Host byte buffer pointer.
   * @param offset Byte offset into target buffer.
   * @param transfer_size Number of bytes to transfer.
   * @param is_last_transfer `true` if this chunk completes the buffer transfer.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status transfer_data(int32_t buffer_index, const void *data,
                                     int64_t offset, int64_t transfer_size,
                                     bool is_last_transfer) noexcept;

  /**
   * @brief Retrieves the completed `Buffer` at `buffer_index`.
   * @param buffer_index Target buffer index (`>= 0`).
   * @param out_buffer Output `Buffer` receiving ownership of the device buffer.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status retrieve_buffer(int32_t buffer_index,
                                       Buffer *out_buffer) noexcept;

  /**
   * @brief Queries the byte size of buffer `buffer_index`.
   * @param buffer_index Target buffer index (`>= 0`).
   * @return Byte size of buffer `buffer_index`.
   */
  [[nodiscard]] size_t buffer_size(int32_t buffer_index) const noexcept;

  /**
   * @brief Queries the target `PJRT_Device*` associated with this transfer
   * manager.
   * @return Raw `PJRT_Device*` pointer (or `nullptr`).
   */
  [[nodiscard]] PJRT_Device *device() const noexcept;

private:
  /** @brief Owned `PJRT_AsyncHostToDeviceTransferManager*` handle. */
  PJRT_AsyncHostToDeviceTransferManager *handle_;
};

/**
 * @brief Strongly-typed C++20 inspector over `PJRT_TpuExecutable_Extension`
 * (`ext_type = 17`).
 */
class TpuExecutableInspector {
public:
  /**
   * @brief Constructs a `TpuExecutableInspector` over an optional
   * `PJRT_Executable*`.
   * @param executable Raw `PJRT_Executable*` handle (defaults to `nullptr`).
   */
  explicit TpuExecutableInspector(
      PJRT_Executable *executable = nullptr) noexcept;

  /**
   * @brief Queries the serialized `TargetArgumentsProto` byte length via
   * `PJRT_TpuExecutable_GetTargetArguments`.
   * @return Serialized `TargetArgumentsProto` size in bytes (`> 0`).
   */
  [[nodiscard]] size_t target_arguments_size() const noexcept;

  /**
   * @brief Queries the serialized `HloModuleProtoWithConfig` byte length via
   * `PJRT_TpuExecutable_GetHloModuleWithConfig`.
   * @return Serialized `HloModuleProtoWithConfig` size in bytes (`> 0`).
   */
  [[nodiscard]] size_t hlo_module_with_config_size() const noexcept;

  /**
   * @brief Queries the generated code size in bytes via
   * `PJRT_TpuExecutable_GetCompiledMemoryStats`.
   * @return Generated VLIW code size in bytes (`> 0`).
   */
  [[nodiscard]] int64_t generated_code_size_bytes() const noexcept;

  /**
   * @brief Sets and reads back a `TpuCompilationEnvironmentProto` flag via
   * `PJRT_TpuExecutable_SetTpuCompilationEnv` and
   * `PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`.
   * @param field_tag Protobuf field number (`>= 1`).
   * @return Current number of compilation environment overrides (`>= 1`).
   */
  [[nodiscard]] size_t
  set_and_query_compilation_env(size_t field_tag) const noexcept;

  /**
   * @brief Queries the serialized `xla.BufferAssignmentProto` byte length for
   * `loaded` via `libtpu_rs_executable_buffer_assignment_proto`.
   * @param loaded Reference to a `LoadedExecutable`.
   * @return Serialized `BufferAssignmentProto` size in bytes (`0` if null).
   */
  [[nodiscard]] size_t
  buffer_assignment_size(const LoadedExecutable &loaded) const noexcept;

private:
  /** @brief Non-owning `PJRT_Executable*` handle. */
  PJRT_Executable *executable_;
};

/**
 * @brief Strongly-typed C++20 inspector over `PJRT_Shardings_Extension`
 * (`ext_type = 19`).
 */
class ShardingsInspector {
public:
  /**
   * @brief Constructs a `ShardingsInspector` over an optional
   * `PJRT_Executable*`.
   * @param executable Raw `PJRT_Executable*` handle (defaults to `nullptr`).
   */
  explicit ShardingsInspector(PJRT_Executable *executable = nullptr) noexcept;

  /**
   * @brief Queries the number of parameter `xla.OpSharding` entries via
   * `PJRT_Shardings_ParameterShardings`.
   * @return Number of parameter shardings (`>= 1`).
   */
  [[nodiscard]] size_t parameter_sharding_count() const noexcept;

  /**
   * @brief Queries the number of output `xla.OpSharding` entries via
   * `PJRT_Shardings_OutputShardings`.
   * @return Number of output shardings (`>= 1`).
   */
  [[nodiscard]] size_t output_sharding_count() const noexcept;

  /**
   * @brief Queries the serialized `xla.OpSharding` protobuf bytes for parameter
   * `index` via `PJRT_Shardings_ParameterShardings`.
   * @param index Zero-based parameter index.
   * @return Serialized `xla.OpSharding` wire bytes (or `""` if `index` is out
   * of range).
   */
  [[nodiscard]] std::string
  parameter_sharding_bytes(size_t index) const noexcept;

  /**
   * @brief Queries the serialized `xla.OpSharding` protobuf bytes for output
   * `index` via `PJRT_Shardings_OutputShardings`.
   * @param index Zero-based output index.
   * @return Serialized `xla.OpSharding` wire bytes (or `""` if `index` is out
   * of range).
   */
  [[nodiscard]] std::string output_sharding_bytes(size_t index) const noexcept;

private:
  /** @brief Non-owning `PJRT_Executable*` handle. */
  PJRT_Executable *executable_;
};

/** @brief Type alias for `Communicator` (`CollectiveCommunicator`). */
using CollectiveCommunicator = Communicator;

/** @brief Type alias for `Communicator` (`CollectivesCommunicator`). */
using CollectivesCommunicator = Communicator;

/** @brief Type alias for `MegascaleContext` (`MegaScaleContext`). */
using MegaScaleContext = MegascaleContext;

/** @brief Type alias for `MegascaleContext` (`MultiSliceConfig`). */
using MultiSliceConfig = MegascaleContext;

/** @brief Type alias for `MegascaleContext` (`MultiSliceCoordinator`). */
using MultiSliceCoordinator = MegascaleContext;

/** @brief Type alias for `EmbeddingEngine` (`SparseCoreEngine`). */
using SparseCoreEngine = EmbeddingEngine;

/** @brief Type alias for `Buffer` (`DlPackTensor`). */
using DlPackTensor = Buffer;

/**
 * @brief Strongly-typed C++20 wrapper for registering OpenXLA Typed FFI v0.2
 * `XLA_FFI_Handler` custom call targets (`FfiRegistry`).
 */
class FfiRegistry {
public:
  /**
   * @brief Registers an OpenXLA Typed FFI v0.2 `XLA_FFI_Handler` callback under
   * `target_name` (`libtpu::register_ffi_handler`).
   * @param target_name Null-terminated UTF-8 custom-call target name.
   * @param handler Function pointer matching `XLA_FFI_Handler`.
   * @return `Status` indicating registration success or failure.
   */
  [[nodiscard]] static Status
  register_handler(const char *target_name, XLA_FFI_Handler handler) noexcept;
};

/**
 * @brief Strongly-typed C++20 move-only RAII wrapper over
 * `DistributedCoordinator` (`libtpu_rs_coordinator_*`).
 */
class Coordinator {
public:
  /**
   * @brief Creates a new in-memory `Coordinator` via
   * `libtpu_rs_coordinator_create`.
   */
  Coordinator() noexcept;

  /**
   * @brief Destroys the underlying coordinator handle via
   * `libtpu_rs_coordinator_destroy`.
   */
  ~Coordinator() noexcept;

  /**
   * @brief Deleted copy constructor.
   * @param other Other coordinator.
   */
  Coordinator(const Coordinator &other) = delete;

  /**
   * @brief Deleted copy assignment operator.
   * @param other Other coordinator.
   * @return Reference to this coordinator.
   */
  Coordinator &operator=(const Coordinator &other) = delete;

  /**
   * @brief Move constructor transferring ownership of the coordinator handle.
   * @param other Rvalue coordinator to move from.
   */
  Coordinator(Coordinator &&other) noexcept;

  /**
   * @brief Move assignment operator transferring ownership of the coordinator
   * handle.
   * @param other Rvalue coordinator to move from.
   * @return Reference to this coordinator.
   */
  Coordinator &operator=(Coordinator &&other) noexcept;

  /**
   * @brief Inserts a key-value entry into this coordinator via
   * `libtpu_rs_coordinator_put_kv`.
   * @param key Null-terminated UTF-8 key string.
   * @param value UTF-8 value string.
   * @return `Status` indicating completion status.
   */
  [[nodiscard]] Status put_key_value(const char *key,
                                     const std::string &value) noexcept;

  /**
   * @brief Looks up a key-value entry from this coordinator via
   * `libtpu_rs_coordinator_get_kv`.
   * @param key Null-terminated UTF-8 key string.
   * @return Value string (or `""` if not found).
   */
  [[nodiscard]] std::string get_key_value(const char *key) const noexcept;

  /**
   * @brief Invokes one of the 19 `tensorflow.CoordinationService` RPC methods
   * using 5-byte gRPC Length-Prefixed-Message framing via
   * `libtpu_rs_coordinator_invoke_grpc_method`.
   * @param grpc_path Null-terminated HTTP/2 gRPC method path.
   * @param req_frame 5-byte gRPC-framed request message bytes.
   * @return 5-byte gRPC-framed response message bytes (or `""` on error).
   */
  [[nodiscard]] std::string
  invoke_grpc_method(const char *grpc_path,
                     const std::string &req_frame) const noexcept;

private:
  /** @brief Owned coordinator handle pointer. */
  void *handle_;
};

/**
 * @brief Emits a canonical `proto3` `.proto` schema definition string from the
 * compiled Rust protobuf descriptor for `full_message_name`.
 * @param full_message_name Null-terminated fully-qualified protobuf message
 * name (e.g., `"xla.TileProto"`).
 * @return Canonical `.proto` schema text, or `""` if `full_message_name` is
 * unknown.
 */
[[nodiscard]] std::string
emit_proto_schema(const char *full_message_name) noexcept;

/**
 * @brief Disassembles a `LoadedExecutable` into a multi-line `DeepSea` LLO
 * assembly listing (`libtpu_rs_disassemble_executable_llo`).
 * @param executable Reference to a `LoadedExecutable`.
 * @return Multi-line `DeepSea` LLO assembly string (or `""` if uninitialized).
 */
[[nodiscard]] std::string
disassemble_executable_llo(const LoadedExecutable &executable) noexcept;

/**
 * @brief Decodes a raw or Riegeli-framed `asic_sw.CoreDump` payload into
 * canonical JSON (`libtpu_rs_read_core_dump`).
 * @param raw_bytes Pointer to raw or Riegeli-framed `CoreDump` bytes.
 * @param raw_len Byte length of `raw_bytes`.
 * @return Canonical JSON string (or `""` on invalid input).
 */
[[nodiscard]] std::string read_core_dump_json(const uint8_t *raw_bytes,
                                              size_t raw_len) noexcept;

/**
 * @brief Decodes a raw or Riegeli-framed `AllCoreStateSummaries` payload into
 * canonical JSON (`libtpu_rs_read_stall_summaries`).
 * @param raw_bytes Pointer to raw or Riegeli-framed `AllCoreStateSummaries`
 * bytes.
 * @param raw_len Byte length of `raw_bytes`.
 * @return Canonical JSON string (or `""` on invalid input).
 */
[[nodiscard]] std::string read_stall_summaries_json(const uint8_t *raw_bytes,
                                                    size_t raw_len) noexcept;

/**
 * @brief Decodes a raw or Riegeli-framed `CompilerMetadata` payload into
 * canonical JSON (`libtpu_rs_read_compiler_metadata`).
 * @param raw_bytes Pointer to raw or Riegeli-framed `CompilerMetadata` bytes.
 * @param raw_len Byte length of `raw_bytes`.
 * @return Canonical JSON string (or `""` on invalid input).
 */
[[nodiscard]] std::string read_compiler_metadata_json(const uint8_t *raw_bytes,
                                                      size_t raw_len) noexcept;

/**
 * @brief Registers an OpenXLA Typed FFI v0.2 `XLA_FFI_Handler` callback under
 * `target_name` (`libtpu_rs_register_ffi_handler`).
 * @param target_name Null-terminated UTF-8 custom-call target name.
 * @param handler Function pointer matching `XLA_FFI_Handler`.
 * @return `Status` indicating registration success or failure.
 */
[[nodiscard]] Status register_ffi_handler(const char *target_name,
                                          XLA_FFI_Handler handler) noexcept;

/**
 * @brief Formats a human-readable physical register allocation and `VMEM` spill
 * summary for `executable` (`libtpu_rs_inspect_register_allocation`).
 * @param executable Reference to a `LoadedExecutable`.
 * @return Register allocation summary string (or `""` if uninitialized).
 */
[[nodiscard]] std::string
inspect_register_allocation(const LoadedExecutable &executable) noexcept;

/**
 * @brief Serializes the wire-compatible `xla.BufferAssignmentProto` for
 * `executable` (`libtpu_rs_executable_buffer_assignment_proto`).
 * @param executable Reference to a `LoadedExecutable`.
 * @return Serialized `BufferAssignmentProto` bytes (or `""` if null).
 */
[[nodiscard]] std::string
executable_buffer_assignment_proto(const LoadedExecutable &executable) noexcept;

/**
 * @brief Formats a canonical hardware chip configuration summary for the active
 * TPU generation (`libtpu_rs_query_chip_config_summary`).
 * @return Hardware chip configuration summary string.
 */
[[nodiscard]] std::string query_chip_config_summary() noexcept;

/**
 * @brief Computes the shortest 3D Torus optical ICI hop count between two chip
 * coordinates (`libtpu_rs_compute_ici_route`).
 * @param src_x Source X chip coordinate.
 * @param src_y Source Y chip coordinate.
 * @param src_z Source Z chip coordinate.
 * @param dst_x Destination X chip coordinate.
 * @param dst_y Destination Y chip coordinate.
 * @param dst_z Destination Z chip coordinate.
 * @param wrap_around Whether twisted-torus wrap-around links are enabled.
 * @return Shortest hop count along the 3D Torus mesh.
 */
[[nodiscard]] uint32_t compute_ici_route(uint32_t src_x, uint32_t src_y,
                                         uint32_t src_z, uint32_t dst_x,
                                         uint32_t dst_y, uint32_t dst_z,
                                         bool wrap_around = true) noexcept;

/**
 * @brief Resolves a hierarchical `DeepSea` CSR path in the 13,261+ CSR catalog
 * (`libtpu_rs_lookup_csr_by_path`).
 * @param path Null-terminated hierarchical CSR path string.
 * @return Physical BAR0 MMIO byte offset, or `UINT64_MAX` if not found.
 */
[[nodiscard]] uint64_t lookup_csr_by_path(const char *path) noexcept;

/**
 * @brief Resolves a physical BAR0 MMIO byte offset in the 13,261+ CSR catalog
 * (`libtpu_rs_lookup_csr_by_offset`).
 * @param byte_offset Physical BAR0 MMIO byte offset.
 * @return Canonical hierarchical CSR register path (or `""` if not found).
 */
[[nodiscard]] std::string lookup_csr_by_offset(uint64_t byte_offset) noexcept;

/**
 * @brief Encodes a textual MLIR module into a standard upstream `ML\xefR`
 * bytecode container and compiles it into `executable_out`
 * (`libtpu_rs_compile_standard_mlir_bytecode`).
 * @param client Reference to an active `Client`.
 * @param mlir_source Null-terminated UTF-8 MLIR module text.
 * @param executable_out Output `LoadedExecutable` receiving the compiled
 * program.
 * @return `Status` indicating compilation success or failure.
 */
[[nodiscard]] Status
compile_standard_mlir_bytecode(const Client &client, const char *mlir_source,
                               LoadedExecutable *executable_out) noexcept;

/**
 * @brief Collects serialized `tensorflow.ProfileToolData` bytes for `tool_name`
 * (`libtpu_rs_profiler_collect_tool_data`).
 * @param tool_name Null-terminated tool name (e.g., `"trace_viewer.json"`,
 * `"input_pipeline.pb"`).
 * @return Serialized tool data payload bytes (or `""` if null).
 */
[[nodiscard]] std::string
collect_profiler_tool_data(const char *tool_name) noexcept;

/**
 * @brief Verifies all 293 embedded `protodesc_cold` descriptor catalog mappings
 * (`libtpu_rs_verify_all_293_protodesc_catalog`).
 * @return Verified catalog entry count (`293`).
 */
[[nodiscard]] size_t verify_protodesc_293_catalog() noexcept;

/**
 * @brief Executes a compiled `LoadedExecutable` on a dual-`TensorCore` +
 * quad-`SparseCore` `MultiCoreChipVmState` (`libtpu_rs_multicore_vm_execute`).
 * @param executable Reference to a compiled `LoadedExecutable`.
 * @param core0_in Pointer to `tc_0` input bytes.
 * @param core0_len Byte length of `core0_in`.
 * @param core1_in Pointer to `tc_1` input bytes.
 * @param core1_len Byte length of `core1_in`.
 * @param out_core0 Destination buffer receiving `tc_0` output bytes.
 * @param max_out0 Capacity of `out_core0` in bytes.
 * @param out_sparse_lookups Optional output pointer receiving completed
 * `SparseCore` lookup count.
 * @return `Status` indicating execution success or failure.
 */
[[nodiscard]] Status
execute_multicore_vm(const LoadedExecutable &executable,
                     const uint8_t *core0_in, size_t core0_len,
                     const uint8_t *core1_in, size_t core1_len,
                     uint8_t *out_core0, size_t max_out0,
                     uint64_t *out_sparse_lookups = nullptr) noexcept;

/**
 * @brief Assembles a multi-line `DeepSea` LLO VLIW text listing into 64-byte
 * `IMEM` binary words (`libtpu_assemble_llo_vliw`).
 * @param listing Null-terminated UTF-8 `DeepSea` LLO VLIW assembly text.
 * @return Assembled `IMEM` bytes as a `std::string` (or `""` on syntax error).
 */
[[nodiscard]] std::string assemble_llo_vliw(const char *listing) noexcept;

/**
 * @brief Disassembles 64-byte `IMEM` binary words into a canonical `DeepSea`
 * LLO VLIW text listing (`libtpu_disassemble_llo_vliw`).
 * @param imem_bytes Pointer to 64-byte-aligned `IMEM` bundle words.
 * @param imem_len Byte length of `imem_bytes`.
 * @return Disassembled LLO VLIW text listing (or `""` if `imem_len < 64`).
 */
[[nodiscard]] std::string disassemble_llo_vliw(const uint8_t *imem_bytes,
                                               size_t imem_len) noexcept;

/**
 * @brief Dispatches a wire-encoded `HostCommandEnvelopeProto` on the TPU
 * continuation queue (`libtpu_rs_service_host_command`).
 * @param envelope_bytes Pointer to wire-encoded `HostCommandEnvelopeProto`
 * bytes.
 * @param envelope_len Byte length of `envelope_bytes`.
 * @param out_ack_sync Output pointer receiving the acknowledged BAR0 `SyncFlag`
 * value.
 * @return `Status` indicating continuation servicing success or failure.
 */
[[nodiscard]] Status service_host_command(const uint8_t *envelope_bytes,
                                          size_t envelope_len,
                                          uint64_t *out_ack_sync) noexcept;

/**
 * @brief Returns a formatted summary combining the LLO VLIW bundle schedule,
 * physical register allocation, and 3D ICI optical routing hop count for a
 * compiled `LoadedExecutable`.
 * @param executable Reference to a compiled `LoadedExecutable`.
 * @return Summary string (or `""` if `executable` is null).
 */
[[nodiscard]] std::string
query_llo_schedule_summary(const LoadedExecutable &executable) noexcept;

/**
 * @brief Zero-exception monadic value-or-status container (`Result<T>`) for
 * C++20 callers.
 * @tparam T Contained value type.
 */
template <typename T> class Result {
public:
  /**
   * @brief Constructs a successful `Result<T>` holding `value`.
   * @param value Contained success value.
   */
  explicit Result(T value) noexcept
      : status_(LibTpuCStatus{0, nullptr}, ""),
        value_(static_cast<T &&>(value)) {}

  /**
   * @brief Constructs a failed `Result<T>` holding `status` and a default `T`.
   * @param status Non-OK `Status` instance.
   */
  explicit Result(Status &&status) noexcept
      : status_(static_cast<Status &&>(status)), value_{} {}

  /**
   * @brief Returns `true` if the contained `Status` is `OK` (`code == 0`).
   * @return `true` on success, `false` on error.
   */
  [[nodiscard]] bool is_ok() const noexcept { return status_.is_ok(); }

  /**
   * @brief Returns a const reference to the underlying `Status`.
   * @return Const reference to `Status`.
   */
  [[nodiscard]] const Status &status() const noexcept { return status_; }

  /**
   * @brief Returns a const reference to the contained value `T`.
   * @return Const reference to `value_`.
   */
  [[nodiscard]] const T &value() const noexcept { return value_; }

private:
  /** @brief Underlying operation status. */
  Status status_;
  /** @brief Contained value payload. */
  T value_;
};

} // namespace libtpu

#endif // LIBTPU_HPP_
