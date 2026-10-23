/**
 * @file libtpu.cpp
 * @brief Implementation of the strongly-typed C++20 RAII wrapper classes for
 * `libtpu-rs` driving `GetPjrtApi()`, `GetLibtpuSdkApi()`, `TpuTopology_*()`,
 * and `TpuProfiler_*()` directly.
 */

#include "libtpu.hpp"

namespace libtpu {

/**
 * @brief Converts a `PJRT_Error*` returned by a `GetPjrtApi()` slot into a
 * `Status` and frees `err`.
 * @param err Pointer to `PJRT_Error` (or `nullptr` on success).
 * @return `Status` representing success (`0`) or the queried `PJRT_Error_Code`.
 */
static Status pjrt_error_to_status(PJRT_Error *err) noexcept {
  if (err == nullptr) {
    return Status(LibTpuCStatus{0, nullptr});
  }
  uintptr_t code_args[4] = {32, 0, reinterpret_cast<uintptr_t>(err), 0};
  GetPjrtApi()->pjrt_error_get_code(code_args);
  uintptr_t del_args[3] = {24, 0, reinterpret_cast<uintptr_t>(err)};
  GetPjrtApi()->pjrt_error_destroy(del_args);
  return Status(LibTpuCStatus{static_cast<int32_t>(code_args[3]), nullptr});
}

Status::Status(LibTpuCStatus raw) noexcept : raw_(raw) {}

Status::~Status() noexcept { libtpu_rs_status_free(raw_); }

Status::Status(Status &&other) noexcept : raw_(other.raw_) {
  other.raw_.code = 0;
  other.raw_.message = nullptr;
}

Status &Status::operator=(Status &&other) noexcept {
  LibTpuCStatus next = other.raw_;
  other.raw_.code = 0;
  other.raw_.message = nullptr;
  libtpu_rs_status_free(raw_);
  raw_ = next;
  return *this;
}

bool Status::is_ok() const noexcept { return raw_.code == 0; }

int32_t Status::code() const noexcept { return raw_.code; }

const char *Status::message() const noexcept {
  return raw_.code == 0 ? "" : "PJRT error";
}

Buffer::Buffer(PJRT_Buffer *handle) noexcept : handle_(handle) {}

Buffer::~Buffer() noexcept {
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_buffer_destroy(args);
}

Buffer::Buffer(Buffer &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  PJRT_Buffer *next = other.handle_;
  other.handle_ = nullptr;
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_buffer_destroy(args);
  handle_ = next;
  return *this;
}

Status Buffer::to_host(uint8_t *host_dst, size_t len) const noexcept {
  uintptr_t args[7] = {56,
                       0,
                       reinterpret_cast<uintptr_t>(handle_),
                       0,
                       reinterpret_cast<uintptr_t>(host_dst),
                       len,
                       0};
  return pjrt_error_to_status(GetPjrtApi()->pjrt_buffer_to_host_buffer(args));
}

size_t Buffer::rank() const noexcept {
  uintptr_t args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_), 0, 0};
  GetPjrtApi()->pjrt_buffer_dimensions(args);
  return static_cast<size_t>(args[4]);
}

int64_t Buffer::dimension(size_t axis) const noexcept {
  uintptr_t args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_), 0, 0};
  GetPjrtApi()->pjrt_buffer_dimensions(args);
  const int64_t *dims_ptr = reinterpret_cast<const int64_t *>(args[3]);
  return dims_ptr[axis];
}

DataType Buffer::element_type() const noexcept {
  uintptr_t args[4] = {32, 0, reinterpret_cast<uintptr_t>(handle_), 0};
  GetPjrtApi()->pjrt_buffer_element_type(args);
  return static_cast<DataType>(static_cast<int32_t>(args[3]));
}

Status Buffer::copy_to_device(PJRT_Client *client, uint32_t dst_device_id,
                              Buffer *out_buffer) const noexcept {
  uintptr_t look_args[5] = {40, 0, reinterpret_cast<uintptr_t>(client),
                            static_cast<uintptr_t>(dst_device_id), 0};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_lookup_device(look_args);
  if (err == nullptr) {
    uintptr_t cp_args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_),
                            look_args[4], 0};
    GetPjrtApi()->pjrt_buffer_copy_to_device(cp_args);
    *out_buffer = Buffer(reinterpret_cast<PJRT_Buffer *>(cp_args[4]));
  }
  return pjrt_error_to_status(err);
}

const PJRT_Buffer *Buffer::raw() const noexcept { return handle_; }

LoadedExecutable::LoadedExecutable(PJRT_LoadedExecutable *handle) noexcept
    : handle_(handle) {}

LoadedExecutable::~LoadedExecutable() noexcept {
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_loaded_executable_destroy(args);
}

LoadedExecutable::LoadedExecutable(LoadedExecutable &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

LoadedExecutable &
LoadedExecutable::operator=(LoadedExecutable &&other) noexcept {
  PJRT_LoadedExecutable *next = other.handle_;
  other.handle_ = nullptr;
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_loaded_executable_destroy(args);
  handle_ = next;
  return *this;
}

Status LoadedExecutable::execute(uint32_t device_id,
                                 const PJRT_Buffer *const *inputs,
                                 size_t num_inputs, PJRT_Buffer **outputs_out,
                                 size_t num_outputs) const noexcept {
  (void)num_inputs;
  (void)num_outputs;
  const PJRT_Buffer *const *arg_lists[1] = {inputs};
  PJRT_Buffer **out_lists[1] = {outputs_out};
  uintptr_t ex_args[7] = {56,
                          0,
                          reinterpret_cast<uintptr_t>(handle_),
                          0,
                          reinterpret_cast<uintptr_t>(arg_lists),
                          static_cast<uintptr_t>(device_id) + 1U,
                          reinterpret_cast<uintptr_t>(out_lists)};
  return pjrt_error_to_status(
      GetPjrtApi()->pjrt_loaded_executable_execute(ex_args));
}

Status LoadedExecutable::execute_binary(uint32_t device_id,
                                        const Buffer &input0,
                                        const Buffer &input1,
                                        Buffer *output_out) const noexcept {
  const PJRT_Buffer *inputs[2] = {input0.raw(), input1.raw()};
  PJRT_Buffer *out_raw = nullptr;
  Status st = execute(device_id, inputs, 2, &out_raw, 1);
  if (st.is_ok()) {
    *output_out = Buffer(out_raw);
  }
  return st;
}

Status LoadedExecutable::serialize(uint8_t *out_bytes, size_t max_len,
                                   size_t *out_len) const noexcept {
  (void)max_len;
  uintptr_t get_ex_args[4] = {32, 0, reinterpret_cast<uintptr_t>(handle_), 0};
  GetPjrtApi()->pjrt_loaded_executable_get_executable(get_ex_args);
  uintptr_t ser_args[6] = {48, 0, get_ex_args[3], 0, 0, 0};
  GetPjrtApi()->pjrt_executable_serialize(ser_args);
  const uint8_t *src = reinterpret_cast<const uint8_t *>(ser_args[3]);
  size_t len = static_cast<size_t>(ser_args[4]);
  *out_len = len;
  for (size_t i = 0; i < len; ++i) {
    out_bytes[i] = src[i];
  }
  return Status(LibTpuCStatus{0, nullptr});
}

size_t LoadedExecutable::fingerprint(char *out_buf,
                                     size_t buf_len) const noexcept {
  (void)buf_len;
  uintptr_t fp_args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_), 0, 0};
  GetPjrtApi()->pjrt_loaded_executable_fingerprint(fp_args);
  const char *src = reinterpret_cast<const char *>(fp_args[3]);
  size_t len = static_cast<size_t>(fp_args[4]);
  for (size_t i = 0; i < len; ++i) {
    out_buf[i] = src[i];
  }
  return len;
}

float LoadedExecutable::flops() const noexcept {
  uintptr_t get_ex_args[4] = {32, 0, reinterpret_cast<uintptr_t>(handle_), 0};
  GetPjrtApi()->pjrt_loaded_executable_get_executable(get_ex_args);
  uintptr_t cost_args[5] = {40, 0, get_ex_args[3], 0, 0};
  GetPjrtApi()->pjrt_executable_get_cost_analysis(cost_args);
  return static_cast<float>(cost_args[4]);
}

const PJRT_LoadedExecutable *LoadedExecutable::raw() const noexcept {
  return handle_;
}

Client::Client(PJRT_Client *handle) noexcept : handle_(handle) {}

Client::~Client() noexcept {
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_client_destroy(args);
}

Client::Client(Client &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Client &Client::operator=(Client &&other) noexcept {
  PJRT_Client *next = other.handle_;
  other.handle_ = nullptr;
  uintptr_t args[3] = {24, 0, reinterpret_cast<uintptr_t>(handle_)};
  GetPjrtApi()->pjrt_client_destroy(args);
  handle_ = next;
  return *this;
}

Status Client::create(Client *client_out) noexcept {
  uintptr_t args[9] = {72, 0, 0, 0, 0, 0, 0, 0, 0};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_create(args);
  *client_out = Client(reinterpret_cast<PJRT_Client *>(args[8]));
  return pjrt_error_to_status(err);
}

int32_t Client::device_count() const noexcept {
  if (handle_ == nullptr) {
    return -1;
  }
  uintptr_t args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_), 0, 0};
  GetPjrtApi()->pjrt_client_addressable_devices(args);
  return static_cast<int32_t>(args[4]);
}

Status Client::buffer_from_host(uint32_t device_id, DataType dtype,
                                const int64_t *dims, size_t num_dims,
                                const uint8_t *host_data, size_t host_data_len,
                                Buffer *buffer_out) noexcept {
  (void)host_data_len;
  uintptr_t look_args[5] = {40, 0, reinterpret_cast<uintptr_t>(handle_),
                            static_cast<uintptr_t>(device_id), 0};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_lookup_device(look_args);
  if (err == nullptr) {
    uintptr_t h2d_args[14] = {
        112,
        0,
        reinterpret_cast<uintptr_t>(handle_),
        reinterpret_cast<uintptr_t>(host_data),
        static_cast<uintptr_t>(static_cast<uint32_t>(dtype)),
        reinterpret_cast<uintptr_t>(dims),
        num_dims,
        0,
        0,
        0,
        look_args[4],
        0,
        0,
        0};
    GetPjrtApi()->pjrt_client_buffer_from_host_buffer(h2d_args);
    *buffer_out = Buffer(reinterpret_cast<PJRT_Buffer *>(h2d_args[13]));
  }
  return pjrt_error_to_status(err);
}

Status Client::compile(const char *format, const uint8_t *code, size_t code_len,
                       LoadedExecutable *exec_out) noexcept {
  size_t fmt_len = 0;
  while (format[fmt_len] != '\0') {
    ++fmt_len;
  }
  uintptr_t prog[6] = {48,
                       0,
                       reinterpret_cast<uintptr_t>(code),
                       code_len,
                       reinterpret_cast<uintptr_t>(format),
                       fmt_len};
  uintptr_t comp_args[7] = {56,
                            0,
                            reinterpret_cast<uintptr_t>(handle_),
                            reinterpret_cast<uintptr_t>(prog),
                            0,
                            0,
                            0};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_compile(comp_args);
  if (err == nullptr) {
    *exec_out = LoadedExecutable(
        reinterpret_cast<PJRT_LoadedExecutable *>(comp_args[6]));
  }
  return pjrt_error_to_status(err);
}

Status Client::deserialize_executable(const uint8_t *bytes, size_t bytes_len,
                                      LoadedExecutable *exec_out) noexcept {
  uintptr_t deser_args[7] = {56,
                             0,
                             reinterpret_cast<uintptr_t>(handle_),
                             reinterpret_cast<uintptr_t>(bytes),
                             bytes_len,
                             0,
                             0};
  PJRT_Error *err =
      GetPjrtApi()->pjrt_executable_deserialize_and_load(deser_args);
  if (err == nullptr) {
    *exec_out = LoadedExecutable(
        reinterpret_cast<PJRT_LoadedExecutable *>(deser_args[5]));
  }
  return pjrt_error_to_status(err);
}

PJRT_Client *Client::raw() const noexcept { return handle_; }

Profiler::Profiler() noexcept : handle_(nullptr), status_(TpuStatus_New()) {
  TpuProfiler_Create(&handle_, status_);
}

Profiler::~Profiler() noexcept {
  TpuProfiler_Destroy(handle_);
  TpuStatus_Free(status_);
}

Profiler::Profiler(Profiler &&other) noexcept
    : handle_(other.handle_), status_(other.status_) {
  other.handle_ = nullptr;
  other.status_ = nullptr;
}

Profiler &Profiler::operator=(Profiler &&other) noexcept {
  TpuProfiler *next_handle = other.handle_;
  TpuStatus *next_status = other.status_;
  other.handle_ = nullptr;
  other.status_ = nullptr;
  TpuProfiler_Destroy(handle_);
  TpuStatus_Free(status_);
  handle_ = next_handle;
  status_ = next_status;
  return *this;
}

bool Profiler::start() noexcept {
  TpuProfiler_Start(handle_, status_);
  return TpuStatus_Ok(status_);
}

bool Profiler::stop() noexcept {
  TpuProfiler_Stop(handle_, status_);
  return TpuStatus_Ok(status_);
}

size_t Profiler::collect_size() noexcept {
  size_t sz = 0;
  TpuProfiler_CollectData(handle_, status_, nullptr, &sz);
  return sz;
}

Topology::Topology() noexcept : handle_(TpuUtil_GetTopologyPtr()) {}

int32_t Topology::logical_devices_per_host() const noexcept {
  return TpuTopology_LogicalDevicesPerHost(handle_, 0);
}

int32_t Topology::chip_bounds_x() const noexcept {
  return TpuTopology_ChipBounds_X(handle_);
}

int32_t Topology::chip_bounds_y() const noexcept {
  return TpuTopology_ChipBounds_Y(handle_);
}

int32_t Topology::chip_bounds_z() const noexcept {
  return TpuTopology_ChipBounds_Z(handle_);
}

int32_t Topology::version() const noexcept {
  return TpuTopology_Version(handle_);
}

SdkClient::SdkClient() noexcept : api_(GetLibtpuSdkApi()), handle_(nullptr) {
  LibtpuSdk_Client_Args args{nullptr};
  (void)api_->client_create(&args);
  handle_ = args.client;
}

SdkClient::~SdkClient() noexcept {
  LibtpuSdk_Client_Args args{handle_};
  (void)api_->client_destroy(&args);
}

SdkClient::SdkClient(SdkClient &&other) noexcept
    : api_(other.api_), handle_(other.handle_) {
  other.handle_ = nullptr;
}

SdkClient &SdkClient::operator=(SdkClient &&other) noexcept {
  LibtpuSdk_Client *next = other.handle_;
  other.handle_ = nullptr;
  LibtpuSdk_Client_Args args{handle_};
  (void)api_->client_destroy(&args);
  handle_ = next;
  return *this;
}

size_t SdkClient::core_state_summary_size() const noexcept {
  LibtpuSdk_GetRuntimeStatus_Args rstat_args{
      handle_, true, {0, 0, 0, 0, 0, 0, 0}, nullptr};
  (void)api_->get_runtime_status(&rstat_args);
  LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args sum_args{rstat_args.status,
                                                            nullptr, 0};
  (void)api_->runtime_status_get_core_state_summary(&sum_args);
  LibtpuSdk_RuntimeStatus_Destroy_Args del_args{rstat_args.status};
  (void)api_->runtime_status_destroy(&del_args);
  return sum_args.summary_size;
}

size_t SdkClient::metric_value_count(const char *metric_name) const noexcept {
  LibtpuSdk_GetMetric_Args m_args{handle_, metric_name, nullptr};
  (void)api_->get_metric(&m_args);
  LibtpuSdk_GetMetricValues_Args v_args{m_args.metric, nullptr, 0};
  (void)api_->get_metric_values(&v_args);
  return v_args.num_values;
}

size_t SdkClient::chip_count() const noexcept {
  LibtpuSdk_GetChipCoordinates_Args c_args{handle_, nullptr, 0};
  (void)api_->get_chip_coordinates(&c_args);
  return c_args.num_coords;
}

int64_t SdkClient::register_debug_callback(const void *callback,
                                           void *user_data) noexcept {
  LibtpuSdk_RegisterDebugCallback_Args args{handle_, callback, user_data, 0};
  (void)api_->register_debug_callback(&args);
  return args.callback_id;
}

bool SdkClient::unregister_debug_callback(int64_t callback_id) noexcept {
  LibtpuSdk_UnregisterDebugCallback_Args args{handle_, callback_id};
  LibtpuSdk_Error *err = api_->unregister_debug_callback(&args);
  if (err != nullptr) {
    struct DestroyArgs {
      LibtpuSdk_Error *error;
    } d_args{err};
    api_->error_destroy(&d_args);
    return false;
  }
  return true;
}

size_t SdkClient::trigger_debug_callbacks() const noexcept {
  return LibtpuSdk_TriggerDebugCallbacks(handle_);
}

} // namespace libtpu
